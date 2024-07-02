import json
import os
import pathlib
from collections import defaultdict
from typing import Optional, Union

import numpy as np
import plotly.graph_objects as go
import pytorch3d.ops
import pytorch3d.structures
import pytorch_kinematics as pk
import torch
import transforms3d
import trimesh as tm
from get_a_grip.dataset_generation.utils.allegro_hand_info import (
    ALLEGRO_HAND_CONTACT_POINTS_PATH,
    ALLEGRO_HAND_FINGER_KEYWORDS,
    ALLEGRO_HAND_PENETRATION_POINTS_PATH,
    ALLEGRO_HAND_URDF_PATH,
)
from get_a_grip.dataset_generation.utils.rot6d import (
    robust_compute_rotation_matrix_from_ortho6d,
)
from kaolin.metrics.trianglemesh import (
    CUSTOM_index_vertices_by_faces as index_vertices_by_faces,
)
from kaolin.metrics.trianglemesh import (
    compute_sdf,
)
from urdf_parser_py.urdf import Box, Robot, Sphere


class HandModel:
    def __init__(
        self,
        n_surface_points: int = 0,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """
        Create a Hand Model for a MJCF robot

        Parameters
        ----------
        n_surface_points: int
            number of surface points to sample
        device: str | torch.Device
            device for torch tensors
        """
        self.n_surface_points = n_surface_points
        self.device = device

        # load articulation
        # should create:
        #   * self.chain: pytorch_kinematics.Chain
        #   * self.mesh dict with link_name keys and dict values with:
        #       * vertices: (N, 3) torch.FloatTensor
        #       * faces: (N, 3) torch.LongTensor
        #       * contact_candidates: (M, 3) torch.FloatTensor
        #       * penetration_keypoints: (K, 3) torch.FloatTensor
        #       * surface_points: (S, 3) torch.FloatTensor
        #       * some others that are type and link specific
        #   * self.areas dict with link_name keys and float values
        #   * self.n_dofs: int
        #   * self.joints_upper: (D,) torch.FloatTensor
        #   * self.joints_lower: (D,) torch.FloatTensor
        self._init_allegro(n_surface_points=n_surface_points)

        # indexing
        self.link_name_to_link_index = dict(
            zip([link_name for link_name in self.mesh], range(len(self.mesh)))
        )

        self.link_name_to_contact_candidates = {
            link_name: self.mesh[link_name]["contact_candidates"]
            for link_name in self.mesh
        }
        contact_candidates = [
            self.link_name_to_contact_candidates[link_name] for link_name in self.mesh
        ]
        self.global_index_to_link_index = sum(
            [
                [i] * len(contact_candidates)
                for i, contact_candidates in enumerate(contact_candidates)
            ],
            [],
        )
        self.link_index_to_global_indices = defaultdict(list)
        for global_idx, link_idx in enumerate(self.global_index_to_link_index):
            self.link_index_to_global_indices[link_idx].append(global_idx)

        self.contact_candidates = torch.cat(contact_candidates, dim=0)
        self.global_index_to_link_index = torch.tensor(
            self.global_index_to_link_index, dtype=torch.long, device=device
        )
        self.n_contact_candidates = self.contact_candidates.shape[0]

        self.penetration_keypoints = [
            self.mesh[link_name]["penetration_keypoints"] for link_name in self.mesh
        ]
        self.global_index_to_link_index_penetration = sum(
            [
                [i] * len(penetration_keypoints)
                for i, penetration_keypoints in enumerate(self.penetration_keypoints)
            ],
            [],
        )
        self.penetration_keypoints = torch.cat(self.penetration_keypoints, dim=0)
        self.global_index_to_link_index_penetration = torch.tensor(
            self.global_index_to_link_index_penetration, dtype=torch.long, device=device
        )
        self.n_keypoints = self.penetration_keypoints.shape[0]

        # parameters
        self.hand_pose = None
        self.contact_point_indices = None
        self.global_translation = None
        self.global_rotation = None
        self.current_status = None
        self.contact_points = None

    def _init_allegro(
        self,
        urdf_path: pathlib.Path = ALLEGRO_HAND_URDF_PATH,
        contact_points_path: pathlib.Path = ALLEGRO_HAND_CONTACT_POINTS_PATH,
        penetration_points_path: pathlib.Path = ALLEGRO_HAND_PENETRATION_POINTS_PATH,
        n_surface_points: int = 0,
    ) -> None:
        device = self.device

        self.chain = pk.build_chain_from_urdf(open(urdf_path).read()).to(
            dtype=torch.float, device=device
        )
        robot = Robot.from_xml_file(urdf_path)
        self.n_dofs = len(self.chain.get_joint_parameter_names())

        penetration_points = json.load(open(penetration_points_path, "r"))
        contact_points = json.load(open(contact_points_path, "r"))

        self.mesh = {}
        self.areas = {}
        for link in robot.links:
            if link.visual is None or link.collision is None:
                continue
            self.mesh[link.name] = {}

            # load collision mesh
            collision = link.collision
            if isinstance(collision.geometry, Sphere):
                link_mesh = tm.primitives.Sphere(radius=collision.geometry.radius)
                self.mesh[link.name]["radius"] = collision.geometry.radius
            elif isinstance(collision.geometry, Box):
                # link_mesh = tm.primitives.Box(extents=collision.geometry.size)
                link_mesh = tm.load_mesh(
                    os.path.join(os.path.dirname(urdf_path), "meshes", "box.obj"),
                    process=False,
                )
                link_mesh.vertices *= np.array(collision.geometry.size) / 2
            else:
                raise ValueError(
                    f"Unknown collision geometry: {type(collision.geometry)}"
                )
            vertices = torch.tensor(
                link_mesh.vertices, dtype=torch.float, device=device
            )
            faces = torch.tensor(link_mesh.faces, dtype=torch.long, device=device)
            if (
                hasattr(collision.geometry, "scale")
                and collision.geometry.scale is None
            ):
                collision.geometry.scale = [1, 1, 1]
            scale = torch.tensor(
                getattr(collision.geometry, "scale", [1, 1, 1]),
                dtype=torch.float,
                device=device,
            )
            translation = torch.tensor(
                getattr(collision.origin, "xyz", [0, 0, 0]),
                dtype=torch.float,
                device=device,
            )
            rotation = torch.tensor(
                transforms3d.euler.euler2mat(
                    *getattr(collision.origin, "rpy", [0, 0, 0])
                ),
                dtype=torch.float,
                device=device,
            )
            vertices = vertices * scale
            vertices = vertices @ rotation.T + translation
            self.mesh[link.name].update(
                {
                    "vertices": vertices,
                    "faces": faces,
                }
            )
            if "radius" not in self.mesh[link.name]:
                self.mesh[link.name]["face_verts"] = index_vertices_by_faces(
                    vertices, faces
                )

            # load visual mesh
            visual = link.visual
            filename = os.path.join(
                os.path.dirname(os.path.dirname(urdf_path)),
                visual.geometry.filename[10:],
            )
            link_mesh = tm.load_mesh(filename)
            visual_vertices = torch.tensor(
                link_mesh.vertices, dtype=torch.float, device=device
            )
            visual_faces = torch.tensor(
                link_mesh.faces, dtype=torch.long, device=device
            )
            if hasattr(visual.geometry, "scale") and visual.geometry.scale is None:
                visual.geometry.scale = [1, 1, 1]
            visual_scale = torch.tensor(
                getattr(visual.geometry, "scale", [1, 1, 1]),
                dtype=torch.float,
                device=device,
            )
            visual_translation = torch.tensor(
                getattr(visual.origin, "xyz", [0, 0, 0]),
                dtype=torch.float,
                device=device,
            )
            visual_rotation = torch.tensor(
                transforms3d.euler.euler2mat(*getattr(visual.origin, "rpy", [0, 0, 0])),
                dtype=torch.float,
                device=device,
            )
            visual_vertices = visual_vertices * visual_scale
            visual_vertices = visual_vertices @ visual_rotation.T + visual_translation
            self.mesh[link.name].update(
                {
                    "visual_vertices": visual_vertices,
                    "visual_faces": visual_faces,
                }
            )
            # load contact candidates and penetration keypoints
            contact_candidates = torch.tensor(
                contact_points[link.name], dtype=torch.float32, device=device
            ).reshape(-1, 3)
            penetration_keypoints = torch.tensor(
                penetration_points[link.name], dtype=torch.float32, device=device
            ).reshape(-1, 3)
            self.mesh[link.name].update(
                {
                    "contact_candidates": contact_candidates,
                    "penetration_keypoints": penetration_keypoints,
                }
            )
            self.areas[link.name] = tm.Trimesh(
                vertices.cpu().numpy(), faces.cpu().numpy()
            ).area.item()

        self.joints_lower = torch.tensor(
            [
                joint.limit.lower
                for joint in robot.joints
                if joint.joint_type == "revolute"
            ],
            dtype=torch.float,
            device=device,
        )
        self.joints_upper = torch.tensor(
            [
                joint.limit.upper
                for joint in robot.joints
                if joint.joint_type == "revolute"
            ],
            dtype=torch.float,
            device=device,
        )

        self._sample_surface_points(n_surface_points)

    def _sample_surface_points(self, n_surface_points: int) -> None:
        device = self.device

        total_area = sum(self.areas.values())
        num_samples = dict(
            [
                (link_name, int(self.areas[link_name] / total_area * n_surface_points))
                for link_name in self.mesh
            ]
        )
        num_samples[list(num_samples.keys())[0]] += n_surface_points - sum(
            num_samples.values()
        )
        for link_name in self.mesh:
            if num_samples[link_name] == 0:
                self.mesh[link_name]["surface_points"] = torch.tensor(
                    [], dtype=torch.float, device=device
                ).reshape(0, 3)
                continue
            mesh = pytorch3d.structures.Meshes(
                self.mesh[link_name]["vertices"].unsqueeze(0),
                self.mesh[link_name]["faces"].unsqueeze(0),
            )
            dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(
                mesh, num_samples=100 * num_samples[link_name]
            )
            surface_points = pytorch3d.ops.sample_farthest_points(
                dense_point_cloud, K=num_samples[link_name]
            )[0][0]
            surface_points.to(dtype=float, device=device)
            self.mesh[link_name]["surface_points"] = surface_points

    def sample_contact_points(
        self, total_batch_size: int, n_contacts_per_finger: int
    ) -> torch.Tensor:
        # Ensure that each finger gets sampled at least once
        # Goal: Output (B, n_fingers * n_contacts_per_finger) torch.LongTensor of sampled contact point indices
        # Each contact point is represented by a global index
        # Each contact point is sampled from a link
        # For each finger:
        #    Get the link indices that contain the finger keyword
        #    Get the possible contact point indices from these link indices
        #    Sample from these contact point indices

        finger_keywords = ALLEGRO_HAND_FINGER_KEYWORDS

        # Get link indices that contain the finger keyword
        finger_possible_link_idxs_list = [
            [
                link_idx
                for link_name, link_idx in self.link_name_to_link_index.items()
                if finger_keyword in link_name
            ]
            for finger_keyword in finger_keywords
        ]

        # Get the possible contact point indices from these link indices
        finger_possible_contact_point_idxs_list = [
            sum(
                [self.link_index_to_global_indices[link_idx] for link_idx in link_idxs],
                [],
            )
            for link_idxs in finger_possible_link_idxs_list
        ]

        # Sample from these contact point indices
        sampled_contact_point_idxs_list = []
        for (
            finger_possible_contact_point_idxs
        ) in finger_possible_contact_point_idxs_list:
            sampled_idxs = torch.randint(
                len(finger_possible_contact_point_idxs),
                size=[total_batch_size, n_contacts_per_finger],
                device=self.device,
            )
            sampled_contact_point_idxs = torch.tensor(
                finger_possible_contact_point_idxs, device=self.device, dtype=torch.long
            )[sampled_idxs]
            sampled_contact_point_idxs_list.append(sampled_contact_point_idxs)
        sampled_contact_point_idxs_list = torch.cat(
            sampled_contact_point_idxs_list, dim=1
        )

        assert sampled_contact_point_idxs_list.shape == (
            total_batch_size,
            len(finger_keywords) * n_contacts_per_finger,
        )

        return sampled_contact_point_idxs_list

    def set_parameters(
        self,
        hand_pose: torch.Tensor,
        contact_point_indices: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Set translation, rotation, joint angles, and contact points of grasps

        Parameters
        ----------
        hand_pose: (B, 3+6+`n_dofs`) torch.FloatTensor
            translation, rotation in rot6d, and joint angles
        contact_point_indices: (B, `n_contact`) [Optional]torch.LongTensor
            indices of contact candidates
        """
        assert len(hand_pose.shape) <= 2
        if len(hand_pose.shape) == 1:
            hand_pose = hand_pose.unsqueeze(0)
        assert hand_pose.shape[1] == 3 + 6 + self.n_dofs

        self.hand_pose = hand_pose
        if self.hand_pose.requires_grad:
            self.hand_pose.retain_grad()
        self.global_translation = self.hand_pose[:, 0:3]
        self.global_rotation = robust_compute_rotation_matrix_from_ortho6d(
            self.hand_pose[:, 3:9]
        )
        self.current_status = self.chain.forward_kinematics(self.hand_pose[:, 9:])
        if contact_point_indices is not None:
            self.contact_point_indices = contact_point_indices
            batch_size, n_contact = contact_point_indices.shape
            self.contact_points = self.contact_candidates[self.contact_point_indices]
            link_indices = self.global_index_to_link_index[self.contact_point_indices]
            transforms = torch.zeros(
                batch_size, n_contact, 4, 4, dtype=torch.float, device=self.device
            )
            for link_name in self.mesh:
                mask = link_indices == self.link_name_to_link_index[link_name]
                cur = (
                    self.current_status[link_name]
                    .get_matrix()
                    .unsqueeze(1)
                    .expand(batch_size, n_contact, 4, 4)
                )
                transforms[mask] = cur[mask]
            self.contact_points = torch.cat(
                [
                    self.contact_points,
                    torch.ones(
                        batch_size, n_contact, 1, dtype=torch.float, device=self.device
                    ),
                ],
                dim=2,
            )
            self.contact_points = (transforms @ self.contact_points.unsqueeze(3))[
                :, :, :3, 0
            ]
            self.contact_points = self.contact_points @ self.global_rotation.transpose(
                1, 2
            ) + self.global_translation.unsqueeze(1)

    def cal_distance(self, x: torch.Tensor) -> torch.Tensor:
        return self._cal_distance_allegro(x)

    def _cal_distance_allegro(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the distance from each point in `x` to the hand model.

        Args:
            x (torch.Tensor): Input array of shape (total_batch_size, num_samples, 3).

        Procedure:
        1. Transform `x` to the local coordinate system of each link to obtain `x_local` with shape (total_batch_size, num_samples, 3).
        2. Compute the distance `dis` for each point in `x_local`, assigning positive values for points inside the link.
        3. The final `dis` is the maximum distance among all links.
        4. For links modeled as spheres, use an analytical method to compute the distance. For other links, use a mesh-based method.

        Returns:
            torch.Tensor: The computed distances for each point in `x`.
        """
        dis = []
        x = (x - self.global_translation.unsqueeze(1)) @ self.global_rotation
        for link_name in self.mesh:
            matrix = self.current_status[link_name].get_matrix()
            x_local = (x - matrix[:, :3, 3].unsqueeze(1)) @ matrix[:, :3, :3]
            x_local = x_local.reshape(-1, 3)  # (total_batch_size * num_samples, 3)
            if "radius" not in self.mesh[link_name]:
                face_verts = self.mesh[link_name]["face_verts"]
                dis_local, dis_signs, _, _ = compute_sdf(x_local, face_verts)
                dis_local = torch.sqrt(dis_local + 1e-8)
                dis_local = dis_local * (-dis_signs)
            else:
                dis_local = self.mesh[link_name]["radius"] - x_local.norm(dim=1)
            dis.append(dis_local.reshape(x.shape[0], x.shape[1]))
        dis = torch.max(torch.stack(dis, dim=0), dim=0)[0]
        return dis

    def cal_self_penetration_energy(self) -> torch.Tensor:
        """
        Calculate self penetration energy

        Returns
        -------
        E_spen: (N,) torch.Tensor
        """
        batch_size = self.global_translation.shape[0]
        points = self.penetration_keypoints.clone().repeat(batch_size, 1, 1)
        link_indices = self.global_index_to_link_index_penetration.clone().repeat(
            batch_size, 1
        )
        transforms = torch.zeros(
            batch_size, self.n_keypoints, 4, 4, dtype=torch.float, device=self.device
        )
        for link_name in self.mesh:
            mask = link_indices == self.link_name_to_link_index[link_name]
            cur = (
                self.current_status[link_name]
                .get_matrix()
                .unsqueeze(1)
                .expand(batch_size, self.n_keypoints, 4, 4)
            )
            transforms[mask] = cur[mask]
        points = torch.cat(
            [
                points,
                torch.ones(
                    batch_size,
                    self.n_keypoints,
                    1,
                    dtype=torch.float,
                    device=self.device,
                ),
            ],
            dim=2,
        )
        points = (transforms @ points.unsqueeze(3))[:, :, :3, 0]
        points = points @ self.global_rotation.transpose(
            1, 2
        ) + self.global_translation.unsqueeze(1)
        dis = (points.unsqueeze(1) - points.unsqueeze(2) + 1e-13).square().sum(3).sqrt()
        dis = torch.where(dis < 1e-6, 1e6 * torch.ones_like(dis), dis)
        dis = 0.02 - dis
        E_spen = torch.where(dis > 0, dis, torch.zeros_like(dis))
        return E_spen.sum((1, 2))

    def cal_joint_limit_energy(self) -> torch.Tensor:
        """
        Calculate joint limit energy

        Returns:
        E_joints: (N,) torch.Tensor
        """
        joint_limit_energy = torch.sum(
            (self.hand_pose[:, 9:] > self.joints_upper)
            * (self.hand_pose[:, 9:] - self.joints_upper),
            dim=-1,
        ) + torch.sum(
            (self.hand_pose[:, 9:] < self.joints_lower)
            * (self.joints_lower - self.hand_pose[:, 9:]),
            dim=-1,
        )
        return joint_limit_energy

    def cal_finger_finger_distance_energy(self) -> torch.Tensor:
        """
        Calculate finger-finger distance energy

        Returns
        -------
        E_ff: (N,) torch.Tensor
        """
        batch_size = self.contact_points.shape[0]
        finger_finger_distance_energy = (
            -torch.cdist(self.contact_points, self.contact_points, p=2)
            .reshape(batch_size, -1)
            .sum(dim=-1)
        )
        return finger_finger_distance_energy

    def cal_finger_palm_distance_energy(self) -> torch.Tensor:
        """
        Calculate finger-palm distance energy

        Returns
        -------
        E_fp: (N,) torch.Tensor
        """
        palm_position = self.global_translation[:, None, :]
        palm_finger_distance_energy = (
            -(palm_position - self.contact_points).norm(dim=-1).sum(dim=-1)
        )
        return palm_finger_distance_energy

    def cal_table_penetration(
        self, table_pos: torch.Tensor, table_normal: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate table penetration energy

        Args
        ----
        table_pos: (B, 3) torch.Tensor
            position of table surface
        table_normal: (B, 3) torch.Tensor
            normal of table

        Returns
        -------
        E_tpen: (B,) torch.Tensor
            table penetration energy
        """
        # Two methods: use sampled points or meshes
        B1, D1 = table_pos.shape
        B2, D2 = table_normal.shape
        assert B1 == B2
        assert D1 == D2 == 3

        sampled_points_world_frame = self.get_surface_points()
        B, N, D = sampled_points_world_frame.shape
        assert B == B1
        assert D == 3

        # Positive = above table, negative = below table
        signed_distance_from_table = torch.sum(
            (sampled_points_world_frame - table_pos.unsqueeze(1))
            * table_normal.unsqueeze(1),
            dim=-1,
        )

        penetration = torch.clamp(signed_distance_from_table, max=0.0)
        penetration = -penetration
        assert penetration.shape == (B, N)

        return penetration.sum(-1)

    def get_surface_points(self) -> torch.Tensor:
        """
        Get surface points

        Returns
        -------
        points: (N, `n_surface_points`, 3)
        """
        points = []
        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            n_surface_points = self.mesh[link_name]["surface_points"].shape[0]
            points.append(
                self.current_status[link_name].transform_points(
                    self.mesh[link_name]["surface_points"]
                )
            )
            if 1 < batch_size != points[-1].shape[0]:
                points[-1] = points[-1].expand(batch_size, n_surface_points, 3)
        points = torch.cat(points, dim=-2).to(self.device)
        points = points @ self.global_rotation.transpose(
            1, 2
        ) + self.global_translation.unsqueeze(1)
        return points

    def get_contact_candidates(self) -> torch.Tensor:
        """
        Get all contact candidates

        Returns
        -------
        points: (N, `n_contact_candidates`, 3) torch.Tensor
        """
        points = []
        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            n_contact_candidates = self.mesh[link_name]["contact_candidates"].shape[0]
            points.append(
                self.current_status[link_name].transform_points(
                    self.mesh[link_name]["contact_candidates"]
                )
            )
            if 1 < batch_size != points[-1].shape[0]:
                points[-1] = points[-1].expand(batch_size, n_contact_candidates, 3)
        points = torch.cat(points, dim=-2).to(self.device)
        points = points @ self.global_rotation.transpose(
            1, 2
        ) + self.global_translation.unsqueeze(1)
        return points

    def get_penetration_keypoints(self) -> torch.Tensor:
        """
        Get penetration keypoints

        Returns
        -------
        points: (N, `n_keypoints`, 3) torch.Tensor
        """
        points = []
        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            n_keypoints = self.mesh[link_name]["penetration_keypoints"].shape[0]
            points.append(
                self.current_status[link_name].transform_points(
                    self.mesh[link_name]["penetration_keypoints"]
                )
            )
            if 1 < batch_size != points[-1].shape[0]:
                points[-1] = points[-1].expand(batch_size, n_keypoints, 3)
        points = torch.cat(points, dim=-2).to(self.device)
        points = points @ self.global_rotation.transpose(
            1, 2
        ) + self.global_translation.unsqueeze(1)
        return points

    def get_plotly_data(
        self,
        i: int,
        opacity: float = 0.5,
        color: str = "lightblue",
        with_contact_points: bool = False,
        with_contact_candidates: bool = False,
        with_surface_points: bool = False,
        with_penetration_keypoints: bool = False,
        pose: Optional[np.ndarray] = None,
        visual: bool = False,
    ) -> list:
        """
        Get visualization data for plotly.graph_objects

        Parameters
        ----------
        i: int
            index of data
        opacity: float
            opacity
        color: str
            color of mesh
        with_contact_points: bool
            whether to visualize contact points
        with_contact_candidates: bool
            whether to visualize contact candidates
        with_surface_points: bool
            whether to visualize surface points
        with_penetration_keypoints: bool
            whether to visualize penetration keypoints
        pose: (4, 4) matrix
            homogeneous transformation matrix
        visual: bool
            whether to visualize the hand with visual components

        Returns
        -------
        data: list
            list of plotly.graph_object visualization data
        """
        if pose is not None:
            pose = np.array(pose, dtype=np.float32)
        data = []
        for link_name in self.mesh:
            v = self.current_status[link_name].transform_points(
                self.mesh[link_name]["visual_vertices"]
                if visual and "visual_vertices" in self.mesh[link_name]
                else self.mesh[link_name]["vertices"]
            )
            if len(v.shape) == 3:
                v = v[i]
            v = v @ self.global_rotation[i].T + self.global_translation[i]
            v = v.detach().cpu()
            f = (
                (
                    self.mesh[link_name]["visual_faces"]
                    if visual and "visual_faces" in self.mesh[link_name]
                    else self.mesh[link_name]["faces"]
                )
                .detach()
                .cpu()
            )
            if pose is not None:
                v = v @ pose[:3, :3].T + pose[:3, 3]
            data.append(
                go.Mesh3d(
                    x=v[:, 0],
                    y=v[:, 1],
                    z=v[:, 2],
                    i=f[:, 0],
                    j=f[:, 1],
                    k=f[:, 2],
                    color=color,
                    opacity=opacity,
                    name="hand",
                )
            )
        if with_contact_points:
            contact_points = self.contact_points[i].detach().cpu()
            if pose is not None:
                contact_points = contact_points @ pose[:3, :3].T + pose[:3, 3]
            data.append(
                go.Scatter3d(
                    x=contact_points[:, 0],
                    y=contact_points[:, 1],
                    z=contact_points[:, 2],
                    mode="markers",
                    marker=dict(color="red", size=10),
                    name="contact points",
                )
            )
        if with_contact_candidates:
            contact_candidates = self.get_contact_candidates()[i].detach().cpu()
            if pose is not None:
                contact_candidates = contact_candidates @ pose[:3, :3].T + pose[:3, 3]
            data.append(
                go.Scatter3d(
                    x=contact_candidates[:, 0],
                    y=contact_candidates[:, 1],
                    z=contact_candidates[:, 2],
                    mode="markers",
                    marker=dict(color="blue", size=5),
                    name="contact candidates",
                )
            )
        if with_surface_points:
            surface_points = self.get_surface_points()[i].detach().cpu()
            if pose is not None:
                surface_points = surface_points @ pose[:3, :3].T + pose[:3, 3]
            data.append(
                go.Scatter3d(
                    x=surface_points[:, 0],
                    y=surface_points[:, 1],
                    z=surface_points[:, 2],
                    mode="markers",
                    marker=dict(color="green", size=2),
                    name="surface points",
                )
            )

        if with_penetration_keypoints:
            penetration_keypoints = self.get_penetration_keypoints()[i].detach().cpu()
            if pose is not None:
                penetration_keypoints = (
                    penetration_keypoints @ pose[:3, :3].T + pose[:3, 3]
                )
            data.append(
                go.Scatter3d(
                    x=penetration_keypoints[:, 0],
                    y=penetration_keypoints[:, 1],
                    z=penetration_keypoints[:, 2],
                    mode="markers",
                    marker=dict(color="red", size=3),
                    name="penetration_keypoints",
                )
            )
            for penetration_keypoint in penetration_keypoints:
                penetration_keypoint = penetration_keypoint.numpy()
                mesh = tm.primitives.Capsule(radius=0.01, height=0)
                v = mesh.vertices + penetration_keypoint
                f = mesh.faces
                data.append(
                    go.Mesh3d(
                        x=v[:, 0],
                        y=v[:, 1],
                        z=v[:, 2],
                        i=f[:, 0],
                        j=f[:, 1],
                        k=f[:, 2],
                        color="burlywood",
                        opacity=0.5,
                        name="penetration_keypoints_mesh",
                    )
                )

        return data

    def get_trimesh_data(self, i: int) -> tm.Trimesh:
        """
        Get full mesh

        Returns
        -------
        data: tm.Trimesh
        """
        data = tm.Trimesh()
        for link_name in self.mesh:
            v = self.current_status[link_name].transform_points(
                self.mesh[link_name]["vertices"]
            )
            if len(v.shape) == 3:
                v = v[i]
            v = v @ self.global_rotation[i].T + self.global_translation[i]
            v = v.detach().cpu()
            f = self.mesh[link_name]["faces"].detach().cpu()
            data += tm.Trimesh(vertices=v, faces=f)
        return data

    @property
    def n_fingers(self) -> int:
        return len(ALLEGRO_HAND_FINGER_KEYWORDS)

    @property
    def batch_size(self) -> int:
        if self.hand_pose is None:
            raise ValueError("Hand pose is not set")
        return self.hand_pose.shape[0]

    @property
    def num_fingers(self) -> int:
        return len(ALLEGRO_HAND_FINGER_KEYWORDS)
