"""
Copied and modified from https://github.com/hsd1503/resnet1d/tree/master May 9, 2023

resnet for 1-d signal data, pytorch version
 
Shenda Hong, Oct 2019
"""

import numpy as np
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return (
            torch.tensor(self.data[index], dtype=torch.float),
            torch.tensor(self.label[index], dtype=torch.long),
        )

    def __len__(self):
        return len(self.data)


class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int = 1,
    ) -> None:
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=self.groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.conv(net)

        return net


class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """

    def __init__(self, kernel_size: int) -> None:
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.max_pool(net)

        return net


class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int,
        downsample: int,
        use_batchnorm: bool,
        use_dropout: bool,
        is_first_block: bool = False,
    ) -> None:
        super(BasicBlock, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            groups=self.groups,
        )

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            groups=self.groups,
        )

        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(
        self,
        x: torch.Tensor,
        beta: Optional[torch.Tensor] = None,
        gamma: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        identity = x

        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_batchnorm:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_dropout:
                out = self.do1(out)
        out = self.conv1(out)

        # the second conv
        if self.use_batchnorm:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_dropout:
            out = self.do2(out)
        out = self.conv2(out)

        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)

        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            ch1 = (self.out_channels - self.in_channels) // 2
            ch2 = self.out_channels - self.in_channels - ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1, -2)

        # ADD FILM START
        if gamma is not None:
            out = out * gamma
        if beta is not None:
            out = out + beta
        # ADD FILM END

        # shortcut
        out += identity

        return out


class ResNet1D(nn.Module):
    """

    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)

    Output:
        out: (n_samples)

    Pararmetes:
        in_channels: dim of input, the same as n_channel
        seq_len: length of input, the same as n_length
        base_filters: number of filters in the first several Conv layer, it will double at every increasefilter_gap layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: see Conv1d documentation
        n_block: number of blocks
        n_classes: number of classes
        downsample_gap: number of blocks before downsample (stride != 1)
        increasefilter_gap: number of blocks before increasing filters
        use_batchnorm: whether to use batch normalization or not
        use_dropout: whether to use dropout or not
        verbose: whether to output shape infomation
    """

    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        base_filters: int,
        kernel_size: int,
        stride: int,
        groups: int,
        n_block: int,
        n_classes: int,
        downsample_gap: int = 2,
        increasefilter_gap: int = 4,
        use_batchnorm: bool = True,
        use_dropout: bool = True,
        verbose: bool = False,
    ) -> None:
        super(ResNet1D, self).__init__()

        self.in_channels = in_channels
        self.seq_len = seq_len
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.n_block = n_block
        self.n_classes = n_classes
        self.downsample_gap = downsample_gap
        self.increasefilter_gap = increasefilter_gap
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.verbose = verbose

        # first block
        self.first_block_conv = MyConv1dPadSame(
            in_channels=in_channels,
            out_channels=base_filters,
            kernel_size=self.kernel_size,
            stride=1,
        )
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters

        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(
                    base_filters * 2 ** ((i_block - 1) // self.increasefilter_gap)
                )
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels

            tmp_block = BasicBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                groups=self.groups,
                downsample=downsample,
                use_batchnorm=self.use_batchnorm,
                use_dropout=self.use_dropout,
                is_first_block=is_first_block,
            )
            self.basicblock_list.append(tmp_block)

        # final prediction
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        # self.do = nn.Dropout(p=0.5)
        self.fc = nn.Linear(out_channels, n_classes)
        # self.softmax = nn.Softmax(dim=1)

        # COMPUTE OUTPUT SIZES FILM START
        example_x = torch.zeros(1, self.in_channels, self.seq_len)

        # Copy first part of forward pass
        example_x = self.first_block_conv(example_x)
        example_x = self.first_block_bn(example_x)
        example_x = self.first_block_relu(example_x)

        # Each block outputs a number of planes (related to the in_channels and base_filters)
        self.num_planes_per_block = []
        for i, block in enumerate(self.basicblock_list):
            if self.verbose:
                print(
                    f"block {i}, in_channels: {block.in_channels}, out_channels: {block.out_channels}, downsample: {block.downsample}"
                )

            # Perform forward pass and store output sizes
            example_x = block(example_x)
            num_planes = example_x.shape[1]
            self.num_planes_per_block.append(num_planes)

        # This is an important attribute to define how many FiLM params are needed
        self.num_film_params = sum(
            [num_planes for num_planes in self.num_planes_per_block]
        )
        # COMPUTE OUTPUT SIZES FILM END

        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(
        self,
        x: torch.Tensor,
        beta: Optional[torch.Tensor] = None,
        gamma: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        assert x.shape == (batch_size, self.in_channels, self.seq_len)
        out = x

        # first conv
        if self.verbose:
            print("input shape", out.shape)
        out = self.first_block_conv(out)
        if self.verbose:
            print("after first conv", out.shape)
        if self.use_batchnorm:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)

        # residual blocks, every block has two conv
        if beta is None and gamma is None:
            # Regular forward pass
            for i_block in range(self.n_block):
                net = self.basicblock_list[i_block]
                if self.verbose:
                    print(
                        "i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}".format(
                            i_block, net.in_channels, net.out_channels, net.downsample
                        )
                    )
                out = net(out)
                if self.verbose:
                    print(out.shape)
        else:
            # beta.shape == gamma.shape == (batch_size, num_film_params)
            start_idx = 0
            for i, block in enumerate(self.basicblock_list):
                num_planes = self.num_planes_per_block[i]
                end_idx = start_idx + num_planes

                # beta_i.shape == gamma_i.shape == (batch_size, num_planes, 1)
                beta_i = (
                    beta[:, start_idx:end_idx].reshape(-1, num_planes, 1)
                    if beta is not None
                    else None
                )
                gamma_i = (
                    gamma[:, start_idx:end_idx].reshape(-1, num_planes, 1)
                    if gamma is not None
                    else None
                )

                out = block(out, beta=beta_i, gamma=gamma_i)

                start_idx = end_idx

        # final prediction
        if self.use_batchnorm:
            out = self.final_bn(out)
        out = self.final_relu(out)

        # Custom pooling
        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        if self.verbose:
            print("final pooling", out.shape)
        # out = self.do(out)
        out = self.fc(out)
        if self.verbose:
            print("dense", out.shape)
        # out = self.softmax(out)
        if self.verbose:
            print("softmax", out.shape)

        # We replace the fc so won't be this shape
        # assert out.shape == (batch_size, self.n_classes)
        return out


if __name__ == "__main__":
    from torchinfo import summary

    # Create encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, in_channels, seq_len = 1, 3, 100
    verbose = False
    conv1d_encoder = ResNet1D(
        in_channels=in_channels,
        seq_len=seq_len,
        base_filters=64,
        kernel_size=3,
        stride=1,
        groups=1,
        n_block=4,
        n_classes=10,
        downsample_gap=2,
        increasefilter_gap=2,
        use_dropout=False,
        verbose=verbose,
    ).to(device)
    input_shape = (batch_size, in_channels, seq_len)
    print(f"Input shape: {input_shape}")
    print(f"Number of FiLM params: {conv1d_encoder.num_film_params}")
    print()

    # Summary comparison
    print("~" * 100)
    print("Summary of FiLM resnet 1d:")
    print("~" * 100)
    summary(conv1d_encoder, input_size=input_shape, depth=float("inf"), device=device)
    print()

    # Compare output with reference encoder
    example_input = torch.rand(*input_shape, device=device)
    example_output = conv1d_encoder(example_input)
    print(f"Output shape: {example_output.shape}")

    # Compare output with defined beta and gamma
    example_output_with_film = conv1d_encoder(
        example_input,
        beta=torch.zeros(batch_size, conv1d_encoder.num_film_params, device=device),
        gamma=torch.ones(batch_size, conv1d_encoder.num_film_params, device=device),
    )
    print(
        f"Output difference with defined beta=0 and gamma=1: {torch.norm(example_output - example_output_with_film)}"
    )
