from pathlib import Path

from setuptools import find_packages, setup

VERSION = "0.1.0"
DESCRIPTION = "Official implementation of Get a Grip: Multi-Finger Grasp Evaluation at Scale Enables Robust Sim-to-Real Transfer"
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="get_a_grip",
    version=VERSION,
    author="Tyler Lum",
    author_email="tylergwlum@gmail.com",
    url="https://github.com/tylerlum/get_a_grip",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[],
    keywords=["robotics", "grasping", "simulation", "evaluation"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
    # python_requires=">=3.7, <3.9",  # Comment this out because streamlit no longer support Python 3.8 (https://docs.streamlit.io/knowledge-base/using-streamlit/sanity-checks)
)
