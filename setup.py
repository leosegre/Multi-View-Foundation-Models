from setuptools import setup, find_packages

setup(
    name="Multi-View-Foundation-Models",
    packages=find_packages(include=("dino3d*", "eval_probe3d*", "FiT3D*")),
    version="0.0.2",
    author="Leo Segre and Or Hirschorn",
    install_requires=open("requirements.txt", "r").read().split("\n"),
)
