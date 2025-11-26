# Multi-View Foundation Models

By *[Or Hirschorn](https://scholar.google.co.il/citations?user=GgFuT_QAAAAJ&hl=iw&oi=ao), *[Leo Segre](https://scholar.google.com/citations?user=A7FWhoIAAAAJ&hl=iw) and [Shai Avidan](https://scholar.google.co.il/citations?hl=iw&user=hpItE1QAAAAJ)  </p>

This repo is the official implementation of "[Multi-View Foundation Models](https://arxiv.org/pdf/TBD.pdf)" (Link TBD).

<p align="center">
<img src="images/framework.jpg" width="384">
</p>

## Introduction
We introduce a novel framework that transforms existing 2D Foundation Models (like DINO, SAM, and CLIP) into Multi-View Foundation Models. Current 2D models process images independently, leading to inconsistent feature representations for the same 3D point viewed from multiple camera angles.

## Setup/Install
We recommend using Anaconda or Miniconda. To setup the environment, follow the instructions below.
### Create environment
```bash
conda create --name multi_view_foundation_models -y python=3.10
conda activate multi_view_foundation_models
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
```
### Demo
1. First download the data from (TBD) and the model from (TBD).
2. Run the demo script:
```bash
python test/correspondence.py --exp_directory {path/to/exp/dir} --exp_name {exp_name} --colmap_path {path/to/data/root/dir} --fit3d --scene pikachu
```

### Training
Run the relevant experiment, for example for DINOv2:
```bash
python train/train_dino.py --exp_name {exp_name} --colmap_path {path/to/data/root/dir} --exp_directory {path/to/exp/dir} --config_name dinov2_reg.yaml
```
### Testing
```bash
python test/test_3d.py --exp_directory {exp_dir} --exp_name {exp_name} --colmap_path {path/to/data/root/dir} --results_dir {path/to/results/dir} --compare_to_base --fit3d
```


## BibTeX
If you find our models useful, please consider citing our paper!
```
TBD
```
