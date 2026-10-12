# Robustness of Multi-Scale Implicit Feature Networks Under Sparse and Incomplete Point Clouds

> **Built on top of:** [IF-Net](https://github.com/jchibane/if-net) (Chibane et al., CVPR 2020)  
> **Course:** AI6131 3D Deep Learning — Nanyang Technological University  
> **Author:** Chang Yuan-Lun  

---

## Overview

This repository extends [IF-Net](https://github.com/jchibane/if-net) with a systematic robustness study under controlled point cloud degradation, and proposes **scale dropout** — a lightweight training-time regularisation strategy that improves reconstruction quality under severe input sparsity.

IF-Net achieves strong results on dense inputs by extracting multi-scale 3D feature grids at six resolution levels. However, it has only been evaluated on clean, dense point clouds. This work asks: *how does IF-Net's multi-scale feature hierarchy behave when the input point cloud is severely sparse or incomplete?*

### Key Findings

- Baseline IF-Net suffers **catastrophic degradation** at 300-point input at 128³ resolution (Chamfer Distance 2.1× worse than at 3000 points)
- **Scale dropout** achieves **4× lower Chamfer Distance** than baseline at 300-point input — by forcing the decoder to rely on coarser, sparsity-robust feature scales
- The failure mode is **resolution-dependent**: it does not manifest at 32³ where sparse inputs still populate high-resolution feature scales adequately

---

## Novel Contributions

| File | Description |
|---|---|
| `models/local_model_dropout.py` | Scale dropout model — identical architecture to baseline with Bernoulli scale masking at training time |
| `data_processing/convert_to_scaled_off_trimesh.py` | Trimesh-based mesh converter replacing the original `meshlabserver` dependency |
| `data_processing/fps_pointcloud_sampling.py` | Farthest Point Sampling for controlled sparsity degradation study |
| `evaluate_simple.py` | Grid-based IoU + Chamfer Distance evaluation without marching cubes |
| `evaluate_degradation.py` | Systematic evaluation across multiple input densities |
| `plot_loss_curves.py` | Training curve visualisation utilities |
| `visualize_results.py` | Reconstruction visualisation across models and input densities |

**Modified files** (relative to original IF-Net): `train.py`, `models/training.py`, `models/data/voxelized_data_shapenet.py`, `data_processing/boundary_sampling.py`, `data_processing/voxelized_pointcloud_sampling.py`, `generate.py`

---

## Results (128³, 500 epochs, ShapeNet Chairs)

| Model | CD @ 3000pts ↓ | CD @ 1000pts ↓ | CD @ 300pts ↓ |
|---|---|---|---|
| IF-Net Baseline | 0.060 | 0.046 | 0.128 |
| IF-Net + Scale Dropout (p=0.1) | 0.079 | 0.070 | **0.032** |

Scale dropout trades a small accuracy penalty at dense input for a 4× improvement at the most challenging sparse setting — the regime most relevant to real-world deployment.

---

## Relationship to IF-Net

This project is built directly on top of the original IF-Net codebase. The core architecture (`models/local_model.py`), data pipeline, and training loop are derived from [IF-Net](https://github.com/jchibane/if-net) (Chibane et al., CVPR 2020) with the modifications listed above.

All original IF-Net code remains under its original license. Our additions are released for academic use.

---

## Setup

### Requirements
- Python 3.8
- CUDA 11.3
- PyTorch 1.10.1

```bash
git clone https://github.com/cccku/if-net-robustness.git
cd if-net-robustness

# Create virtual environment
python3 -m venv ifnet38
source ifnet38/bin/activate

# Install PyTorch with CUDA 11.3
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# Install other dependencies
pip install numpy==1.24.4 scipy trimesh matplotlib pyyaml tqdm tensorboard

# Compile C extensions
cd data_processing/libvoxelize && python setup.py build_ext --inplace && cd ../..
cd data_processing/libmesh && python setup.py build_ext --inplace && cd ../..
```

---

## Data

Download [ShapeNetCore.v1](https://shapenet.org/) and symlink the chair category (03001627):

```bash
ln -s /path/to/ShapeNetCore.v1/03001627 shapenet/data/03001627
```

---

## Preprocessing

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH

# 1. Convert meshes to scaled .off (trimesh-based, no meshlabserver needed)
python data_processing/convert_to_scaled_off_trimesh.py

# 2. Boundary sampling for occupancy supervision
python data_processing/boundary_sampling.py -sigma 0.15
python data_processing/boundary_sampling.py -sigma 0.015

# 3. Voxelized input at 128³ with 3000-point dense input
python data_processing/voxelized_pointcloud_sampling.py -res 128 -num_points 3000

# 4. FPS subsampling for degradation study
python data_processing/fps_pointcloud_sampling.py -num_points 1000
python data_processing/fps_pointcloud_sampling.py -num_points 300
```

---

## Training

```bash
# Baseline IF-Net
python train.py -pointcloud -pc_samples 3000 -res 128 -batch_size 8 \
    -m ShapeNetPoints

# IF-Net + Scale Dropout (p=0.1)
python train.py -pointcloud -pc_samples 3000 -res 128 -batch_size 8 \
    -m ShapeNetPointsDropout -dropout_p 0.1

# Ablation over dropout probability (32³, faster)
python train.py -pointcloud -pc_samples 3000 -res 32 -batch_size 12 \
    -m ShapeNetPointsDropout -dropout_p 0.05
```

---

## Evaluation

```bash
# Evaluate on dense input (3000 pts)
python evaluate_simple.py \
    -model_type ShapeNetPoints \
    -checkpoint experiments/.../checkpoint_epoch_499.tar \
    -num_points 3000 -num_samples 200

# Evaluate on sparse input (FPS to 300 pts)
python evaluate_simple.py \
    -model_type ShapeNetPoints \
    -checkpoint experiments/.../checkpoint_epoch_499.tar \
    -num_points 300 -use_fps -num_samples 200

# Full degradation sweep (3000 / 1000 / 300 pts)
python evaluate_degradation.py \
    -checkpoint experiments/.../checkpoint_epoch_499.tar \
    -num_samples 200
```

---

## Citation

If you use this work, please also cite the original IF-Net paper on which it is based:

```bibtex
@inproceedings{chibane2020implicit,
  title     = {Implicit Functions in Feature Space for 3D Shape Reconstruction and Completion},
  author    = {Chibane, Julian and Alldieck, Thiemo and Pons-Moll, Gerard},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2020}
}
```

---

## License

This project follows the license of the original [IF-Net](https://github.com/jchibane/if-net) repository. Our novel contributions (scale dropout, FPS pipeline, trimesh converter, evaluation scripts) are released for academic and research use.
