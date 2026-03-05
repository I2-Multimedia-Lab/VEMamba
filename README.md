# VEMamba

<p align="center">

<a href="https://arxiv.org/abs/2603.00887">
<img src="https://img.shields.io/badge/arXiv-2603.00887-b31b1b.svg">
</a>

<a href="https://github.com/I2-Multimedia-Lab/VEMamba">
<img src="https://img.shields.io/badge/Code-GitHub-blue">
</a>

<a href="#">
<img src="https://img.shields.io/badge/Conference-CVPR%202026-green">
</a>

</p>

Official PyTorch implementation of  

**VEMamba: Efficient Isotropic Reconstruction of Volume Electron Microscopy with Axial-Lateral Consistent Mamba**

📄 Paper: https://arxiv.org/abs/2603.00887  
💻 Code: https://github.com/I2-Multimedia-Lab/VEMamba

---

# 🔥 News

- **2026.03** Paper released on arXiv  
- **2026.03** Code released  

---

# 📖 Introduction

Volume Electron Microscopy (VEM) enables high-resolution **3D visualization of biological tissues**, but many imaging techniques produce **anisotropic volumes** where the axial resolution is much lower than the lateral resolution. This anisotropy significantly affects visualization and downstream biological analysis. :contentReference[oaicite:0]{index=0}  

To address this issue, we propose **VEMamba**, an efficient framework for **isotropic reconstruction of VEM data**.

Our method introduces a **3D Dependency Reordering paradigm** based on state space models (Mamba), which efficiently captures volumetric spatial dependencies while maintaining low computational complexity. :contentReference[oaicite:1]{index=1}  

Key contributions include:

- **ALCSSM (Axial-Lateral Chunking Selective Scan Module)**  
  Reorders 3D spatial dependencies into optimized sequences for Mamba modeling.

- **DWAM (Dynamic Weights Aggregation Module)**  
  Adaptively aggregates multi-directional scan features.

- **MoCo-based degradation learning**  
  Learns degradation-aware representations to better simulate realistic anisotropic EM data.

Extensive experiments demonstrate that **VEMamba achieves strong reconstruction performance with significantly lower computational cost**.

---

## Dataset Download
- EPFL Dataset : https://www.epfl.ch/labs/cvlab/data/data-em/
- CREMI Dataset : https://cremi.org/data/
## Environment
- python = 3.10
- torch = 2.4.0
- causal_conv1d = 1.5.2
- mamba_ssm = 2.2.5


Then use pip install -r requirements.txt

Note: To avoid environment conflicts, it is recommended to install the casual_conv1d and mamba_ssm libraries locally.

casual_conv1d : https://github.com/Dao-AILab/causal-conv1d/releases

mamba_ssm : https://github.com/state-spaces/mamba/releases

## Train
stage1 : train moco to learn degradation

train_moco.py

stage2 : frozen moco, train backbone

train.py

## Test

run test.py to generate the whole volume


## Acknowledgements

This project is built upon the excellent work from the following open-source repository:

- [SCST](https://github.com/ssj9596/SCST)
- [IsoVEM](https://github.com/cbmi-group/IsoVEM)
- [MambaIR](https://github.com/csguoh/MambaIR)
- [CDFormer](https://github.com/I2-Multimedia-Lab/CDFormer)



We thank the authors for making their code publicly available.  
Our implementation is based on their work with several modifications and extensions.

## Citation


