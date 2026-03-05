# VEMamba: Efficient Isotropic Reconstruction of Volume Electron Microscopy with Axial-Lateral Consistent Mamba

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

<p align="center">
<img src="figs/main.png" width="85%">
</p>

*Figure: Overall architecture of VEMamba.*
---

# 🔥 News

- **2026.03** Paper released on arXiv  
- **2026.03** Code released  

---

# 🧠 Method Overview

<p align="center">
<img src="figs/block.png" width="85%">
</p>

*Figure: The Detail of VEMamba Module.*

The VEMamba pipeline contains four stages:

1. **Shallow feature extraction**
2. **Degradation representation learning (MoCo)**
3. **Deep feature extraction with Residual Volume Mamba Groups**
4. **Reconstruction module**

The core component is the **VEMamba Module (VEMM)**:

- ALCSSM → multi-directional 3D dependency scanning  
- SSM → global dependency modeling  
- DWAM → dynamic feature aggregation

---

# 📂 Dataset

We evaluate our method on two public VEM datasets.

## EPFL Dataset

https://www.epfl.ch/labs/cvlab/data/data-em/

- FIB-SEM hippocampus dataset
- Resolution: **5×5×5 nm**
- Contains annotated mitochondria segmentation labels. :contentReference[oaicite:2]{index=2}  

---

## CREMI Dataset

https://cremi.org/data/

- ssTEM dataset of *Drosophila melanogaster* brain
- Resolution: **4×4×40 nm**
- Contains three training volumes (A, B, C). :contentReference[oaicite:3]{index=3}  

---

---

# ⚙️ Environment

Recommended environment:

```

python = 3.10
torch = 2.4.0
causal_conv1d = 1.5.2
mamba_ssm = 2.2.5

````

Install dependencies:

```bash
pip install -r requirements.txt
````

⚠️ To avoid environment conflicts, we recommend installing the following packages manually.

**causal_conv1d**

```
https://github.com/Dao-AILab/causal-conv1d/releases
```

**mamba_ssm**

```
https://github.com/state-spaces/mamba/releases
```

---

# 🚀 Training
Training consists of **two stages**.

---

## Stage 1: Degradation Learning (MoCo)

Train the MoCo encoder to learn degradation representations.

```bash
python train_moco.py
```

---

## Stage 2: Reconstruction Training

Freeze the MoCo encoder and train the reconstruction backbone.

```bash
python train.py
```

---

# 🧪 Testing

To reconstruct the full volume:

```bash
python test.py
```

The reconstructed isotropic volume will be saved in the output directory.


# 🙏Acknowledgements

This project is built upon the excellent work from the following open-source repository:

- [SCST](https://github.com/ssj9596/SCST)
- [IsoVEM](https://github.com/cbmi-group/IsoVEM)
- [MambaIR](https://github.com/csguoh/MambaIR)
- [CDFormer](https://github.com/I2-Multimedia-Lab/CDFormer)


We thank the authors for making their code publicly available.

---

# 📚Citation
If you find this work useful, please cite:

```bibtex
@article{gao2026vemamba,
  title={VEMamba: Efficient Isotropic Reconstruction of Volume Electron Microscopy with Axial-Lateral Consistent Mamba},
  author={Gao, Longmi and Gao, Pan},
  journal={arXiv preprint arXiv:2603.00887},
  year={2026}
}
```
