# Self-Supervised AI-Generated Image Detection: A Camera Metadata Perspective

[![Journal](https://img.shields.io/badge/Journal-IEEE%20TPAMI-blue.svg)](https://arxiv.org/abs/2512.05651)


Official implementation of the paper "**Self-Supervised AI-Generated Image Detection: A Camera Metadata Perspective**".

⭐️Our series work: [BLADES (ICCV'25 Highlight)](https://github.com/MZMMSEC/AIGFD_BLO)

## 🆕 News
Our paper has been accepted for publication in **IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)** 🎉🎉🎉

## 📜 Abstract

The proliferation of AI-generated imagery poses escalating challenges for multimedia forensics, yet many existing detectors depend on assumptions about the internals of specific generative models, limiting their cross-model applicability. We introduce a self-supervised approach for detecting AI-generated images that leverages camera metadata---specifically
exchangeable image file format (EXIF) tags---to learn features intrinsic to digital photography. Our pretext task trains a feature extractor solely on camera-captured photographs by classifying categorical EXIF tags (\eg, camera model and scene type) and pairwise-ranking ordinal and continuous EXIF tags (\eg, focal length and aperture value). Using these EXIF-induced features, we first perform one-class detection by modeling the distribution of photographic images with a Gaussian mixture model and flagging low-likelihood samples as AI-generated. We then extend to binary detection that treats the learned extractor as a strong regularizer for a classifier of the same architecture, operating on high-frequency residuals from spatially scrambled patches. Extensive experiments across various generative models demonstrate that our EXIF-induced detectors substantially advance the state of the art, delivering strong generalization to in-the-wild samples and robustness to common benign image perturbations.

---

### Key Contributions:
* A self-supervised pretext task that leverages EXIF tags to learn camera-intrinsic features from photographs only.
* A feature extractor that operates on high-frequency residuals of scrambled patches to suppress semantics and accentuate imaging regularities.
* A one-class detector that models photographic features with a GMM and detects anomalies without seeing AI-generated images during training.
* A binary detector that uses the pretext extractor as a strong regularizer, improving generalization and robustness across generators and post-processing.

---


## 🛠️ Environment Configuration
This project recommends using uv for fast package management:
```bash
uv sync
```
## 📂 Dataset Description & Downloads
| Parameter | Description | Resource Link |
| :--- | :--- | :--- |
| `-exif_image_path` | Images with EXIF metadata required for backbone training | [Download](https://drive.google.com/drive/folders/1-xV41wjdorr0tl8vI9q6l8zsBJHBvfpq?usp=sharing) |
| `-test_image_path` | Test sets | [Download](https://drive.google.com/file/d/1ptbEdfbovwS_31ElHUeKIPMas764Hye7/view?usp=sharing) |
| `-oc_realonly_image_path` | One-class training set (contains only ImageNet/LSUN real images) | [Download](https://drive.google.com/file/d/1sl7qBAA5kE2uFFTSPztvihmBXLbmvbPY/view?usp=sharing) |
| `-bc_trainset_path` | Training dataset for binary classification | [Download](https://drive.google.com/file/d/1tI6zdhRV3PCimnkDmy3ikPzSQ_Ya8-Rp/view?usp=sharing) |

## 🚀 Quick Start
### 1. Training the Backbone
```
torchrun --nproc_per_node=8 backbone_train.py
```
### 2. One-Class Detection Pipeline (Backbone Evaluation)
```
python oc_main.py
```
### 3. Binary Classification Model Training
```
torchrun --nproc_per_node=8 bc_train.py
```
### 4. Binary Classification Model Evaluation
```
python bc_eval.py
```

## 📦 Pre-trained Weights
We provide pre-trained model [checkpoints](https://drive.google.com/drive/folders/10cMAWnlo6SFLdp5gR3ykiQHwRRbOJlT-?usp=sharing) for quick reproduction: backbone checkpoint and BC classifier checkpoint.

## ✍️ Citation

If you find our work or code useful for your research, please cite:

```bibtex
@article{zhong2025self,
  title={Self-Supervised AI-Generated Image Detection: A Camera Metadata Perspective},
  author={Zhong, Nan and Zou, Mian and Xu, Yiran and Qian, Zhenxing and Zhang, Xinpeng and Wu, Baoyuan and Ma, Kede},
  journal={arXiv preprint arXiv:2512.05651},
  year={2025}
}
