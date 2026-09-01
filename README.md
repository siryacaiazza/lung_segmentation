<h1 align="center">Lung CT Segmentation with Deep Learning</h1>

<p align="center">
  <b>Automated lung segmentation from thoracic CT scans using 2D and 3D convolutional neural networks</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/MONAI-1.3+-green?logo=data:image/svg+xml;base64,..." alt="MONAI"/>
  <img src="https://img.shields.io/badge/SMP-0.3+-blueviolet" alt="SMP"/>
  <img src="https://img.shields.io/badge/Best_Dice-0.972-success" alt="Best Dice"/>
</p>

---

## Overview

This project tackles the problem of **automated lung segmentation** from thoracic CT scans — a critical task in radiation therapy planning, where accurate organ delineation directly impacts treatment quality and patient safety.

Using the [Lung CT Segmentation Challenge (LCTSC)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=24284539) dataset, this work implements a complete end-to-end pipeline: from raw DICOM data ingestion and preprocessing, through training and benchmarking multiple segmentation architectures (both 2D and 3D), to inference on unseen external data.

The best model — **MAnet 3D with a ResNet34 encoder** — achieves a mean Dice score of **0.972** and a Hausdorff Distance (95th percentile) of **1.15 px** on the official test set.
This project was created in collaboration with [AndreaBaraldi99]https://github.com/AndreaBaraldi99
---

## Table of Contents

- [Dataset](#dataset)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Model Architectures](#model-architectures)
- [Training Strategy](#training-strategy)
- [Results](#results)
- [Generalization on External Data](#generalization-on-external-data)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [References](#references)

---

## Dataset

The **LCTSC dataset** contains thoracic CT scans from **60 patients** collected across 3 clinical sites, each with expert-drawn contours for 5 organs at risk:

| Structure | Description |
|-----------|-------------|
| Lung (Right) | Right lung parenchyma |
| Lung (Left) | Left lung parenchyma |
| Heart | Cardiac structure |
| Esophagus | Esophageal tube |
| Spinal Cord | Spinal canal |

**Data split (official):**

| Split | Patients | Source |
|-------|----------|--------|
| Training | 36 | 3 sites × 12 patients |
| Test | 24 | 3 sites × 8 patients |
| Hidden external | 10 | LUNG1 cohort (no ground truth) |

This project focuses on **binary lung segmentation** (combined left + right lung masks).

---

## Preprocessing Pipeline

Raw DICOM data undergoes a multi-step preprocessing pipeline before being fed to the models:

```
DICOM CT Series + RTSTRUCT → HU Conversion → Lung-Based Cropping → HU Windowing → Resize → Normalize → .npz
```

1. **DICOM Loading** — CT series read via SimpleITK; voxel values converted to Hounsfield Units using `RescaleSlope` and `RescaleIntercept`
2. **ROI Extraction** — Segmentation masks parsed from RTSTRUCT files using [rt-utils](https://github.com/qurit/rt-utils)
3. **Lung-Based Cropping** — Combined lung mask used to compute a bounding box, expanded by **20% padding** on each spatial axis to retain relevant anatomy while removing background
4. **HU Windowing** — Values clipped to **[-1000, 400] HU**, preserving the air-to-bone range relevant for thoracic imaging
5. **Spatial Resizing** — Each axial slice resized to **128 × 128** via `scipy.ndimage.zoom`; the depth axis (Z) is preserved
6. **Normalization** — Linear mapping to [0, 1] range
7. **Storage** — Compressed `.npz` files containing the image volume, voxel spacing, and individual ROI masks

---

## Model Architectures

Six models were trained and compared — three architectures, each in a 2D and 3D variant — all built with [Segmentation Models PyTorch (SMP)](https://github.com/qubvel-org/segmentation_models.pytorch):

| Architecture | Encoder | Depth | Decoder Channels | Key Feature |
|-------------|---------|-------|-------------------|-------------|
| **U-Net** | ResNet34 | 4 | 512 → 256 → 128 → 64 | Standard encoder-decoder with skip connections |
| **UNet++** | ResNet34 | 4 | 512 → 256 → 128 → 64 | Nested dense skip connections for multi-scale features |
| **MAnet** | ResNet34 | 4 | 512 → 256 → 128 → 64 | Multi-scale Attention mechanism in the decoder |

All models: 1 input channel (grayscale CT), 1 output class (binary lung mask), no pretrained encoder weights (trained from scratch on medical data).

<p align="center">
  <img src="images/params_comparison_3d.png" alt="Model Parameters Comparison" width="600"/>
</p>

---

## Training Strategy

### 2D Models
- **Input:** Individual axial slices extracted from 3D volumes
- **Batch size:** 64
- **Data augmentation:** HorizontalFlip, Rotation (±15°), Gaussian Noise, Brightness/Contrast adjustments, Gaussian Blur (via [Albumentations](https://albumentations.ai/))

### 3D Models
- **Input:** Random **64 × 64 × 32** patches extracted from full volumes (8 patches per volume per epoch)
- **Batch size:** 2
- **Data augmentation:** Random flips, 90° rotations, Gaussian noise, contrast adjustments, Gaussian smoothing (via [MONAI](https://monai.io/))
- **Inference:** MONAI sliding window with `roi_size=(64, 64, 32)`, `overlap=0.5`, and Gaussian importance weighting

### Common Settings

| Parameter | Value |
|-----------|-------|
| Loss function | Dice Loss (from logits) |
| Optimizer | Adam (lr = 1×10⁻⁴) |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Early stopping | Patience = 10, min delta = 0.001 |
| Max epochs | 50 |
| Gradient clipping | max_norm = 1.0 |
| Precision | Mixed precision (FP16) |

<p align="center">
  <img src="images/training_comparison_3d.png" alt="Training Curves" width="600"/>
</p>

---

## Results

### 3D Model Performance on Test Set (24 cases)

| Model | Dice | IoU | Sensitivity | Precision | HD95 (px) |
|-------|------|-----|-------------|-----------|-----------|
| **MAnet** | **0.9716 ± 0.012** | **0.9450 ± 0.022** | **0.9696 ± 0.024** | **0.9740 ± 0.011** | **1.15 ± 0.31** |
| UNet++ | 0.9674 ± 0.016 | 0.9373 ± 0.029 | 0.9598 ± 0.028 | 0.9756 ± 0.014 | 2.10 ± 4.59 |
| U-Net | 0.9637 ± 0.014 | 0.9302 ± 0.025 | 0.9615 ± 0.028 | 0.9664 ± 0.012 | 1.32 ± 0.47 |

> **MAnet 3D** achieves the highest Dice score and the most consistent boundary accuracy (lowest HD95 with minimal variance).

<p align="center">
  <img src="images/model_comparison_3d.png" alt="3D Model Comparison" width="600"/>
</p>

### 2D Model Performance (Best Validation Dice Loss)

| Model | Best Val Loss |
|-------|--------------|
| **MAnet** | **0.0364** |
| U-Net | 0.0428 |
| UNet++ | 0.0454 |

<p align="center">
  <img src="images/model_comparison_2d.png" alt="2D Model Comparison" width="600"/>
</p>

### 2D vs 3D Comparison

<p align="center">
  <img src="images/comparison.png" alt="2D vs 3D Comparison" width="600"/>
</p>

3D models benefit from native volumetric context, producing smoother, more anatomically consistent segmentations by jointly reasoning over neighboring slices — eliminating the inter-slice inconsistencies that 2D models can exhibit.

### Validation Loss Curves

<p align="center">
  <img src="images/val_loss_comparison_3d.png" alt="Validation Loss Comparison" width="600"/>
</p>

### Inference Time Benchmark

<p align="center">
  <img src="images/inference_time_2d_vs_3d.png" alt="Inference Time Comparison" width="600"/>
</p>

---

## Generalization on External Data

To assess robustness beyond the LCTSC distribution, the best model (MAnet 3D) was applied to **10 unseen cases from the LUNG1 cohort** — a separate dataset with no ground truth annotations.

Preprocessing for these external volumes follows a simplified pipeline (no lung-based cropping, since ground truth masks are unavailable). A **body mask post-processing** step removes false positive predictions outside the patient body contour.

<p align="center">
  <img src="images/comparison_bodymask.png" alt="Body Mask Post-Processing" width="600"/>
</p>

Predictions are stored in `test_cases_inference/` as `.npy` arrays and are qualitatively evaluated through 3D surface rendering (marching cubes via Plotly).

---

## Project Structure

```
HealthcareImaging/
├── README.md                          # This file
├── Lctsc_dataset.ipynb                # 2D: preprocessing, training & evaluation
├── Lctsc_dataset_3d.ipynb             # 3D: preprocessing, training & evaluation
├── LSCT_unetplusplus3d.ipynb          # UNet++ 3D standalone training
│
├── LCTSC/                             # Raw DICOM data (60 patients)
├── data_processed/                    # Preprocessed .npz volumes
│   ├── Train/                         #   36 training volumes
│   └── Test/                          #   24 test volumes
│
├── hidden_test_set/                   # External LUNG1 data (10 cases)
├── test_cases_inference/              # Inference outputs on external data
│
├── manet_3d__best.pth                 # Best model weights (MAnet 3D)
├── unet_3d__best.pth                  # U-Net 3D weights
├── unetplusplus_3d__best.pth         # UNet++ 3D weights
├── manet_best.pth/                    # MAnet 2D (SMP Hub format)
├── unet_best.pth/                     # U-Net 2D (SMP Hub format)
├── unetplusplus_best.pth/            # UNet++ 2D (SMP Hub format)
│
├── *_detailed_metrics.csv             # Per-case evaluation metrics
├── images/                            # Plots and figures
│
├── LCTSCProject2024/                  # Preprocessing library
│   ├── lctsc_preprocessor/            #   DICOM/RTSTRUCT utilities
│   │   ├── dicom_utils.py             #   CT loading + HU conversion
│   │   ├── rtstruct_utils.py          #   ROI mask extraction
│   │   ├── utils.py                   #   Core preprocessing function
│   │   └── config.py                  #   Output paths configuration
│   ├── lctsc_cnn/                     #   CNN data loading utilities
│   └── scripts/                       #   CLI preprocessing scripts
│
└── lctsc_metadata.csv                 # Patient-to-DICOM path mapping
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended: ≥ 8 GB VRAM)

### Installation

```bash
git clone https://github.com/AndreaBaraldi99/HealthcareImaging.git
cd HealthcareImaging

python -m venv .venv
source .venv/bin/activate

pip install -r LCTSCProject2024/requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install segmentation-models-pytorch monai albumentations
```

### Run Preprocessing

```bash
python LCTSCProject2024/scripts/preprocess_lctsc.py lctsc_metadata.csv ./LCTSC --plot
```

### Train & Evaluate

Open the Jupyter notebooks for the complete training and evaluation workflows:

- **2D models:** `Lctsc_dataset.ipynb`
- **3D models:** `Lctsc_dataset_3d.ipynb`

### Inference with Pretrained Weights

```python
import torch
import segmentation_models_pytorch_3d as smp

model = smp.MAnet(
    encoder_name="resnet34",
    encoder_depth=4,
    encoder_weights=None,
    decoder_channels=(512, 256, 128, 64),
    in_channels=1,
    classes=1,
)
model.load_state_dict(torch.load("manet_3d__best.pth", map_location="cpu"))
model.eval()
```

---

## References

- **LCTSC Dataset:** Yang, J., et al. "Autosegmentation for thoracic radiation treatment planning: A grand challenge at AAPM 2017." *Medical Physics*, 2018. [[Link]](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=24284539)
- **U-Net:** Ronneberger, O., et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation." *MICCAI*, 2015.
- **UNet++:** Zhou, Z., et al. "UNet++: A Nested U-Net Architecture for Medical Image Segmentation." *DLMIA*, 2018.
- **MAnet:** Fan, T., et al. "MA-Net: A Multi-Scale Attention Net for Medical Image Segmentation." *ISBI*, 2020.
- **Segmentation Models PyTorch:** [github.com/qubvel-org/segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)
- **MONAI:** [monai.io](https://monai.io/)

---

<p align="center">
  <i>Built with PyTorch, MONAI, and Segmentation Models PyTorch</i>
</p>
