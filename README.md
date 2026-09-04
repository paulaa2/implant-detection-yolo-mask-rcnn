# Dental Implant Detection: YOLO11 vs. Mask R-CNN

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2024-orange.svg)](https://www.mathworks.com/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)](https://pytorch.org/)
[![YOLO](https://img.shields.io/badge/Model-YOLO11-yellow.svg)](https://github.com/ultralytics/ultralytics)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()
[![Topic](https://img.shields.io/badge/Topic-Computer_Vision_%26_Medical_Imaging-purple.svg)]()

A comparative study of two state-of-the-art object detection and instance segmentation architectures — **YOLO11** and **Mask R-CNN** — applied to the automatic detection and localization of **dental implants in X-ray images**. This project constitutes the final assignment (PF) of an advanced computer vision course.

---

## 📌 Project Overview

The core goal is to train, fine-tune, and benchmark two fundamentally different detection paradigms on a custom medical imaging dataset of **200 dental X-ray images** labeled with bounding boxes for the implant region.

### Models Compared

| Model | Paradigm | Task |
| :--- | :--- | :--- |
| **YOLO11** | Single-stage, anchor-free detector | Object detection (bounding box) |
| **Mask R-CNN** | Two-stage, region proposal-based | Instance segmentation (bounding box + pixel mask) |

> **Hardware used during training**: NVIDIA GeForce RTX 4060 Laptop GPU (8.59 GB VRAM)

---

## 📂 Repository Structure

```directory
implant-detection-yolo-mask-rcnn/
│
├── data/
│   ├── images/               # 200 raw dental X-ray images (input dataset)
│   └── labels/
│       └── bounding_boxes_yolo.txt   # YOLO-format annotations (class, cx, cy, w, h — normalized)
│
├── notebooks/
│   ├── PF_C2_ClaudiaPaula.ipynb   # Main Python notebook: YOLO11 & Mask R-CNN training & evaluation
│   └── PF_C1_ClaudiaPaula.mlx    # MATLAB Live Script: preliminary analysis / part 1
│
├── results/
│   ├── bounding_boxes/       # YOLO11 detection outputs: images with predicted bounding boxes
│   └── masks/                # Mask R-CNN outputs: images with predicted instance segmentation masks
│
├── .gitignore
└── README.md
```

---

## 🧪 Methodology

### Dataset
- **200 dental periapical X-ray images** containing one or more visible implants.
- **Single class**: `implant` (class index `0`).
- **Annotation format**: YOLO normalized format — `<image_id> <class> <cx> <cy> <width> <height>`.

### Pipeline (Notebook `PF_C2_ClaudiaPaula.ipynb`)

1. **Environment Setup**: Import libraries (`ultralytics`, `torchvision`, `albumentations`, `pycocotools`, `torchmetrics`), configure device.
2. **Data Preparation**: Load images, parse YOLO annotations, apply data augmentation via `albumentations`.
3. **YOLO11 Training**: Fine-tune a pre-trained YOLO11 model on the implant dataset.
4. **Mask R-CNN Training**: Fine-tune a pre-trained `torchvision` Mask R-CNN (ResNet-50 FPN backbone) for instance segmentation.
5. **Evaluation**: Compute **mAP** (Mean Average Precision), IoU, precision, recall — using `pycocotools` and `torchmetrics`.
6. **Comparison**: Side-by-side analysis of both models on held-out test images.

---

## 🔬 Key Libraries & Dependencies

```text
torch / torchvision
ultralytics          # YOLO11
albumentations       # Data augmentation
pycocotools          # COCO evaluation metrics
torchmetrics         # MeanAveragePrecision
opencv-python        # Image I/O and visualization
Pillow
scikit-learn
pandas / numpy / matplotlib
```

Install all dependencies:
```bash
pip install torch torchvision ultralytics albumentations pycocotools torchmetrics opencv-python scikit-learn pandas matplotlib Pillow
```

---

## 🚀 How to Run

Open the main notebook and execute all cells sequentially:

```bash
jupyter notebook notebooks/PF_C2_ClaudiaPaula.ipynb
```

> ⚠️ A GPU with ≥6 GB VRAM is recommended for training both models.  
> The notebook auto-detects `cuda`, `mps`, or `cpu` and configures accordingly.

---

## 📊 Results Summary

| | YOLO11 | Mask R-CNN |
| :--- | :--- | :--- |
| **Task** | Detection (bbox) | Segmentation (bbox + mask) |
| **Speed** | Fast (single-stage) | Slower (two-stage) |
| **Outputs** | `results/bounding_boxes/` | `results/masks/` |
| **Metric** | mAP@0.5 | mAP@0.5:0.95 |

> Detailed quantitative metrics and visualizations are available inside the notebook.

---

## 👥 Authors & License

Developed by **Claudia Gallego & Paula Esteve** — January 2025.  
Final project for an Advanced Computer Vision course.  
Distributed under the **MIT License**.
