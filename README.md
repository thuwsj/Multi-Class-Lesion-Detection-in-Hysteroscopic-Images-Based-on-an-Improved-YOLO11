# Multi-Class Lesion Detection in Hysteroscopic Images Based on an Improved YOLO11

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-AGPL--3.0-green.svg)](LICENSE)

## Overview

This project implements an improved YOLO11-based model for multi-class lesion detection in hysteroscopic images. The system is designed to accurately detect and classify 8 different types of intrauterine lesions, providing valuable assistance for medical diagnosis.

### Key Features

- **Multi-class Detection**: Detects 8 types of uterine lesions with high accuracy
- **Improved YOLO11**: Enhanced with RegNetY backbone, MSAM attention mechanism, and WIoU loss
- **User-friendly GUI**: PyQt5-based graphical interface for easy image/video/camera detection
- **Real-time Processing**: Supports real-time detection with webcam input
- **Visualization**: Comprehensive detection results with bounding boxes and confidence scores
- **Comparison Models**: Includes SSD, RetinaNet, RT-DETR, YOLOv8, and YOLOv10 for benchmarking

### Detected Lesion Types

| Class | Full Name | Abbreviation |
|-------|-----------|--------------|
| 0 | Submucous Myoma | SM |
| 1 | Endometrial Carcinoma | EC |
| 2 | Endometrial Polyp | EP |
| 3 | Polypoid Hyperplasia of Endometrium | PHE |
| 4 | Endometrial Hyperplasia without Atypia | EHWA |
| 5 | Intrauterine Foreign Body | IFB |
| 6 | Cervical Polyp | CP |
| 7 | Atypical Endometrial Hyperplasia | AEH |

## Architecture

The improved YOLO11 model incorporates three key enhancements:

### 1. RegNetY Backbone

**Core Logic:**
- RegNetY is a variant in the RegNet family that introduces the Squeeze-and-Excitation (SE) module
- The SE module enhances representational capacity by adaptively recalibrating channel-wise feature responses
- RegNetY uses quantized linear parameterization to design the network
- Network width and depth are controlled via parameters such as `w_a`, `w_0`, and `w_m`
- Group convolutions and bottleneck structures improve computational efficiency
- It serves as an efficient backbone for object detectors such as YOLO

### 2. MSAM (Multi-Scale Attention Module)

**Core Logic:**
- Built as an improvement over CBAM: replace the original channel attention with multi-scale convolutions, giving the channel attention multi-scale capability
- The first half is replaced with an **MSCAAttention** module, which extracts features using multiple convolutions at different scales
- The latter half still uses CBAM's **spatialAttention** module
- Element-wise multiply the outputs of both parts to obtain the final output

### 3. WIoU Loss

**Core Logic:**
- Initialize the `WIoU_Scale` class with IoU values
- Automatically call the `_update` method to maintain a running mean (`iou_mean`)
- Choose different loss formulations based on the monotonous flag
- Apply composite scaling based on the gamma and delta parameters
- Return the scaled loss value

**Architecture Flow:**
```
Input Image → RegNetY Backbone → SPPF → C2PSA → MSAM → Detection Head → Output
```

## Project Structure

```
Multi-Class-Lesion-Detection-in-Hysteroscopic-Images-Based-on-an-Improved-YOLO11-main/
├── 1_yolo_qt.py                    # PyQt5 GUI for detection
├── 2_train.py                      # Training script
├── 3_get_size.py                   # Dataset analysis utility
├── data.yaml                       # Dataset configuration
├── load_yaml.py                    # YAML loader utility
├── requirements.txt                # Python dependencies
├── yaml/                           # Model configuration files
│   ├── yolo11-msam.yaml           # YOLO11 + MSAM
│   ├── yolov11_regnety.yaml       # YOLO11 + RegNetY
│   └── yolov11_regnety_msam.yaml  # YOLO11 + RegNetY + MSAM
├── weights/                        # Trained model weights
│   ├── 1_base/                    # Baseline model
│   ├── 2_regnety/                 # With RegNetY
│   ├── 3_msam/                    # With MSAM
│   ├── 4_wiou/                    # With WIoU
│   └── 5_regnety_msam_wiou/       # Full improved model
├── models/                         # Pre-trained model files
├── a_ssd-pytorch-master/          # SSD implementation
├── b_retinanet-pytorch-master/    # RetinaNet implementation
├── c_rtdetr/                      # RT-DETR implementation
├── d_yolov8/                      # YOLOv8 implementation
└── e_yolov10/                     # YOLOv10 implementation
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Installation

1. Clone the repository:
```bash
git clone https://github.com/thuwsj/Multi-Class-Lesion-Detection-in-Hysteroscopic-Images-Based-on-an-Improved-YOLO11.git
cd Multi-Class-Lesion-Detection-in-Hysteroscopic-Images-Based-on-an-Improved-YOLO11
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install additional dependencies for specific models:
```bash
cd c_rtdetr
pip install -r requirements.txt
cd ..
```

### Dataset Preparation

1. Organize your dataset in YOLO format:
```
dataset/
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

2. Update `data.yaml` with your dataset paths:
```yaml
train: /path/to/your/dataset/train/images
val: /path/to/your/dataset/val/images
nc: 8
names: ['SM', 'EC', 'EP', 'PHE', 'EHWA', 'IFB', 'CP', 'AEH']
```

## Training

### Train the Improved YOLO11 Model

```bash
python 2_train.py \
    --mode train \
    --weights yaml/yolov11_regnety_msam.yaml \
    --data data.yaml \
    --epoch 100 \
    --batch 64 \
    --device 0 \
    --wiou True \
    --name regnety_msam_wiou
```

### Training Parameters

- `--mode`: Choose between `train` or `val` (validation)
- `--weights`: Path to model configuration YAML file
- `--data`: Path to dataset configuration file
- `--epoch`: Number of training epochs (default: 100)
- `--batch`: Batch size (default: 64)
- `--workers`: Number of data loading workers (default: 8)
- `--device`: GPU device ID (default: 0)
- `--wiou`: Enable WIoU loss (default: True)
- `--optimizer`: Optimizer choice (default: SGD)
- `--name`: Experiment name for saving results

### Ablation Studies

Train different model variants for comparison:

```bash
# Baseline YOLO11
python 2_train.py --weights yaml/yolo11-base.yaml --name base --wiou False

# YOLO11 + RegNetY
python 2_train.py --weights yaml/yolov11_regnety.yaml --name regnety --wiou False

# YOLO11 + MSAM
python 2_train.py --weights yaml/yolo11-msam.yaml --name msam --wiou False

# YOLO11 + WIoU
python 2_train.py --weights yaml/yolo11-base.yaml --name wiou --wiou True

# Full improved model (RegNetY + MSAM + WIoU)
python 2_train.py --weights yaml/yolov11_regnety_msam.yaml --name regnety_msam_wiou --wiou True
```

## Inference

### GUI Application

Launch the PyQt5 graphical interface:

```bash
python 1_yolo_qt.py --weights runs/weights/5_regnety_msam_wiou/weights/best.pt
```

Features of the GUI:
- **Image Detection**: Load and detect lesions in single images
- **Video Detection**: Process video files frame by frame
- **Camera Detection**: Real-time detection with webcam
- **Statistics**: View detection statistics and class distribution
- **History**: Track all detection records
- **Export**: Save detection records to CSV

### Command-line Inference

```bash
from ultralytics import YOLO

# Load model
model = YOLO('runs/weights/5_regnety_msam_wiou/weights/best.pt')

# Inference on image
results = model('path/to/image.jpg', conf=0.3, iou=0.5)

# Inference on video
results = model('path/to/video.mp4', conf=0.3, iou=0.5)

# Inference on webcam
results = model(0, conf=0.3, iou=0.5)
```

### Inference Parameters

- `--weights`: Path to trained model weights
- `--conf_thre`: Confidence threshold (default: 0.3)
- `--iou_thre`: IoU threshold for NMS (default: 0.5)

## Evaluation

Evaluate model performance:

```bash
python 2_train.py \
    --mode val \
    --weights runs/weights/5_regnety_msam_wiou/weights/best.pt \
    --data data.yaml \
    --batch 64 \
    --device 0
```

The evaluation will output:
- Precision, Recall, F1-score per class
- mAP@0.5 and mAP@0.5:0.95
- Confusion matrix
- PR curves

## Comparison Models

This repository includes implementations of several state-of-the-art object detection models for comparison:

### SSD (Single Shot Detector)
```bash
cd a_ssd-pytorch-master
python train.py
```

### RetinaNet
```bash
cd b_retinanet-pytorch-master
python train.py
```

### RT-DETR (Real-time Detection Transformer)
```bash
cd c_rtdetr
python 1_train_detr.py
```

### YOLOv8
```bash
python 2_train.py --weights d_yolov8/yolov8n.yaml
```

### YOLOv10
```bash
python 2_train.py --weights e_yolov10/yolov10n.yaml
```

## Experimental Results

### Parameters
- **P**: Precision
- **R**: Recall
- **Wsz**: Weight size (MB)

### Ablation Study

| Model | RegNetY | MSAM | WIoU | Map.5 | Map.95 | P | R | Wsz |
|-------|---------|------|------|-------|--------|------|------|-----|
| Baseline | | | | 89.92 | 62.81 | 89.01 | 85.37 | 5.22 |
| RegNetY | ✓ | | | 90.43 | 62.12 | 86.69 | 87.76 | 9.01 |
| MSAM | | ✓ | | 91.47 | 61.38 | 92.52 | 85.33 | 5.4 |
| WIoU | | | ✓ | 91.12 | 60.91 | 89.67 | 86.1 | 5.22 |
| **RegNetY+MSAM+WIoU** | ✓ | ✓ | ✓ | **93.4** | **66.66** | **91.66** | **88.75** | **9.07** |

### Comparative Experiments

| Model | Map.5 | Map.95 | P | R | Wsz |
|-------|-------|--------|------|------|-----|
| yolov11n | 89.92 | 62.81 | 89.01 | 85.37 | 5.22 |
| yolov10n | 81.53 | 52.99 | 76.25 | 76.96 | 5.48 |
| yolov8n | 83.69 | 53.74 | 83.03 | 76.16 | 5.36 |
| ssd | 59.83 | 27.1 | 31.11 | 86.62 | 17.86 |
| retinanet | 51.55 | 21.4 | 57.96 | 25.03 | 76.07 |
| rtdetr | 81.79 | 53 | 81.75 | 72.63 | 38.53 |
| **ours** | **93.4** | **66.66** | **91.66** | **88.75** | **9.07** |

### Key Findings

The experimental results demonstrate that:
- The complete improved model (RegNetY + MSAM + WIoU) achieves **93.4% mAP@0.5** and **66.66% mAP@0.5:0.95**
- Compared to baseline YOLO11, our model improves mAP@0.5 by **3.48%** and mAP@0.5:0.95 by **3.85%**
- The model outperforms all comparison models including YOLOv10, YOLOv8, SSD, RetinaNet, and RT-DETR
- Model size remains compact at **9.07 MB**, suitable for deployment

## Implementation Details

### Modification Locations

To implement the improvements in the Ultralytics framework, the following files need to be modified:

1. **ultralytics/nn/modules/block.py**: Import the various improved modules
2. **ultralytics/nn/modules/__init__.py**: Import the improved module classes and functions and add them to the package namespace
3. **ultralytics/nn/tasks.py**: Add functions in the parsing module to register the improved modules
4. **YAML files**: Add the corresponding modules and insert attention before the detect head
5. **ultralytics/utils/loss.py**: Add the loss function definition
6. **ultralytics/utils/loss.py**: Apply the loss function

### Utilities

#### Get Dataset Statistics
```bash
python 3_get_size.py
```

This utility analyzes your dataset and provides:
- Image size distribution
- Class distribution
- Bounding box statistics
- Dataset balance metrics

## Data Access

We make the data of this research available for sharing.

### Data Acquisition

**Data acquisition is extremely simple.**

We provide de-identified data; due to hospital privacy requirements, we use a request-by-email distribution.

**Email**: bsuw@foxmail.com

**Request format**:
- Subject: Data Request for [Your Institution/Purpose]
- Body: Name + Purpose (reproduce this paper) + pledge to delete after reproduction and not redistribute

**Response time**: Within 48 hours, we'll send the data via large attachment/temporary download link.

**Usage restrictions**:
- Academic reproduction only; no commercial use or redistribution
- If you are a journal reviewer, you do not need to provide your name—just mention the journal name

### Documentation

For detailed documentation, please refer to:
- `4_Important! Must-read!.pdf` - Important guidelines and best practices
- `Data acquisition methods.pdf` - Dataset collection and annotation procedures

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request to the [GitHub repository](https://github.com/thuwsj/Multi-Class-Lesion-Detection-in-Hysteroscopic-Images-Based-on-an-Improved-YOLO11).

## Citation

If you use this code or dataset in your research, please cite our work accordingly.

## Issues and Questions

If you encounter any problems or have questions, please open an issue on the [GitHub repository](https://github.com/thuwsj/Multi-Class-Lesion-Detection-in-Hysteroscopic-Images-Based-on-an-Improved-YOLO11/issues).

## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the YOLO framework
- [PyTorch](https://pytorch.org/) for the deep learning framework
- All contributors and researchers in the medical imaging community

## References

1. Ultralytics YOLO: https://github.com/ultralytics/ultralytics
2. RegNet: Designing Network Design Spaces
3. Attention mechanisms in computer vision
4. Medical image analysis and object detection

---

**Repository**: [https://github.com/thuwsj/Multi-Class-Lesion-Detection-in-Hysteroscopic-Images-Based-on-an-Improved-YOLO11](https://github.com/thuwsj/Multi-Class-Lesion-Detection-in-Hysteroscopic-Images-Based-on-an-Improved-YOLO11)

If you find this project helpful, please consider giving it a star on [GitHub](https://github.com/thuwsj/Multi-Class-Lesion-Detection-in-Hysteroscopic-Images-Based-on-an-Improved-YOLO11)!

