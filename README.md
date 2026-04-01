# 🍅 fyp-tomato_leaf_disease_localization_detection

A hybrid deep learning model for tomato leaf disease detection, combining Faster R-CNN (ResNet50 + FPN) for region localization and EfficientNet-B4 as a classification head.

---

## 📊 Dataset

This project uses the **Tomato Leaf Dataset** provided by Imtiaz et al. (2025).

🔗 Dataset Link: https://data.mendeley.com/datasets/bpfd9cns5g/2  

**Citation:**

Imtiaz, Ahmed; Islam Swapnil, Fahad Bin; Masud, Syed Rayhan; Karmaker, Debajoyti (2025),  
“Tomato Leaf Dataset: A dataset for multiclass disease detection and classification”,  
*Mendeley Data, V2*  
DOI: https://doi.org/10.17632/bpfd9cns5g.2  

**Usage Note:**  
This dataset is used strictly for academic and research purposes. All credits belong to the original authors.
---

## ⚙️ Methodology

This project implements a **two-stage hybrid deep learning pipeline** for disease localization and classification:

### 🔹 1. Localization (Object Detection)
- Model: **Faster R-CNN with ResNet50 backbone + Feature Pyramid Network (FPN)**
- Purpose: Detect and localize disease regions in tomato leaves
- Output: Bounding boxes around infected areas

### 🔹 2. Classification (Disease Recognition)
- Model: **EfficientNet-B4**
- Purpose: Classify the detected regions into specific disease categories
- Input: Cropped regions from Faster R-CNN detections

---

## 🔄 Data Preparation

- The dataset authors originally provided annotations in **YOLO format**
- These annotations were **converted to Faster R-CNN compatible format** (Pascal VOC-style bounding boxes)
- The **same train/validation/test split** provided by the dataset authors was strictly followed to ensure consistency and fair evaluation

---

## 🚀 Pipeline Overview

1. Input leaf image  
2. Faster R-CNN detects disease regions  
3. Detected regions are cropped  
4. Cropped regions are passed to EfficientNet-B4  
5. Final disease classification is generated  

---
