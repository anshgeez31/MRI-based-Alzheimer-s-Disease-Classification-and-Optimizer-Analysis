# 🧠 XceptionNet for MRI-Based Alzheimer's Disease Classification

This project implements a deep learning-based system for classifying stages of Alzheimer's disease using structural MRI scans. Built on the XceptionNet architecture with transfer learning, the model accurately identifies four cognitive states:

- 🟢 Non-Demented  
- 🟡 Very Mild Demented  
- 🟠 Mild Demented  
- 🔴 Moderate Demented  

## 📌 Features

- ✅ Transfer learning with pretrained **XceptionNet**
- ✅ Implemented in **PyTorch**
- ✅ Advanced **data preprocessing**: resizing, normalization, grayscale-to-RGB conversion
- ✅ Data augmentation: flipping, rotation, brightness/contrast jittering
- ✅ Custom classification head with **dropout regularization**
- ✅ Optimizer benchmarking: **Adam**, **SGD + Momentum**, **RMSprop**

## 📊 Results

| Metric                   | Value                           |
|--------------------------|---------------------------------|
| **Training Accuracy**    | 97.92%                          |
| **Validation Accuracy**  | ~95%                            |
| **Precision (Validation)**| 0.97 – 0.99 (per class)         |
| **F1 Score (Validation)**| Up to **0.989**                |
| **High-SNR Slope (Theory)** | Downlink SR: `NM / L`        |
| **Convergence**          | Validation accuracy >95% by 20 epochs |

## 🧠 Model Explainability

- 🎯 **Grad-CAM** heatmaps show high relevance in brain areas like the hippocampus.
- 🧠 **Saliency Maps** confirm model focus on medically relevant regions.
- 🔍 **Misclassification Analysis**: Mostly occurs between adjacent stages (e.g., VMD vs MD).

## 📂 Dataset

- 🧬 Source: Hugging Face (originally from Kaggle)
- 🖼️ Modalities: Structural T1-weighted MRI scans
- 🔄 Label Distribution: Managed via class-weighted loss and stratified sampling

## ⚙️ Architecture

```text
Pretrained XceptionNet (ImageNet)
       ↓
Adaptive Avg Pool → Flatten
       ↓
FC(2048→512) + ReLU + Dropout(0.5)
       ↓
FC(512→128) + ReLU + Dropout(0.3)
       ↓
FC(128→4) → Softmax
