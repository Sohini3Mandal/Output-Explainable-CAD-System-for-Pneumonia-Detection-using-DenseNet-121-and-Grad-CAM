# Output-Explainable Computer Aided Diagnostic System for Pneumonia Detection using DenseNet-121 and Grad-CAM

An end-to-end, explainable Computer-Aided Diagnostic (CAD) system designed to classify pediatric chest X-rays into **NORMAL** or **PNEUMONIA** categories. The system integrates a fine-tuned **DenseNet-121** deep transfer learning architecture with **Grad-CAM (Gradient-weighted Class Activation Mapping)** to deliver both diagnostic probability scores and visual spatial explanations over affected lung regions.

---

## 📌 Project Overview

Many deep neural networks used in medical imaging function as "black boxes," delivering diagnostic predictions without revealing the underlying visual logic. This project addresses key clinical challenges (such as trust, error identification, and visual transparency) by producing:
1. **Diagnostic Classification:** High-accuracy probabilistic prediction (`NORMAL` vs. `PNEUMONIA`).
2. **Visual Explainability (XAI):** Gradient-based thermal heatmaps targeting the model's final feature extraction layer (`conv5_block16_concat`) to highlight spatial regions driving the diagnosis.
3. **Automated Diagnostic Reporting:** Instant clinical summaries formatted for point-of-care decision support.

---

## 📊 Dataset Specifications

* **Dataset Name:** [Pediatric Chest X-Ray Pneumonia — Balanced Dataset](https://www.kaggle.com/datasets/yusufmurtaza01/chest-xray-pneumonia-balanced-dataset)
* **Original Citation:** Kermany, D., et al., *Cell* 2018 (Guangzhou Women and Children’s Medical Center)
* **Target Task:** Binary Image Classification (`NORMAL` vs. `PNEUMONIA`)
* **Cohort:** Pediatric anterior-posterior (AP) chest X-rays (ages 1–5)
* **Total Images:** 8,530 preprocessed RGB chest radiographs ($224 \times 224$ pixels)

### **Data Distribution & Splits**

| Split | NORMAL | PNEUMONIA | Total Images |
| :--- | :---: | :---: | :---: |
| **Train** | 3,400 | 3,400 | 6,800 |
| **Validation** | 850 | 850 | 1,700 |
| **Test** | 15 | 15 | 30 |
| **Total** | **4,265** | **4,265** | **8,530** |

---

## 🏗️ System Architecture

The proposed CAD pipeline follows a 6-stage modular design:

$$\text{Input} \longrightarrow \text{Preprocessing} \longrightarrow \text{AI Model} \longrightarrow \text{Prediction} \longrightarrow \text{Explainability Module} \longrightarrow \text{Diagnostic Report}$$

1. **Input:** User uploads a raw frontal pediatric chest X-ray image.
2. **Preprocessing:** Resizes images to $224 \times 224$ pixels and normalizes pixel intensities into the range $(-1.0, 1.0)$ using `preprocess_input`.
3. **AI Model:** Base DenseNet-121 backbone connected to a custom classification head (`GlobalAveragePooling2D` $\rightarrow$ `Dense(128)` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Softmax`).
4. **Prediction:** Computes class probabilities and extracts confidence metrics.
5. **Explainability Module:** Calculates Grad-CAM heatmaps from `conv5_block16_concat` and overlays a thermal `JET` colormap onto the original radiograph ($40\%$ heatmap + $60\%$ original).
6. **Diagnostic Report:** Displays side-by-side visual overlays along with structured text summaries and medical disclaimers.

---

## 📈 Model Performance

Evaluated on an unseen test set of 30 chest X-rays (15 NORMAL, 15 PNEUMONIA):

* **Accuracy:** 96.67%
* **Sensitivity (Recall):** 100.00%
* **Specificity:** 93.33%
* **Precision:** 93.75%
* **F1-Score:** 96.77%
* **ROC-AUC:** 1.00

---

## 🛠️ Tech Stack & Tools

* **Language:** Python
* **Deep Learning Frameworks:** TensorFlow, Keras
* **Model Backbone:** DenseNet-121
* **Computer Vision & Math:** OpenCV, NumPy, Matplotlib
* **Dashboard Deployment:** Streamlit

---
