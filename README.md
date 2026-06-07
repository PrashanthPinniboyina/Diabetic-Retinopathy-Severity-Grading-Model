# 🩺 Diabetic Retinopathy Severity Grading Model
### CNN-Based Medical Image Classification | Machine Learning Project

---

## 📋 Project Overview

Diabetic Retinopathy (DR) is a diabetes complication that affects the retina and is the **leading cause of preventable blindness** among working-age adults worldwide. This project implements a deep learning pipeline to automatically grade DR severity from fundus (retinal) images using Convolutional Neural Networks and Transfer Learning.

| Detail | Info |
|--------|------|
| **Subject** | Machine Learning |
| **Phase** | 2 — Proposal, Code Development & Technical Implementation |
| **Framework** | TensorFlow / Keras |
| **Platform** | Kaggle Notebook |

---

## 🎯 DR Severity Grades

| Grade | Severity | Description |
|-------|----------|-------------|
| 0 | No DR | Healthy retina — no abnormalities |
| 1 | Mild | Microaneurysms only |
| 2 | Moderate | More than microaneurysms but less than severe |
| 3 | Severe | Extensive hemorrhages, venous beading, IRMA |
| 4 | Proliferative DR | Neovascularization — high risk of vision loss |

---

## 📂 Repository Structure

```
diabetic-retinopathy-grading/
│
├── diabetic_retinopathy_grading.ipynb   ← Main Kaggle notebook (full pipeline)
├── README.md                            ← This file
│
├── figures/                             ← Output figures (auto-generated)
│   ├── class_distribution.png
│   ├── sample_images.png
│   ├── preprocessing_comparison.png
│   ├── data_augmentation.png
│   ├── cnn_curves.png
│   ├── effnet_curves.png
│   ├── resnet_curves.png
│   ├── custom_cnn_confusion_matrix.png
│   ├── efficientnetb3_confusion_matrix.png
│   ├── resnet50_confusion_matrix.png
│   ├── cnn_roc.png
│   ├── effnet_roc.png
│   ├── resnet_roc.png
│   ├── model_comparison_chart.png
│   ├── model_comparison.csv
│   ├── gradcam_custom_cnn.png
│   ├── error_analysis.png
│   ├── misclassified_samples.png
│   └── single_prediction.png
│
└── models/                              ← Saved models (auto-generated)
    ├── custom_cnn_final.h5
    ├── efficientnetb3_final.h5
    └── resnet50_final.h5
```

---

## 🗂️ Dataset

**Dataset:** [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data)

| Property | Value |
|----------|-------|
| Total Images | ~3,662 (training set) |
| Number of Classes | 5 (Grades 0–4) |
| Image Type | Retinal fundus photographs (JPEG/PNG) |
| Image Resolution | Variable → resized to 224 × 224 |
| Class Balance | Imbalanced (Grade 0 dominates) |

---

## 🔬 Research Questions

1. Can a CNN accurately classify diabetic retinopathy severity from fundus images?
2. Does transfer learning (EfficientNetB3 / ResNet50) outperform a custom CNN?
3. Which visual regions does the model focus on? (Grad-CAM analysis)
4. How does the model perform across different severity grades?

---

## 🏗️ Methodology

### Preprocessing
- Resize all images to 224 × 224 pixels
- Normalize pixel values to [0, 1]
- Ben Graham preprocessing (Gaussian blur subtraction for lesion enhancement)

### Data Splits
| Split | Percentage | Count (approx.) |
|-------|-----------|-----------------|
| Train | 70% | ~2,563 |
| Validation | 15% | ~549 |
| Test | 15% | ~549 |

All splits are **stratified** to preserve class distribution.

### Data Augmentation (Training Only)
- Random rotations ± 30°
- Horizontal & vertical flipping
- Width/height shifts ± 10%
- Zoom range ± 20%
- Brightness variation [0.8, 1.2]

### Models Implemented

#### 1. Custom CNN (Baseline)
- 4 convolutional blocks: 32 → 64 → 128 → 256 filters
- Each block: Conv2D → BatchNorm → ReLU → MaxPool → Dropout
- Head: GlobalAveragePooling → Dense(256) → Dropout → Softmax(5)

#### 2. EfficientNetB3 (Primary Transfer Learning)
- Pretrained on ImageNet
- Two-phase training: Feature Extraction → Fine-tuning (top 50 layers)
- Custom classification head added

#### 3. ResNet50 (Secondary Transfer Learning)
- Pretrained on ImageNet
- Used for comparison against EfficientNetB3

### Training Strategy
- Optimizer: Adam
- Loss: Categorical Cross-Entropy
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

---

## 📊 Evaluation Metrics

- Accuracy (train, validation, test)
- Loss curves
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score per class)
- Multi-class ROC Curves (One-vs-Rest) + AUC
- Grad-CAM visualizations
- Error analysis (misclassification pairs)
- Model comparison table

---

## 🚀 How to Run

### On Kaggle (Recommended)

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click **New Notebook** → Upload `diabetic_retinopathy_grading.ipynb`
3. Add the APTOS 2019 dataset:
   - Click **Add Data** → Search *"APTOS 2019 Blindness Detection"* → Add
4. Enable **GPU accelerator** (Settings → Accelerator → GPU T4 x2)
5. Click **Run All**

### Local (Optional)

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/diabetic-retinopathy-grading.git
cd diabetic-retinopathy-grading

# Install dependencies
pip install tensorflow opencv-python scikit-learn matplotlib seaborn pandas numpy pillow

# Download dataset from Kaggle
kaggle competitions download -c aptos2019-blindness-detection
unzip aptos2019-blindness-detection.zip -d data/

# Run the notebook
jupyter notebook diabetic_retinopathy_grading.ipynb
```

---

## 🖼️ Single Image Prediction

After training, classify any retinal image with one line:

```python
# Predict DR grade from a single image
predicted_class, confidence = predict_single_image(
    '/path/to/your/retinal_image.png',
    model=effnet_model,
    model_name='EfficientNetB3'
)
```

The function will display:
- The input image
- A probability bar chart for all 5 grades
- The predicted grade and confidence score

---

## 📦 Dependencies

| Library | Version |
|---------|---------|
| TensorFlow | ≥ 2.10 |
| Keras | (bundled with TF) |
| NumPy | ≥ 1.21 |
| Pandas | ≥ 1.3 |
| OpenCV | ≥ 4.5 |
| Scikit-learn | ≥ 1.0 |
| Matplotlib | ≥ 3.5 |
| Seaborn | ≥ 0.11 |
| Pillow | ≥ 8.0 |

---

## 📈 Expected Results

| Model | Expected Accuracy | Expected AUC |
|-------|------------------|--------------|
| Custom CNN | 75–80% | 0.88–0.92 |
| EfficientNetB3 | 85–90% | 0.93–0.97 |
| ResNet50 | 80–85% | 0.91–0.94 |

---

## 🔮 Future Work

- Handle class imbalance with class weights or oversampling (SMOTE)
- Ensemble multiple transfer learning models
- Incorporate attention mechanisms
- Deploy as a web application using Streamlit or Flask
- Explore regression-based ordinal loss for severity grading

---

## 📄 References

1. Gulshan et al. (2016). Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy. *JAMA*.
2. Tan & Le (2019). EfficientNet: Rethinking Model Scaling for CNNs. *ICML*.
3. He et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
4. APTOS 2019 Blindness Detection. Kaggle Competition.
5. Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. *ICCV*.

---

*Machine Learning Course Project — Phase 2*
