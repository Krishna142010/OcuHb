ConjunctivaScan 👁️

Non-Invasive Anemia Screening using Deep Learning on Conjunctival (Eye) Images

> ⚠️ DISCLAIMER: This is a research prototype for early screening ONLY. It is NOT a medical diagnostic tool and must not be used for clinical decisions.


📌 Overview

ConjunctivaScan is a deep learning-based system that analyzes conjunctival (inner eyelid) images to detect signs of anemia (pallor). The goal is to enable **low-cost, accessible, and non-invasive early screening**, especially in resource-limited settings.


❓ Problem

* 1.6 billion people are affected by anemia globally (WHO)
* Many cases remain undetected due to lack of access to blood tests (CBC)
* Traditional diagnosis requires lab infrastructure and costs


💡 Solution

We use computer vision + deep learning to analyze eye images and predict anemia risk:

* Input: Conjunctival eye image
* Output: Anemia risk (screening only)
* Supports uncertainty detection to flag inconclusive cases


🧠 Methodology

Pipeline

1. Data Collection (~1800 images)
2. Preprocessing (resize, normalize, augmentation)
3. Model Training (CNN + Transfer Learning)
4. Evaluation (AUC, F1, Sensitivity, Specificity)
5. Explainability (Grad-CAM)
6. Uncertainty Estimation (MC Dropout)


🏗️ Models Used

* Custom CNN (Baseline)
* MobileNetV2 (Transfer Learning)
* EfficientNetB0 ⭐ (Best Performance)
* ResNet50V2
* DenseNet121


📊 Results

| Model          | Accuracy | AUC   | Sensitivity | Specificity |
| -------------- | -------- | ----- | ----------- | ----------- |
| EfficientNetB0 | 89.4%    | 0.937 | 87.8%       | 91.0%       |

* Uncertainty filtering improves accuracy to **96.3%** on confident predictions
* Grad-CAM confirms model focuses on conjunctival region


🔍 Key Features

* ✅ Deep Learning-based screening
* ✅ Transfer learning for small datasets
* ✅ Grad-CAM explainability
* ✅ MC Dropout uncertainty estimation
* ✅ Fully reproducible pipeline


⚖️ Ethics & Limitations

This project is NOT:

* ❌ A diagnostic tool
* ❌ Clinically validated
* ❌ A replacement for blood tests

Limitations:

* Small dataset (~1800 images)
* Lighting variability affects predictions
* Possible demographic bias


📁 Project Structure

project/
├── anemia_detection.py
├── dataset/
├── best_model.keras
├── results/
└── README.md


🔬 Future Work

* Larger multi-ethnic dataset
* Hemoglobin regression model
* Mobile app deployment
* Clinical validation studies


📚 References

* WHO (2011) – Hemoglobin thresholds
* Grad-CAM (Selvaraju et al., 2017)
* MobileNet (Howard et al., 2017)

👨‍💻 Author

Krishna Bansal
Class 11 Student | AI & Research Enthusiast

⭐ Acknowledgment

This project was built as part of a science research initiative for IRIS/ISEF-level competitions, focusing on accessible healthcare using AI.
