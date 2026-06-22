# AI-Based Detection and Prediction of Non-Alcoholic Fatty Liver Disease (NAFLD)

## Overview

This project is an AI-powered healthcare application developed for the early detection and prediction of Non-Alcoholic Fatty Liver Disease (NAFLD). The system combines Machine Learning and Deep Learning techniques to analyze both clinical parameters and ultrasound liver images.

The application provides:

- Clinical Risk Prediction using Machine Learning
- Ultrasound Image Classification using Deep Learning
- Interactive Web Interface using Streamlit
- Automated NAFLD Risk Assessment

---

## Project Architecture

### Clinical Data Prediction

Clinical parameters such as:

- Age
- Gender
- BMI
- Weight
- Height
- Glucose
- Cholesterol
- Triglycerides
- ALT
- AST
- Bilirubin
- Albumin
- Blood Pressure
- Diabetes History

are processed using Machine Learning algorithms.

Multiple models were evaluated:

- Logistic Regression
- Random Forest
- XGBoost
- Gradient Boosting
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree
- Naive Bayes

### Final Clinical Model

**Logistic Regression**

Accuracy Achieved:

**70.5%**

---

### Ultrasound Image Classification

Liver ultrasound images are analyzed using Deep Learning.

Models evaluated:

- ResNet18
- Swin Transformer (Tiny, Small, Base)
- MobileNetV3

### Final Image Model

**ResNet18 CNN**

Performance:

| Metric | Value |
|----------|----------|
| Accuracy | 97.01% |
| Precision | 99.33% |
| Recall | 97.37% |
| F1 Score | 98.34% |
| ROC-AUC | 98.89% |

---

## Technologies Used

- Python
- Streamlit
- Scikit-Learn
- PyTorch
- Torchvision
- Pandas
- NumPy
- Matplotlib
- XGBoost

---

## Project Structure

## Project Structure

```text
NAFLD/
│
├── app.py                    # Streamlit web application
├── predict.py                # Prediction pipeline
├── config.yaml               # Model configuration
├── requirements.txt          # Dependencies
│
├── assets/                   # Sample ultrasound images
├── data/                     # Clinical and user data
├── models/                   # Trained ML and DL models
│   ├── clinical_model.joblib
│   └── image_model.pt
│
├── results/                  # Evaluation reports and metrics
│
├── src/
│   ├── data/
│   │   ├── clinical.py
│   │   └── imaging.py
│   │
│   ├── models/
│   │   ├── clinical_model.py
│   │   └── image_model.py
│   │
│   ├── auth.py
│   ├── config.py
│   └── evaluation.py
│
├── train_clinical.py
└── train_image.py
```
---

## Installation

Clone the repository:

```bash
git clone https://github.com/Rohanjaswal8/NAFLD.git
cd NAFLD
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run application:

```bash
streamlit run app.py
```

---

## Results

### Clinical Prediction

Final Model: Logistic Regression

Accuracy: 70.5%

### Ultrasound Classification

Final Model: ResNet18

Accuracy: 97.01%

---

## Live Demo

https://nafldprediction.streamlit.app

---

## Future Scope

- Integration with hospital databases
- Mobile Application Development
- Real-time Clinical Decision Support
- Multi-Disease Liver Screening

---

## Author

Rohan Jaswal

B.Tech Computer Science Engineering

Artificial Intelligence in Healthcare
