# AI-Based Detection and Prediction of NAFLD

Detection and prediction of **Non-Alcoholic Fatty Liver Disease (NAFLD)** using clinical tabular data and liver ultrasound imaging.

## Dataset

| Data | Location | Size |
|------|----------|------|
| Clinical records | `~/Downloads/nafld_dataset.csv` | 2,000 rows, 15 features |
| NAFLD ultrasound images | `~/Downloads/BEHSOF/NAFLD/` | 1,517 images |
| Non-NAFLD ultrasound images | `~/Downloads/BEHSOF/Non-NAFLD/` | 152 images |

### Clinical Features
Age, Gender, BMI, Weight, Height, Glucose, Cholesterol, Triglycerides, ALT, AST, Bilirubin, Albumin, Blood Pressure, Diabetes History

## Project Structure

```
nafld/
├── config.yaml           # Dataset paths and hyperparameters
├── requirements.txt
├── train_clinical.py     # Train and compare ML models on clinical data
├── train_image.py        # Train ResNet18 on ultrasound images
├── predict.py            # CLI prediction
├── app.py                # Streamlit web interface
├── src/
│   ├── data/             # Data loading pipelines
│   ├── models/           # Model definitions
│   └── evaluation.py     # Metrics and plots
├── models/               # Saved models (after training)
└── results/              # Evaluation plots and reports
```

## Setup

```bash
cd nafld
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**macOS (XGBoost):** If you see a `libomp` error, run:
```bash
brew install libomp
```

Update `config.yaml` if your dataset paths differ.

## Training

```bash
# Clinical model (~1-2 min, compares 8 models)
python train_clinical.py

# Swin Transformer image model (~15-30 min on CPU)
python train_image.py
```

## Prediction

```bash
# Clinical sample
python predict.py --clinical

# Ultrasound image
python predict.py --image /path/to/ultrasound.jpg
```

## Web App

```bash
streamlit run app.py
```

## Models

| Model | Algorithm | Input | Selection |
|-------|-----------|-------|-----------|
| Clinical | Logistic Regression, Random Forest, XGBoost, Gradient Boosting, SVM, KNN, Decision Tree, Naive Bayes | 15 clinical features | Best accuracy auto-selected |
| Image | Swin Transformer (transfer learning) | Ultrasound JPG | `swin_t` backbone |

## Disclaimer

This project is for **research and educational purposes only**. It is not a medical device and should not be used for clinical diagnosis.
