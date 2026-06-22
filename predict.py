"""Unified prediction for clinical data and ultrasound images."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from src.config import ROOT_DIR, load_config
from src.data.clinical import FEATURE_COLUMNS
from src.models.image_model import build_image_model

IMAGE_PROB_TEMPERATURE = 2.5


def _risk_level(prob: float) -> str:
    if prob < 0.35:
        return "Low"
    if prob < 0.65:
        return "Moderate"
    return "High"


def _build_advice(prediction: str, risk_level: str, patient_data: dict | None = None) -> dict:
    """Generate lifestyle solutions and doctor consultation guidance."""
    is_nafld = prediction == "NAFLD"
    doctor_consult = (
        "This result is AI-generated and not a medical diagnosis. "
        "Please consult a qualified doctor or hepatologist for confirmation, "
        "further tests, and a personalized treatment plan."
    )

    if not is_nafld:
        solutions = [
            "Maintain a balanced diet rich in vegetables, whole grains, and lean protein.",
            "Exercise regularly — aim for at least 150 minutes of moderate activity per week.",
            "Keep BMI, blood sugar, cholesterol, and triglycerides within healthy ranges.",
            "Limit processed foods, sugary drinks, and saturated fats.",
            "Get periodic liver function tests (ALT, AST) during routine health checkups.",
            "Avoid excessive alcohol and maintain healthy sleep habits.",
        ]
        if patient_data:
            if patient_data.get("BMI", 0) >= 25:
                solutions.insert(0, "Your BMI is elevated — focus on gradual, sustainable weight management.")
            if patient_data.get("Diabetes_History") == 1:
                solutions.insert(1, "Monitor blood glucose regularly and follow your diabetes care plan.")
        return {"solutions": solutions, "doctor_consult": doctor_consult}

    if risk_level == "High":
        solutions = [
            "Schedule an appointment with a doctor or hepatologist as soon as possible.",
            "Follow a liver-friendly diet — Mediterranean-style meals are recommended.",
            "Reduce refined carbohydrates, fried foods, and sugary beverages immediately.",
            "Begin a structured weight-loss plan if overweight (even 5–10% loss helps the liver).",
            "Exercise daily — walking, cycling, or swimming for 30–45 minutes.",
            "Strictly control blood sugar, blood pressure, and cholesterol levels.",
            "Avoid self-medication and hepatotoxic drugs without medical supervision.",
            "Repeat liver ultrasound and LFTs (ALT, AST) as advised by your doctor.",
        ]
    elif risk_level == "Moderate":
        solutions = [
            "Consult a doctor for clinical evaluation and follow-up testing.",
            "Adopt a low-fat, high-fiber diet with reduced sugar and processed foods.",
            "Increase physical activity to at least 30 minutes per day, 5 days a week.",
            "Work toward a healthy BMI through gradual and sustainable weight loss.",
            "Monitor liver enzymes (ALT, AST) every 3–6 months.",
            "Manage diabetes, hypertension, and lipid levels with lifestyle and medical support.",
            "Limit alcohol intake and avoid unnecessary medications that stress the liver.",
        ]
    else:
        solutions = [
            "Consult a doctor to confirm findings and plan preventive care.",
            "Improve diet quality — more vegetables, fruits, and whole grains.",
            "Stay physically active with regular cardio and strength exercises.",
            "Maintain healthy body weight and waist circumference.",
            "Get follow-up liver function tests during your next health checkup.",
            "Reduce intake of sugary and ultra-processed foods.",
        ]

    if patient_data:
        if patient_data.get("BMI", 0) >= 30:
            solutions.insert(1, "Priority: reduce body weight — obesity is a major NAFLD driver.")
        if patient_data.get("ALT", 0) > 40 or patient_data.get("AST", 0) > 40:
            solutions.append("Your liver enzymes appear elevated — doctor review is important.")
        if patient_data.get("Glucose", 0) > 100:
            solutions.append("Keep blood glucose under control through diet, activity, and medical guidance.")
        if patient_data.get("Triglycerides", 0) > 150:
            solutions.append("Lower triglycerides with dietary changes and doctor-recommended interventions.")

    return {"solutions": solutions, "doctor_consult": doctor_consult}


def predict_clinical(model, patient_data: dict) -> dict:
    df = pd.DataFrame([patient_data])
    prob = float(model.predict_proba(df[FEATURE_COLUMNS])[0, 1])
    pred = int(prob >= 0.5)
    prediction = "NAFLD" if pred == 1 else "Non-NAFLD"
    risk_level = _risk_level(prob)
    advice = _build_advice(prediction, risk_level, patient_data)
    return {
        "prediction": prediction,
        "probability": prob,
        "risk_level": risk_level,
        "valid_input": True,
        "solutions": advice["solutions"],
        "doctor_consult": advice["doctor_consult"],
    }


def _is_ultrasound_like(image: Image.Image) -> tuple[bool, str]:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    gray = arr.mean(axis=2)

    # Natural photos are usually more colorful than grayscale ultrasound images.
    color_distance = float(np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(r - b)) / 3.0)
    brightness = float(gray.mean())
    bright_ratio = float((gray > 235).mean())
    dark_ratio = float((gray < 25).mean())

    # Ultrasound has speckle texture; too smooth images are likely non-medical photos.
    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    texture = float((dx.mean() + dy.mean()) / 2.0)

    if color_distance > 12.0:
        return False, "Uploaded image looks like a natural photo, not an ultrasound scan."
    if brightness < 8.0 or brightness > 130.0:
        return False, "Image brightness is not typical for liver ultrasound scans."
    if bright_ratio > 0.20:
        return False, "Image has too much white/bright area and does not look like an ultrasound frame."
    if dark_ratio < 0.12:
        return False, "Image does not have the dark background pattern typical in ultrasound scans."
    if texture < 2.0:
        return False, "Image texture is too smooth for a diagnostic ultrasound frame."
    return True, ""


def predict_image(model, image_path: str, img_size: int, device) -> dict:
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    is_valid, reason = _is_ultrasound_like(image)
    if not is_valid:
        return {
            "prediction": "Invalid Ultrasound Input",
            "probability": 0.0,
            "risk_level": "Low",
            "valid_input": False,
            "message": f"{reason} Please upload a liver ultrasound image.",
        }

    tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(tensor)
        # Keep class prediction from raw logits, but calibrate displayed probability.
        calibrated_probs = torch.softmax(outputs / IMAGE_PROB_TEMPERATURE, dim=1)[0].cpu().numpy()
        pred = int(torch.argmax(outputs, dim=1).item())

    nafld_prob = float(np.clip(calibrated_probs[1], 0.001, 0.999))
    prediction = "NAFLD" if pred == 1 else "Non-NAFLD"
    risk_level = _risk_level(nafld_prob)
    advice = _build_advice(prediction, risk_level)
    return {
        "prediction": prediction,
        "probability": nafld_prob,
        "risk_level": risk_level,
        "valid_input": True,
        "solutions": advice["solutions"],
        "doctor_consult": advice["doctor_consult"],
    }


def main():
    parser = argparse.ArgumentParser(description="NAFLD prediction")
    parser.add_argument("--clinical", action="store_true", help="Use clinical model")
    parser.add_argument("--image", type=str, help="Path to ultrasound image")
    args = parser.parse_args()

    cfg = load_config()
    models_dir = ROOT_DIR / cfg["models_dir"]

    if args.clinical:
        model = joblib.load(models_dir / "clinical_model.joblib")
        sample = {
            "Age": 52,
            "Gender": 0,
            "BMI": 32.5,
            "Weight_kg": 88.0,
            "Height_cm": 165.0,
            "Glucose": 110.0,
            "Cholesterol": 220.0,
            "Triglycerides": 180.0,
            "ALT": 45.0,
            "AST": 38.0,
            "Bilirubin": 0.9,
            "Albumin": 4.2,
            "Systolic_BP": 130,
            "Diastolic_BP": 85,
            "Diabetes_History": 1,
        }
        result = predict_clinical(model, sample)
        print("Clinical prediction:", result)

    if args.image:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(
            models_dir / "image_model.pt", map_location=device, weights_only=False
        )
        model = build_image_model(
            checkpoint["backbone"], pretrained=False
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        result = predict_image(model, args.image, checkpoint["img_size"], device)
        print("Image prediction:", result)


if __name__ == "__main__":
    main()
