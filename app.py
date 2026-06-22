"""NAFLD Detection System — polished medical dashboard UI."""

import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch
from PIL import Image
from streamlit_javascript import st_javascript

from predict import predict_clinical, predict_image
from src.auth import login, logout, signup, validate_token
from src.config import ROOT_DIR, load_config
from src.models.image_model import build_image_model

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NAFLD AI Diagnostics",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

cfg = load_config()
models_dir = ROOT_DIR / cfg["models_dir"]
results_dir = ROOT_DIR / cfg["results_dir"]

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&family=Playfair+Display:wght@600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #f1f5f9;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding-top: 0.5rem;
    padding-bottom: 2rem;
    max-width: 100% !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* Hide sidebar — single top navbar only */
[data-testid="stSidebar"], [data-testid="collapsedControl"] {
    display: none !important;
}

/* Unified navbar row */
div[data-testid="stHorizontalBlock"]:has(.nav-brand-inline) {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 0.65rem 1rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 30px rgba(15, 23, 42, 0.08);
    position: sticky;
    top: 0.5rem;
    z-index: 999;
    align-items: center !important;
}

/* Nav buttons in navbar row only */
div[data-testid="stHorizontalBlock"]:has(.nav-brand-inline) .stButton > button[kind="secondary"] {
    background: #f8fafc !important;
    color: #334155 !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 0.55rem 0.5rem !important;
    transition: transform 0.22s cubic-bezier(0.34, 1.56, 0.64, 1),
                box-shadow 0.22s ease, background 0.22s ease, border-color 0.22s ease !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
}
div[data-testid="stHorizontalBlock"]:has(.nav-brand-inline) .stButton > button[kind="secondary"]:hover {
    transform: translateY(-4px) scale(1.06) !important;
    box-shadow: 0 12px 28px rgba(13, 148, 136, 0.18) !important;
    border-color: #0d9488 !important;
    background: #f0fdfa !important;
    color: #0f766e !important;
}
div[data-testid="stHorizontalBlock"]:has(.nav-brand-inline) .stButton > button[kind="primary"] {
    transform: translateY(-3px) scale(1.04) !important;
    box-shadow: 0 10px 24px rgba(13, 148, 136, 0.35) !important;
}
div[data-testid="stHorizontalBlock"]:has(.nav-brand-inline) .stButton > button:active {
    transform: translateY(-1px) scale(1.02) !important;
}

.nav-brand-inline {
    font-size: 1.2rem;
    font-weight: 700;
    color: #0f172a;
    padding: 0.55rem 0.25rem;
    white-space: nowrap;
}
.nav-brand-inline span { color: #0d9488; }

/* Zoom-up effect for cards */
.zoom-card {
    transition: transform 0.28s cubic-bezier(0.34, 1.56, 0.64, 1),
                box-shadow 0.28s ease, border-color 0.28s ease;
    will-change: transform;
}
.zoom-card:hover {
    transform: translateY(-10px) scale(1.03);
    box-shadow: 0 20px 40px rgba(15, 23, 42, 0.14);
}
.zoom-card:active {
    transform: translateY(-5px) scale(1.015);
}

/* Hero */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 45%, #0d9488 100%);
    border-radius: 24px;
    padding: 2.8rem 3rem;
    color: white;
    margin-bottom: 1.75rem;
    box-shadow: 0 24px 48px rgba(15, 23, 42, 0.28);
    position: relative;
    overflow: hidden;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.hero:hover {
    transform: translateY(-6px) scale(1.008);
    box-shadow: 0 32px 56px rgba(15, 23, 42, 0.32);
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(13,148,136,0.25) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 999px;
    padding: 0.35rem 1rem;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0 0 0.75rem 0;
    color: white;
    line-height: 1.2;
}
.hero p {
    font-size: 1.08rem;
    opacity: 0.92;
    margin: 0 0 1.25rem 0;
    color: #e2e8f0;
    max-width: 680px;
    line-height: 1.65;
}
.hero-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.hero-tag {
    background: rgba(255,255,255,0.12);
    border-radius: 8px;
    padding: 0.35rem 0.75rem;
    font-size: 0.82rem;
    color: #cbd5e1;
    transition: transform 0.22s ease, background 0.22s ease;
    display: inline-block;
}
.hero-tag:hover {
    transform: translateY(-3px) scale(1.05);
    background: rgba(255,255,255,0.22);
}

/* Stat cards */
.stat-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.35rem 1.5rem;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
    height: 100%;
}
.stat-card:hover {
    border-color: #99f6e4;
}
.stat-card .label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #64748b;
    margin-bottom: 0.4rem;
}
.stat-card .value {
    font-size: 1.85rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.1;
}
.stat-card .sub {
    font-size: 0.82rem;
    color: #94a3b8;
    margin-top: 0.35rem;
}
.stat-card.accent { border-left: 4px solid #0d9488; }
.stat-card.blue   { border-left: 4px solid #3b82f6; }
.stat-card.purple { border-left: 4px solid #8b5cf6; }
.stat-card.amber  { border-left: 4px solid #f59e0b; }

/* Info / disease cards */
.info-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 1.5rem 1.75rem;
    height: 100%;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
}
.info-card h3 {
    font-size: 1.05rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0 0 0.75rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.info-card p, .info-card li {
    font-size: 0.92rem;
    color: #475569;
    line-height: 1.65;
    margin: 0;
}
.info-card ul {
    margin: 0;
    padding-left: 1.2rem;
}
.info-card li { margin-bottom: 0.35rem; }

.disease-banner {
    background: linear-gradient(135deg, #ecfdf5 0%, #f0fdfa 100%);
    border: 1px solid #99f6e4;
    border-radius: 20px;
    padding: 2rem 2.25rem;
    margin-bottom: 1.5rem;
}
.disease-banner h2 {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    color: #0f766e;
    margin: 0 0 0.75rem 0;
}
.disease-banner p {
    color: #334155;
    font-size: 0.98rem;
    line-height: 1.75;
    margin: 0;
}

.feature-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 1.5rem;
    height: 100%;
    cursor: pointer;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
}
.feature-card:hover {
    border-color: #0d9488;
}
.feature-icon {
    font-size: 2rem;
    margin-bottom: 0.75rem;
    transition: transform 0.25s ease;
}
.feature-card:hover .feature-icon {
    transform: scale(1.15) translateY(-4px);
}
.feature-card h4 {
    font-size: 1.05rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0 0 0.5rem 0;
}
.feature-card p {
    font-size: 0.88rem;
    color: #64748b;
    line-height: 1.6;
    margin: 0;
}

.stage-pill {
    display: inline-block;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.5rem 0.85rem;
    font-size: 0.82rem;
    color: #475569;
    margin: 0.25rem 0.25rem 0 0;
    transition: transform 0.22s ease, background 0.22s ease, box-shadow 0.22s ease;
    cursor: default;
}
.stage-pill:hover {
    transform: translateY(-4px) scale(1.04);
    background: #ecfdf5;
    box-shadow: 0 8px 20px rgba(13, 148, 136, 0.12);
}

/* Section headers */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0 0 0.3rem 0;
}
.section-sub {
    color: #64748b;
    font-size: 0.94rem;
    margin-bottom: 1.35rem;
    line-height: 1.5;
}

/* Form panel */
.form-panel {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.5rem;
}

/* Result card */
.result-card {
    border-radius: 18px;
    padding: 1.75rem;
    text-align: center;
    margin-top: 1rem;
}
.result-card.low {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border: 2px solid #6ee7b7;
}
.result-card.moderate {
    background: linear-gradient(135deg, #fffbeb, #fef3c7);
    border: 2px solid #fcd34d;
}
.result-card.high {
    background: linear-gradient(135deg, #fef2f2, #fecaca);
    border: 2px solid #f87171;
}
.result-card .pred-label {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
}
.result-card .pred-value {
    font-size: 2rem;
    font-weight: 800;
    margin: 0.4rem 0;
}
.result-card .prob {
    font-size: 1.1rem;
    font-weight: 600;
    color: #334155;
}

.upload-hint {
    border: 2px dashed #cbd5e1;
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    color: #64748b;
    background: #f8fafc;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] hr { border-color: #334155; }

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0d9488, #0f766e) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.65rem 2rem !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(13, 148, 136, 0.35) !important;
    transition: transform 0.22s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.22s ease !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-4px) scale(1.04) !important;
    box-shadow: 0 10px 28px rgba(13, 148, 136, 0.45) !important;
}
.stButton > button[kind="primary"]:active {
    transform: translateY(-1px) scale(1.01) !important;
}

[data-testid="stMetricValue"] { font-weight: 700; }

.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    background: #ccfbf1;
    color: #0f766e;
}

.page-header {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to { opacity: 1; transform: translateY(0); }
}
.page-content {
    animation: fadeUp 0.45s ease;
}

.advice-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 1.35rem 1.5rem;
    margin-top: 1rem;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
}
.advice-card h4 {
    font-size: 1rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0 0 0.85rem 0;
}
.advice-card ul {
    margin: 0;
    padding-left: 1.25rem;
}
.advice-card li {
    font-size: 0.92rem;
    color: #475569;
    line-height: 1.65;
    margin-bottom: 0.45rem;
}
.doctor-alert {
    background: linear-gradient(135deg, #fff7ed, #ffedd5);
    border: 1px solid #fdba74;
    border-radius: 16px;
    padding: 1.1rem 1.25rem;
    margin-top: 1rem;
    color: #9a3412;
    font-size: 0.92rem;
    line-height: 1.6;
    font-weight: 500;
}
.doctor-alert strong {
    color: #c2410c;
}

/* Auth page */
.auth-wrap {
    max-width: 460px;
    margin: 2rem auto 3rem auto;
}
.auth-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 22px;
    padding: 2rem 2rem 1.5rem 2rem;
    box-shadow: 0 16px 40px rgba(15, 23, 42, 0.1);
}
.auth-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.75rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0 0 0.35rem 0;
    text-align: center;
}
.auth-sub {
    text-align: center;
    color: #64748b;
    font-size: 0.92rem;
    margin-bottom: 1.25rem;
}
.liver-compare-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
}
.liver-compare-card h4 {
    margin: 0.75rem 0 0.25rem 0;
    font-size: 1rem;
    color: #0f172a;
}
.liver-compare-card p {
    margin: 0;
    font-size: 0.82rem;
    color: #64748b;
}
.liver-compare-card.nafld { border-top: 4px solid #ef4444; }
.liver-compare-card.non-nafld { border-top: 4px solid #10b981; }

/* About page — full width */
.about-hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0d9488 100%);
    border-radius: 24px;
    padding: 3rem 3.5rem;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 24px 48px rgba(15, 23, 42, 0.2);
}
.about-hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.5rem;
    margin: 0 0 0.75rem 0;
    color: white;
}
.about-hero p {
    font-size: 1.1rem;
    color: #e2e8f0;
    line-height: 1.7;
    max-width: 800px;
    margin: 0;
}
.about-grid-3 {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.25rem;
    margin-bottom: 1.5rem;
}
@media (max-width: 900px) {
    .about-grid-3 { grid-template-columns: 1fr; }
}
.about-tile {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 1.5rem;
    height: 100%;
    box-shadow: 0 4px 20px rgba(15, 23, 42, 0.05);
    transition: transform 0.28s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.28s ease;
}
.about-tile:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 16px 40px rgba(15, 23, 42, 0.12);
}
.about-tile .icon { font-size: 2rem; margin-bottom: 0.75rem; }
.about-tile h3 {
    font-size: 1.1rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0 0 0.5rem 0;
}
.about-tile p {
    font-size: 0.9rem;
    color: #64748b;
    line-height: 1.65;
    margin: 0;
}
.about-section {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 2rem 2.25rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(15, 23, 42, 0.04);
}
.about-section h2 {
    font-family: 'Playfair Display', serif;
    font-size: 1.45rem;
    color: #0f172a;
    margin: 0 0 1rem 0;
    padding-bottom: 0.75rem;
    border-bottom: 2px solid #f1f5f9;
}
.about-section p, .about-section li {
    font-size: 0.95rem;
    color: #475569;
    line-height: 1.75;
}
.about-section ul { margin: 0.5rem 0 0 0; padding-left: 1.25rem; }
.about-section li { margin-bottom: 0.4rem; }
.tech-badge {
    display: inline-block;
    background: #f0fdfa;
    border: 1px solid #99f6e4;
    color: #0f766e;
    border-radius: 999px;
    padding: 0.35rem 0.85rem;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 0.25rem 0.25rem 0 0;
}
.pipeline-step {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.1rem 1.25rem;
    text-align: center;
    height: 100%;
}
.pipeline-step .num {
    background: #0d9488;
    color: white;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.9rem;
    margin-bottom: 0.5rem;
}
.pipeline-step h4 {
    font-size: 0.95rem;
    color: #0f172a;
    margin: 0 0 0.35rem 0;
}
.pipeline-step p {
    font-size: 0.82rem;
    color: #64748b;
    margin: 0;
    line-height: 1.5;
}
.about-disclaimer {
    background: linear-gradient(135deg, #fff7ed, #ffedd5);
    border: 1px solid #fdba74;
    border-radius: 18px;
    padding: 1.5rem 2rem;
    color: #9a3412;
    font-size: 0.95rem;
    line-height: 1.7;
}
.about-disclaimer strong { color: #c2410c; }
</style>
<script>
document.addEventListener('click', function(e) {
    const btn = e.target.closest('button');
    if (btn) {
        btn.style.transform = 'translateY(-2px) scale(0.97)';
        setTimeout(() => { btn.style.transform = ''; }, 150);
    }
});
</script>
""",
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_best_clinical_info():
    path = results_dir / "clinical_best_model.json"
    if not path.exists():
        return None, None
    with open(path) as f:
        data = json.load(f)
    return data.get("best_model"), data.get("accuracy")


@st.cache_resource
def load_clinical_model():
    path = models_dir / "clinical_model.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_resource
def load_image_model(model_mtime: float):
    path = models_dir / "image_model.pt"
    if not path.exists():
        return None, None, None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_image_model(checkpoint["backbone"], pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint["img_size"], device, checkpoint.get("backbone", "resnet18")


@st.cache_data
def load_model_comparison():
    path = results_dir / "clinical_model_comparison.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_metrics(name: str):
    path = results_dir / f"{name}_metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def risk_class(level: str) -> str:
    return {"Low": "low", "Moderate": "moderate", "High": "high"}.get(level, "moderate")


def display_probability(prob: float) -> float:
    """Compress extreme probabilities for more readable UI output."""
    p = float(np.clip(prob, 1e-6, 1 - 1e-6))
    logit = np.log(p / (1 - p))
    smoothed = 1.0 / (1.0 + np.exp(-logit / 6.0))
    return float(np.clip(smoothed, 0.05, 0.95))


def render_result_card(result: dict):
    if not result.get("valid_input", True):
        st.error(result.get("message", "Please upload a valid liver ultrasound image."))
        return

    css = risk_class(result["risk_level"])
    color = {"low": "#059669", "moderate": "#d97706", "high": "#dc2626"}[css]
    shown_prob = display_probability(result["probability"])
    st.markdown(
        f"""
        <div class="result-card zoom-card {css}">
            <div class="pred-label">Prediction</div>
            <div class="pred-value" style="color:{color}">{result['prediction']}</div>
            <div class="prob">NAFLD Probability: <strong>{shown_prob:.2%}</strong></div>
            <div style="margin-top:0.75rem">
                <span class="badge" style="background:{'#d1fae5' if css=='low' else '#fef3c7' if css=='moderate' else '#fecaca'};
                     color:{color}">Risk: {result['risk_level']}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(shown_prob)

    solutions = result.get("solutions", [])
    if solutions:
        items = "".join(f"<li>{item}</li>" for item in solutions)
        st.markdown(
            f"""
            <div class="advice-card zoom-card">
                <h4>💡 Recommended Solutions & Lifestyle Changes</h4>
                <ul>{items}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    doctor_msg = result.get("doctor_consult")
    if doctor_msg:
        st.markdown(
            f"""
            <div class="doctor-alert zoom-card">
                <strong>⚕️ Important — Consult a Doctor</strong><br>
                {doctor_msg}
            </div>
            """,
            unsafe_allow_html=True,
        )


def stat_card(label, value, sub="", style="accent"):
    st.markdown(
        f"""
        <div class="stat-card zoom-card {style}">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def parse_option(val: str) -> int:
    return int(val.split("(")[1].rstrip(")"))


def save_auth_to_browser(token: str) -> None:
    components.html(
        f"""
        <script>
        localStorage.setItem("nafld_auth", "{token}");
        </script>
        """,
        height=0,
    )


def clear_auth_from_browser() -> None:
    components.html(
        """
        <script>
        localStorage.removeItem("nafld_auth");
        </script>
        """,
        height=0,
    )


def try_restore_session() -> None:
    """Restore login from browser localStorage without blocking the home page."""
    if st.session_state.get("authenticated"):
        return

    if not st.session_state.get("auth_lookup_done"):
        token = st_javascript("localStorage.getItem('nafld_auth')", key="read_nafld_auth")
        if token is None:
            return
        st.session_state.auth_lookup_done = True
        st.session_state.pending_auth_token = token

    token = st.session_state.get("pending_auth_token")
    if token and token not in ("null", ""):
        profile = validate_token(token)
        if profile:
            st.session_state.authenticated = True
            st.session_state.user = profile


def show_auth_page() -> None:
    st.markdown(
        """
        <div class="auth-wrap">
            <div class="auth-card zoom-card">
                <div class="auth-title">🫀 Welcome Back</div>
                <div class="auth-sub">Login or sign up to access clinical analysis, ultrasound scan & more</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_login, tab_signup = st.tabs(["Login", "Sign Up"])
    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="you@example.com")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", type="primary", use_container_width=True)
        if submitted:
            ok, msg, profile, token = login(email, password)
            if ok and profile and token:
                st.session_state.authenticated = True
                st.session_state.user = {**profile, "token": token}
                st.session_state.pending_auth_token = token
                st.session_state.auth_lookup_done = True
                st.session_state.active_page = "clinical"
                save_auth_to_browser(token)
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    with tab_signup:
        with st.form("signup_form"):
            name = st.text_input("Full Name", placeholder="Your name")
            email = st.text_input("Email ", placeholder="you@example.com")
            password = st.text_input("Password ", type="password")
            confirm = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Create Account", type="primary", use_container_width=True)
        if submitted:
            if password != confirm:
                st.error("Passwords do not match.")
            else:
                ok, msg = signup(name, email, password)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)


PROTECTED_PAGES = {"clinical", "ultrasound", "analytics", "about"}

if "active_page" not in st.session_state:
    st.session_state.active_page = "home"
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user" not in st.session_state:
    st.session_state.user = {}

try_restore_session()

if st.session_state.active_page in PROTECTED_PAGES and not st.session_state.authenticated:
    st.session_state.active_page = "auth"

NAV_ITEMS = [("home", "🏠 Home")]
if st.session_state.authenticated:
    NAV_ITEMS += [
        ("clinical", "🩺 Clinical"),
        ("ultrasound", "📷 Ultrasound"),
        ("analytics", "📊 Analytics"),
        ("about", "ℹ️ About"),
    ]
else:
    NAV_ITEMS.append(("auth", "🔐 Login / Sign Up"))

nav_col_weights = [2.2] + [1] * len(NAV_ITEMS)
if st.session_state.authenticated:
    nav_col_weights.append(0.7)
nav_cols = st.columns(nav_col_weights, gap="small")

with nav_cols[0]:
    if st.session_state.authenticated:
        user_name = st.session_state.user.get("name", "User")
        brand = f'🫀 <span>NAFLD</span> AI · {user_name}'
    else:
        brand = '🫀 <span>NAFLD</span> AI Diagnostics'
    st.markdown(f'<div class="nav-brand-inline">{brand}</div>', unsafe_allow_html=True)

for idx, (page_key, page_label) in enumerate(NAV_ITEMS):
    with nav_cols[idx + 1]:
        is_active = st.session_state.active_page == page_key
        if st.button(
            page_label,
            key=f"nav_{page_key}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.active_page = page_key
            st.rerun()

if st.session_state.authenticated:
    with nav_cols[-1]:
        if st.button("Logout", key="logout_btn", use_container_width=True):
            logout(st.session_state.user.get("token"))
            clear_auth_from_browser()
            st.session_state.authenticated = False
            st.session_state.user = {}
            st.session_state.pending_auth_token = None
            st.session_state.auth_lookup_done = False
            st.session_state.active_page = "home"
            st.rerun()

page = st.session_state.active_page
st.markdown('<div class="page-content">', unsafe_allow_html=True)

if page == "auth":
    show_auth_page()

elif page == "home":
    st.markdown(
        """
        <div class="hero zoom-card">
            <div class="hero-badge">Liver Health · AI Screening Platform</div>
            <h1>Non-Alcoholic Fatty Liver Disease (NAFLD) Detection</h1>
            <p>An intelligent screening system that analyzes clinical biomarkers and liver ultrasound
            images to help identify fatty liver disease early — before serious complications develop.</p>
            <div class="hero-tags">
                <span class="hero-tag">🩺 Clinical Biomarkers</span>
                <span class="hero-tag">📷 Ultrasound Imaging</span>
                <span class="hero-tag">🤖 Machine Learning</span>
                <span class="hero-tag">⚡ Instant Results</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fatty_img = ROOT_DIR / "assets" / "liver_samples" / "fatty_liver.jpg"
    non_fatty_img = ROOT_DIR / "assets" / "liver_samples" / "non_fatty_liver.jpg"
    if fatty_img.exists() and non_fatty_img.exists():
        st.markdown('<p class="section-title">Ultrasound Comparison</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-sub">Sample liver ultrasound scans — fatty liver (NAFLD) vs healthy liver</p>',
            unsafe_allow_html=True,
        )
        img1, img2 = st.columns(2)
        with img1:
            st.image(str(fatty_img), caption="Fatty Liver (NAFLD)", use_container_width=True)
            st.markdown(
                """
                <div class="liver-compare-card zoom-card nafld">
                    <h4>🔴 Fatty Liver (NAFLD)</h4>
                    <p>Increased echogenicity (bright liver) with blurred vessel borders on ultrasound</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with img2:
            st.image(str(non_fatty_img), caption="Non-Fatty Liver (Healthy)", use_container_width=True)
            st.markdown(
                """
                <div class="liver-compare-card zoom-card non-nafld">
                    <h4>🟢 Non-Fatty Liver (Healthy)</h4>
                    <p>Normal liver echotexture with clear vascular pattern and homogeneous appearance</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="disease-banner zoom-card">
            <h2>What is NAFLD?</h2>
            <p>
                <strong>Non-Alcoholic Fatty Liver Disease (NAFLD)</strong> is a condition where excess fat
                builds up in the liver of people who drink little or no alcohol. It is one of the most common
                chronic liver diseases worldwide, affecting nearly <strong>25–30% of the global population</strong>.
                NAFLD often develops silently with no obvious symptoms in early stages, making early detection
                through screening critical for preventing progression to more severe liver damage.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown(
            """
            <div class="info-card zoom-card">
                <h3>⚠️ Common Symptoms</h3>
                <ul>
                    <li>Fatigue and general weakness</li>
                    <li>Discomfort in upper right abdomen</li>
                    <li>Unexplained weight gain</li>
                    <li>Elevated liver enzymes (ALT, AST)</li>
                    <li>Often no symptoms in early stages</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with d2:
        st.markdown(
            """
            <div class="info-card zoom-card">
                <h3>🔍 Risk Factors</h3>
                <ul>
                    <li>Obesity and high BMI</li>
                    <li>Type 2 Diabetes</li>
                    <li>High cholesterol & triglycerides</li>
                    <li>Metabolic syndrome</li>
                    <li>Sedentary lifestyle</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with d3:
        st.markdown(
            """
            <div class="info-card zoom-card">
                <h3>🛡️ Prevention Tips</h3>
                <ul>
                    <li>Maintain healthy body weight</li>
                    <li>Regular physical exercise</li>
                    <li>Balanced, low-sugar diet</li>
                    <li>Control blood sugar & lipids</li>
                    <li>Limit processed foods</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-card zoom-card" style="margin-bottom:1.5rem;">
            <h3>📈 Disease Progression Stages</h3>
            <p style="margin-bottom:0.75rem;">NAFLD can progress through several stages if left untreated:</p>
            <span class="stage-pill"><strong>Stage 1:</strong> Simple Steatosis (fat accumulation)</span>
            <span class="stage-pill"><strong>Stage 2:</strong> NASH (inflammation)</span>
            <span class="stage-pill"><strong>Stage 3:</strong> Fibrosis (scarring)</span>
            <span class="stage-pill"><strong>Stage 4:</strong> Cirrhosis (severe damage)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    best_name, best_acc = load_best_clinical_info()
    img_metrics = load_metrics("image")
    clin_metrics = load_metrics("clinical")

    st.markdown('<p class="section-title">System Overview</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Real-time AI models trained on clinical and imaging data</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stat_card("Clinical Accuracy", f"{best_acc:.1%}" if best_acc else "—", best_name or "Not trained", "accent")
    with c2:
        img_acc = img_metrics["accuracy"] if img_metrics else None
        stat_card("Ultrasound Accuracy", f"{img_acc:.1%}" if img_acc else "—", "ResNet18 CNN", "blue")
    with c3:
        stat_card("Clinical Records", "2,000", "15 biomarker features", "purple")
    with c4:
        stat_card("Ultrasound Images", "1,669", "NAFLD + Non-NAFLD", "amber")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">How It Works</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Two complementary screening pathways for comprehensive NAFLD assessment</p>', unsafe_allow_html=True)

    q1, q2 = st.columns(2)
    with q1:
        st.markdown(
            """
            <div class="feature-card zoom-card">
                <div class="feature-icon">🩺</div>
                <h4>Clinical Analysis</h4>
                <p>Enter patient vitals, blood pressure, lab values (ALT, AST, glucose, cholesterol)
                and diabetes history. Our system compares 8 ML models and uses the best one to
                predict NAFLD risk instantly.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with q2:
        st.markdown(
            """
            <div class="feature-card zoom-card">
                <div class="feature-icon">📷</div>
                <h4>Ultrasound Scan</h4>
                <p>Upload a liver ultrasound image. Our ResNet18 deep learning model analyzes the scan
                to detect fatty liver patterns. Only valid ultrasound images are accepted —
                random photos are automatically rejected.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if clin_metrics and img_metrics:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-title">Model Performance</p>', unsafe_allow_html=True)
        overview = pd.DataFrame({
            "Model": ["Clinical (Best)", "Ultrasound (ResNet18)"],
            "Accuracy": [clin_metrics["accuracy"], img_metrics["accuracy"]],
            "F1 Score": [clin_metrics["f1"], img_metrics["f1"]],
            "ROC-AUC": [clin_metrics["roc_auc"], img_metrics["roc_auc"]],
        })
        chart = (
            alt.Chart(overview.melt("Model", var_name="Metric", value_name="Score"))
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
            .encode(
                x=alt.X("Metric:N", title=None),
                y=alt.Y("Score:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Model:N", scale=alt.Scale(range=["#0d9488", "#3b82f6"])),
                xOffset="Model:N",
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)

    if not st.session_state.authenticated:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("🔐 Login or sign up to access Clinical Analysis, Ultrasound Scan, and more.")
        if st.button("Continue to Login / Sign Up", type="primary", use_container_width=True):
            st.session_state.active_page = "auth"
            st.rerun()


elif page == "clinical":
    st.markdown(
        """
        <div class="page-header zoom-card">
            <p class="section-title" style="margin:0;">Clinical Data Analysis</p>
            <p class="section-sub" style="margin:0.5rem 0 0 0;">
                Enter patient biomarkers for AI-powered NAFLD risk assessment
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    clinical_model = load_clinical_model()
    best_name, best_acc = load_best_clinical_info()

    if clinical_model is None:
        st.error("Clinical model not found. Run `python train_clinical.py` first.")
    else:
        if best_name:
            st.success(f"Active model: **{best_name.replace('_', ' ').title()}** · Accuracy: **{best_acc:.1%}**")

        with st.form("clinical_form", clear_on_submit=False):
            st.markdown('<div class="form-panel">', unsafe_allow_html=True)

            st.markdown("**👤 Demographics**")
            d1, d2, d3 = st.columns(3)
            with d1:
                age = st.number_input("Age (years)", 18, 100, 50)
                gender = st.selectbox("Gender", ["Female (0)", "Male (1)"])
            with d2:
                bmi = st.number_input("BMI (kg/m²)", 15.0, 50.0, 28.0, step=0.1)
                weight = st.number_input("Weight (kg)", 40.0, 150.0, 75.0, step=0.1)
            with d3:
                height = st.number_input("Height (cm)", 140.0, 200.0, 165.0, step=0.1)
                diabetes = st.selectbox("Diabetes History", ["No (0)", "Yes (1)"])

            st.markdown("**🩸 Blood Pressure**")
            bp1, bp2 = st.columns(2)
            with bp1:
                systolic = st.number_input("Systolic BP (mmHg)", 80, 200, 120)
            with bp2:
                diastolic = st.number_input("Diastolic BP (mmHg)", 50, 130, 80)

            st.markdown("**🔬 Laboratory Results**")
            l1, l2, l3 = st.columns(3)
            with l1:
                glucose = st.number_input("Glucose (mg/dL)", 50.0, 300.0, 95.0)
                cholesterol = st.number_input("Cholesterol (mg/dL)", 100.0, 350.0, 200.0)
                triglycerides = st.number_input("Triglycerides (mg/dL)", 30.0, 500.0, 150.0)
            with l2:
                alt_val = st.number_input("ALT (U/L)", 5.0, 200.0, 30.0)
                ast_val = st.number_input("AST (U/L)", 5.0, 200.0, 25.0)
                bilirubin = st.number_input("Bilirubin (mg/dL)", 0.1, 5.0, 0.8, step=0.01)
            with l3:
                albumin = st.number_input("Albumin (g/dL)", 2.0, 6.0, 4.2, step=0.01)

            st.markdown("</div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("🔍 Analyze Patient Data", type="primary", use_container_width=True)

        if submitted:
            patient = {
                "Age": age, "Gender": parse_option(gender),
                "BMI": bmi, "Weight_kg": weight, "Height_cm": height,
                "Glucose": glucose, "Cholesterol": cholesterol,
                "Triglycerides": triglycerides, "ALT": alt_val, "AST": ast_val,
                "Bilirubin": bilirubin, "Albumin": albumin,
                "Systolic_BP": systolic, "Diastolic_BP": diastolic,
                "Diabetes_History": parse_option(diabetes),
            }
            result = predict_clinical(clinical_model, patient)
            st.markdown("---")
            render_result_card(result)

            with st.expander("View entered data"):
                st.dataframe(pd.DataFrame([patient]).T.rename(columns={0: "Value"}), use_container_width=True)


elif page == "ultrasound":
    st.markdown(
        """
        <div class="page-header zoom-card">
            <p class="section-title" style="margin:0;">Ultrasound Image Analysis</p>
            <p class="section-sub" style="margin:0.5rem 0 0 0;">
                Upload a liver ultrasound scan for deep learning based NAFLD detection
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning("⚠️ Upload only liver ultrasound scans. Face photos and random images are rejected automatically.")

    model_path = models_dir / "image_model.pt"
    model_mtime = model_path.stat().st_mtime if model_path.exists() else 0.0
    image_model, img_size, device, backbone = load_image_model(model_mtime)

    if image_model is None:
        st.error("Image model not found. Run `python train_image.py` first.")
    else:
        st.info(f"Model: **{backbone}** (pretrained ResNet) · Input: {img_size}×{img_size}px")

        col_img, col_result = st.columns([1.1, 0.9])

        with col_img:
            uploaded = st.file_uploader(
                "Drop your ultrasound image here",
                type=["jpg", "jpeg", "png"],
                help="Supported formats: JPG, JPEG, PNG",
            )
            if uploaded is None:
                st.markdown(
                    '<div class="upload-hint">📁 Drag & drop or click to upload<br><small>JPG · PNG · Max recommended 10MB</small></div>',
                    unsafe_allow_html=True,
                )
            else:
                image = Image.open(uploaded).convert("RGB")
                st.image(image, caption="Uploaded Ultrasound", use_container_width=True)

        with col_result:
            if uploaded:
                with st.spinner("Analyzing ultrasound image..."):
                    temp_path = ROOT_DIR / "temp_upload.jpg"
                    image.save(temp_path)
                    result = predict_image(image_model, str(temp_path), img_size, device)
                    temp_path.unlink(missing_ok=True)
                render_result_card(result)

                img_m = load_metrics("image")
                if img_m:
                    st.markdown("**Model Performance**")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Accuracy", f"{img_m['accuracy']:.1%}")
                    m2.metric("F1 Score", f"{img_m['f1']:.1%}")
                    m3.metric("ROC-AUC", f"{img_m['roc_auc']:.1%}")


elif page == "analytics":
    st.markdown(
        """
        <div class="page-header zoom-card">
            <p class="section-title" style="margin:0;">Model Analytics</p>
            <p class="section-sub" style="margin:0.5rem 0 0 0;">
                Performance metrics and comparison across all trained models
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    comparison = load_model_comparison()
    img_m = load_metrics("image")
    clin_m = load_metrics("clinical")

    tab_clin, tab_img = st.tabs(["Clinical Models Comparison", "Ultrasound Model"])

    with tab_clin:
        if comparison is None:
            st.warning("Run `python train_clinical.py` to generate comparison data.")
        else:
            comparison["model"] = comparison["model"].str.replace("_", " ").str.title()
            chart = (
                alt.Chart(comparison)
                .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
                .encode(
                    x=alt.X("accuracy:Q", title="Accuracy", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("model:N", sort="-x", title=None),
                    color=alt.condition(
                        alt.datum.accuracy == comparison["accuracy"].max(),
                        alt.value("#0d9488"),
                        alt.value("#94a3b8"),
                    ),
                    tooltip=["model", "accuracy", "f1", "roc_auc"],
                )
                .properties(height=380, title="Clinical Model Accuracy Ranking")
            )
            st.altair_chart(chart, use_container_width=True)

            st.dataframe(
                comparison.rename(columns={
                    "model": "Model", "accuracy": "Accuracy",
                    "precision": "Precision", "recall": "Recall",
                    "f1": "F1", "roc_auc": "ROC-AUC",
                }).style.format({
                    "Accuracy": "{:.1%}", "Precision": "{:.1%}",
                    "Recall": "{:.1%}", "F1": "{:.1%}", "ROC-AUC": "{:.1%}",
                }),
                use_container_width=True,
            )

            roc_path = results_dir / "clinical_roc_curve.png"
            cm_path = results_dir / "clinical_confusion_matrix.png"
            if roc_path.exists() and cm_path.exists():
                g1, g2 = st.columns(2)
                with g1:
                    st.image(str(roc_path), caption="ROC Curve — Best Clinical Model")
                with g2:
                    st.image(str(cm_path), caption="Confusion Matrix — Best Clinical Model")

    with tab_img:
        if img_m is None:
            st.warning("Run `python train_image.py` to train the ultrasound model.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{img_m['accuracy']:.1%}")
            c2.metric("Precision", f"{img_m['precision']:.1%}")
            c3.metric("Recall", f"{img_m['recall']:.1%}")
            c4.metric("ROC-AUC", f"{img_m['roc_auc']:.1%}")

            roc_path = results_dir / "image_roc_curve.png"
            cm_path = results_dir / "image_confusion_matrix.png"
            if roc_path.exists() and cm_path.exists():
                g1, g2 = st.columns(2)
                with g1:
                    st.image(str(roc_path), caption="Ultrasound Model — ROC Curve")
                with g2:
                    st.image(str(cm_path), caption="Ultrasound Model — Confusion Matrix")


elif page == "about":
    best_name, best_acc = load_best_clinical_info()
    img_metrics = load_metrics("image")
    clin_metrics = load_metrics("clinical")

    st.markdown(
        """
        <div class="about-hero zoom-card">
            <h1>About NAFLD AI Diagnostics</h1>
            <p>A comprehensive AI-powered platform for early detection and screening of
            Non-Alcoholic Fatty Liver Disease — built with clinical machine learning
            and deep learning ultrasound analysis to make liver health screening more accessible.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="about-grid-3">
            <div class="about-tile zoom-card">
                <div class="icon">🎯</div>
                <h3>Our Mission</h3>
                <p>Enable early NAFLD detection before irreversible liver damage occurs,
                using affordable AI screening accessible to patients and healthcare workers.</p>
            </div>
            <div class="about-tile zoom-card">
                <div class="icon">🔬</div>
                <h3>Dual-Modality AI</h3>
                <p>Combines clinical biomarker analysis (blood tests, vitals) with liver
                ultrasound imaging for more comprehensive fatty liver assessment.</p>
            </div>
            <div class="about-tile zoom-card">
                <div class="icon">🛡️</div>
                <h3>Patient Safety</h3>
                <p>Provides screening guidance and lifestyle recommendations, always
                directing users to consult qualified doctors for confirmed diagnosis.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="about-section zoom-card">
            <h2>🫀 What Problem We Solve</h2>
            <p><strong>NAFLD (Non-Alcoholic Fatty Liver Disease)</strong> affects nearly 25–30% of the
            global population, yet most cases go undetected because early stages show no symptoms.
            Fat accumulates in the liver without alcohol consumption, and if untreated can progress to
            NASH, fibrosis, and cirrhosis. Our platform helps identify at-risk individuals early
            through two screening pathways:</p>
            <ul>
                <li><strong>Clinical Screening</strong> — Uses patient demographics, BMI, blood pressure,
                glucose, cholesterol, liver enzymes (ALT, AST), and diabetes history.</li>
                <li><strong>Ultrasound Screening</strong> — Analyzes liver ultrasound scans using deep
                learning to detect fatty liver patterns with high accuracy.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="section-title">How the System Works</p>', unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    steps = [
        ("1", "Data Input", "Enter clinical biomarkers or upload a liver ultrasound scan"),
        ("2", "AI Analysis", "ML/DL models process data and generate NAFLD risk prediction"),
        ("3", "Results", "Get prediction, probability score, and risk level instantly"),
        ("4", "Guidance", "Receive lifestyle solutions and doctor consultation advice"),
    ]
    for col, (num, title, desc) in zip([p1, p2, p3, p4], steps):
        with col:
            st.markdown(
                f"""
                <div class="pipeline-step zoom-card">
                    <div class="num">{num}</div>
                    <h4>{title}</h4>
                    <p>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    tech1, tech2 = st.columns(2)
    with tech1:
        img_acc_str = f"{img_metrics['accuracy']:.1%}" if img_metrics else "—"
        clin_acc_str = f"{best_acc:.1%}" if best_acc else "—"
        st.markdown(
            f"""
            <div class="about-section zoom-card">
                <h2>🧠 AI & Technology Stack</h2>
                <p><strong>Clinical Module</strong></p>
                <p>8 machine learning models are trained and compared automatically. The best
                model by accuracy is selected for predictions.</p>
                <p style="margin-top:0.75rem;">
                    <span class="tech-badge">Logistic Regression</span>
                    <span class="tech-badge">XGBoost</span>
                    <span class="tech-badge">Random Forest</span>
                    <span class="tech-badge">SVM</span>
                    <span class="tech-badge">KNN</span>
                    <span class="tech-badge">Gradient Boosting</span>
                </p>
                <p style="margin-top:1rem;"><strong>Accuracy:</strong> {clin_acc_str} ({best_name or 'N/A'})</p>
                <br>
                <p><strong>Ultrasound Module</strong></p>
                <p>Pretrained ResNet18 CNN with transfer learning, class balancing, and
                ultrasound input validation to reject non-medical images.</p>
                <p style="margin-top:0.75rem;">
                    <span class="tech-badge">ResNet18</span>
                    <span class="tech-badge">PyTorch</span>
                    <span class="tech-badge">Transfer Learning</span>
                    <span class="tech-badge">BEHSOF Dataset</span>
                </p>
                <p style="margin-top:1rem;"><strong>Accuracy:</strong> {img_acc_str}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with tech2:
        st.markdown(
            """
            <div class="about-section zoom-card">
                <h2>📂 Datasets Used</h2>
                <p><strong>Clinical Dataset</strong></p>
                <ul>
                    <li>2,000 patient records with 15 biomarker features</li>
                    <li>Age, Gender, BMI, Weight, Height, Glucose, Cholesterol</li>
                    <li>Triglycerides, ALT, AST, Bilirubin, Albumin</li>
                    <li>Systolic/Diastolic BP, Diabetes History</li>
                </ul>
                <p style="margin-top:1rem;"><strong>BEHSOF Ultrasound Dataset</strong></p>
                <ul>
                    <li>1,669 liver ultrasound images total</li>
                    <li>1,517 NAFLD (fatty liver) images</li>
                    <li>152 Non-NAFLD (healthy liver) images</li>
                    <li>Used for training and validating the CNN model</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    feat1, feat2, feat3 = st.columns(3)
    with feat1:
        st.markdown(
            """
            <div class="about-tile zoom-card">
                <div class="icon">⚡</div>
                <h3>Instant Results</h3>
                <p>Real-time NAFLD risk prediction with probability scores and
                color-coded risk levels (Low / Moderate / High).</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with feat2:
        st.markdown(
            """
            <div class="about-tile zoom-card">
                <div class="icon">💡</div>
                <h3>Smart Recommendations</h3>
                <p>Personalized lifestyle solutions, diet tips, and exercise guidance
                based on prediction results and patient data.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with feat3:
        st.markdown(
            """
            <div class="about-tile zoom-card">
                <div class="icon">🔐</div>
                <h3>Secure Access</h3>
                <p>User login with local session storage so returning users don't need
                to re-enter credentials every time.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if clin_metrics and img_metrics:
        st.markdown('<p class="section-title">Model Performance Summary</p>', unsafe_allow_html=True)
        perf = pd.DataFrame({
            "Model": ["Clinical (Best)", "Ultrasound (ResNet18)"],
            "Accuracy": [clin_metrics["accuracy"], img_metrics["accuracy"]],
            "F1 Score": [clin_metrics["f1"], img_metrics["f1"]],
            "ROC-AUC": [clin_metrics["roc_auc"], img_metrics["roc_auc"]],
        })
        chart = (
            alt.Chart(perf.melt("Model", var_name="Metric", value_name="Score"))
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
            .encode(
                x=alt.X("Metric:N", title=None),
                y=alt.Y("Score:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Model:N", scale=alt.Scale(range=["#0d9488", "#3b82f6"])),
                xOffset="Model:N",
            )
            .properties(height=340)
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown(
        """
        <div class="about-disclaimer zoom-card" style="margin-top:1.5rem;">
            <strong>⚠️ Medical Disclaimer</strong><br><br>
            This platform is a <strong>research and educational screening tool only</strong>.
            It is NOT a medical device and must NOT be used as a substitute for professional
            medical diagnosis, treatment, or clinical decision-making. AI predictions may
            contain errors. Always consult a qualified doctor, hepatologist, or healthcare
            professional for confirmed diagnosis, further testing (FibroScan, biopsy, MRI),
            and personalized treatment plans.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)
