"""
==========================================================================
  Streamlit App: Perbandingan Filter Median dan Gaussian
  untuk Reduksi Noise Salt-and-Pepper pada Citra Digital
==========================================================================
  Interface interaktif untuk demo dan presentasi.
  Mengintegrasikan Google Gemini untuk penjelasan otomatis.
==========================================================================
"""

import streamlit as st
import numpy as np
import cv2
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import io
import os
import sys

# Import fungsi dari program utama
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filter_comparison import (
    add_salt_pepper_noise,
    apply_median_filter,
    apply_gaussian_filter,
    calculate_mse,
    calculate_psnr,
    KERNEL_SIZES,
    GAUSSIAN_SIGMA,
    MAX_PIXEL_VALUE,
    NOISE_DENSITIES,
)
import importlib
import gemini_helper
importlib.reload(gemini_helper)
from gemini_helper import generate_explanation, ask_followup

# ── Konfigurasi Halaman ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Perbandingan Filter Citra Digital",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — Darkroom Film Lab Theme ────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif&family=IBM+Plex+Mono:wght@400;500;600;700&family=Outfit:wght@300;400;500;600;700&display=swap');

    :root {
        --bg-deep: #080807;
        --bg-panel: #111110;
        --bg-card: #1A1918;
        --border: #2A2825;
        --amber: #E89B3E;
        --amber-dim: #B87A2E;
        --amber-glow: rgba(232, 151, 62, 0.12);
        --gold: #D4A843;
        --median-blue: #5BA3CF;
        --gaussian-rose: #CF5B7C;
        --text: #E8E2D6;
        --text-muted: #8A8275;
        --film-perf: #2D2B27;
    }

    html, body, [class*="css"], .stMarkdown {
        font-family: 'Outfit', sans-serif !important;
        color: var(--text);
    }

    code, pre, [class*="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace !important;
    }

    .stApp {
        background: var(--bg-deep);
    }

    /* ── Film Grain Overlay ── */
    .stApp::after {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
        pointer-events: none;
        z-index: 9999;
        opacity: 0.4;
    }

    /* ── Header — Editorial Serif ── */
    .main-header {
        padding: 2.5rem 0 2rem 0;
        margin-bottom: 2.5rem;
        border-bottom: 1px solid var(--border);
        text-align: left;
        position: relative;
    }

    .main-header::after {
        content: '';
        position: absolute;
        bottom: -1px;
        left: 0;
        width: 120px;
        height: 3px;
        background: var(--amber);
    }

    .main-header h1 {
        font-family: 'Instrument Serif', serif !important;
        color: var(--text);
        font-size: 2.8rem;
        margin: 0;
        font-weight: 400;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }

    .main-header .subtitle {
        color: var(--text-muted);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        margin-top: 0.8rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    /* ── Metric Cards — Minimal Warm ── */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-top: 2px solid var(--text-muted);
        padding: 1.4rem 1.2rem;
        text-align: center;
        transition: border-color 0.4s ease, box-shadow 0.4s ease;
    }

    .metric-card:hover {
        border-top-color: var(--amber);
    }

    .metric-card.best {
        border-top-color: var(--gold) !important;
        box-shadow: 0 -4px 20px var(--amber-glow);
        position: relative;
        overflow: visible;
    }

    .metric-card.best::after {
        content: 'TERBAIK';
        position: absolute;
        top: -10px;
        right: 8px;
        background: var(--amber);
        color: var(--bg);
        padding: 2px 8px;
        font-size: 0.55rem;
        font-weight: 700;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.05em;
        border-radius: 3px;
    }

    .metric-label {
        color: var(--text-muted);
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        font-family: 'IBM Plex Mono', monospace;
        margin-bottom: 0.6rem;
    }

    .metric-value {
        color: var(--text);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.6rem;
        font-weight: 600;
        letter-spacing: -0.02em;
    }

    .metric-value.highlight {
        color: var(--gold);
    }

    /* ── Film Negative Frame ── */
    .film-frame {
        background: #050504;
        border: 8px solid #1A1917;
        padding: 6px;
        position: relative;
        box-shadow: 0 4px 20px rgba(0,0,0,0.6);
    }

    .film-frame::before, .film-frame::after {
        content: '◻ ◻ ◻ ◻ ◻';
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
        font-size: 5px;
        letter-spacing: 8px;
        color: var(--film-perf);
        opacity: 0.6;
    }

    .film-frame::before { top: -2px; }
    .film-frame::after  { bottom: -2px; }

    /* ── Recommendation Box ── */
    .recommendation-box {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-left: 3px solid var(--gold);
        padding: 1.8rem 2rem;
        margin: 1.5rem 0;
    }

    .recommendation-box h4 {
        font-family: 'Instrument Serif', serif !important;
        font-weight: 400 !important;
        font-size: 1.3rem !important;
    }

    /* ── Step Badge — Film Counter ── */
    .step-badge {
        display: inline-block;
        background: transparent;
        color: var(--amber);
        border: 1px solid var(--amber-dim);
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        font-size: 0.7rem;
        padding: 0.25rem 0.7rem;
        letter-spacing: 0.15em;
        margin-bottom: 0.6rem;
    }

    /* ── AI Darkroom Log ── */
    .ai-terminal {
        background: var(--bg-deep) !important;
        border: 1px solid var(--border) !important;
        padding: 1.8rem !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.85rem !important;
        color: var(--text) !important;
        line-height: 1.7 !important;
        position: relative;
        border-left: 3px solid var(--amber) !important;
    }

    .ai-terminal::before {
        content: 'ANALYSIS LOG';
        position: absolute;
        top: -10px;
        left: 16px;
        background: var(--bg-deep);
        padding: 0 8px;
        font-size: 0.6rem;
        color: var(--amber);
        letter-spacing: 0.2em;
    }

    /* ── Sidebar — Darkroom Controls ── */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-deep) !important;
        border-right: 1px solid var(--border) !important;
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        font-family: 'Instrument Serif', serif !important;
        font-weight: 400 !important;
        color: var(--text) !important;
    }

    /* ── Primary Button — Amber Exposure ── */
    button[kind="primary"] {
        background-color: var(--amber) !important;
        color: var(--bg-deep) !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.05em !important;
        border: none !important;
        border-radius: 0 !important;
        transition: all 0.3s ease !important;
    }

    button[kind="primary"]:hover {
        background-color: var(--gold) !important;
        box-shadow: 0 0 20px var(--amber-glow) !important;
    }

    /* ── Download Buttons ── */
    button[data-testid="stDownloadButton"] > div {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.8rem !important;
    }

    /* ── Hide Streamlit Footer ── */
    footer { visibility: hidden; }

    /* ── Image Captions ── */
    .stImage > div > div > p {
        text-align: center;
        font-size: 0.78rem;
        color: var(--text-muted);
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.03em;
        margin-top: 6px;
    }

    /* ── Section Divider — Amber Thread ── */
    .section-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, var(--amber) 0%, var(--border) 30%, transparent 100%);
        margin: 3rem 0;
    }

    /* ── Headings Override ── */
    .stMarkdown h3 {
        font-family: 'Instrument Serif', serif !important;
        font-weight: 400 !important;
        font-size: 1.6rem !important;
        color: var(--text) !important;
    }

    .stMarkdown h4 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 500 !important;
        font-size: 1.1rem !important;
        color: var(--text) !important;
    }

    /* ── Landing Cards ── */
    .landing-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        padding: 2rem 1.5rem;
        text-align: center;
        transition: border-color 0.3s ease;
    }

    .landing-card:hover {
        border-color: var(--amber-dim);
    }

    .landing-card .lc-num {
        font-family: 'Instrument Serif', serif;
        font-size: 2.5rem;
        color: var(--amber);
        line-height: 1;
        margin-bottom: 0.8rem;
    }

    .landing-card .lc-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: var(--text);
        margin-bottom: 0.6rem;
    }

    .landing-card .lc-desc {
        font-size: 0.85rem;
        color: var(--text-muted);
        line-height: 1.5;
    }

    /* ══════════════════════════════════════════════
       ANIMATION SYSTEM — Staggered Cinematic Load
       ══════════════════════════════════════════════ */

    /* ── Core Keyframes ── */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-40px) skewX(2deg); }
        to   { opacity: 1; transform: translateX(0) skewX(0); }
    }

    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(40px) skewX(-2deg); }
        to   { opacity: 1; transform: translateX(0) skewX(0); }
    }

    @keyframes scaleReveal {
        from { opacity: 0; transform: scale(0.88); filter: blur(4px); }
        60%  { opacity: 1; filter: blur(0); }
        to   { opacity: 1; transform: scale(1); filter: blur(0); }
    }

    @keyframes filmReveal {
        from { opacity: 0; transform: scaleY(0.6); transform-origin: top; filter: brightness(2) contrast(0.5); }
        50%  { filter: brightness(1.2) contrast(0.8); }
        to   { opacity: 1; transform: scaleY(1); filter: brightness(1) contrast(1); }
    }

    @keyframes glowPulse {
        0%, 100% { box-shadow: 0 -4px 20px var(--amber-glow); }
        50% { box-shadow: 0 -4px 40px rgba(232, 155, 62, 0.25), 0 0 60px rgba(232, 155, 62, 0.08); }
    }

    @keyframes shimmerLine {
        from { background-position: -200% 0; }
        to   { background-position: 200% 0; }
    }

    @keyframes typeReveal {
        from { width: 0; opacity: 0; }
        20%  { opacity: 1; }
        to   { width: 100%; opacity: 1; }
    }

    @keyframes counterUp {
        from { opacity: 0; transform: translateY(8px) scale(0.9); filter: blur(2px); }
        to   { opacity: 1; transform: translateY(0) scale(1); filter: blur(0); }
    }

    @keyframes borderGlow {
        0%, 100% { border-color: var(--border); }
        50% { border-color: var(--amber-dim); }
    }

    @keyframes dotPulse {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 0.8; }
    }

    /* ── Global Entrance — Staggered Cascade ── */
    .stMarkdown {
        animation: fadeUp 0.6s cubic-bezier(0.22, 1, 0.36, 1) both;
    }

    .stImage {
        animation: filmReveal 0.8s cubic-bezier(0.22, 1, 0.36, 1) both;
        animation-delay: 0.15s;
    }

    .stDataFrame {
        animation: fadeUp 0.5s cubic-bezier(0.22, 1, 0.36, 1) both;
        animation-delay: 0.3s;
    }

    /* ── Header Cinematic Entrance ── */
    .main-header {
        animation: fadeUp 0.9s cubic-bezier(0.22, 1, 0.36, 1) both;
    }

    .main-header h1 {
        animation: slideInLeft 1s cubic-bezier(0.22, 1, 0.36, 1) both;
        animation-delay: 0.2s;
    }

    .main-header .subtitle {
        animation: slideInLeft 0.8s cubic-bezier(0.22, 1, 0.36, 1) both;
        animation-delay: 0.5s;
    }

    .main-header::after {
        animation: shimmerLine 3s ease-in-out infinite;
        background: linear-gradient(90deg, var(--amber), var(--gold), var(--amber));
        background-size: 200% 100%;
    }

    /* ── Step Badge — Reveal with Glow ── */
    .step-badge {
        animation: slideInLeft 0.5s cubic-bezier(0.22, 1, 0.36, 1) both;
        position: relative;
        overflow: hidden;
    }

    .step-badge::after {
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(232, 155, 62, 0.15), transparent);
        animation: shimmerLine 2.5s ease-in-out infinite;
        animation-delay: 1s;
    }

    /* ── Metric Cards — Staggered Scale Reveal ── */
    .metric-card {
        animation: scaleReveal 0.6s cubic-bezier(0.22, 1, 0.36, 1) both;
        transition: all 0.4s cubic-bezier(0.22, 1, 0.36, 1);
    }

    .metric-card:hover {
        border-top-color: var(--amber);
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
    }

    .metric-card.best {
        border-top-color: var(--gold) !important;
        animation: scaleReveal 0.6s cubic-bezier(0.22, 1, 0.36, 1) both,
                   glowPulse 3s ease-in-out infinite;
        animation-delay: 0.2s, 0.8s;
    }

    .metric-value {
        animation: counterUp 0.7s cubic-bezier(0.22, 1, 0.36, 1) both;
        animation-delay: 0.4s;
    }

    /* ── Film Frames — Cinematic Exposure ── */
    .film-frame {
        animation: filmReveal 0.7s cubic-bezier(0.22, 1, 0.36, 1) both;
        transition: all 0.5s cubic-bezier(0.22, 1, 0.36, 1);
    }

    .film-frame:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 40px rgba(0, 0, 0, 0.7), 0 0 1px var(--amber-dim);
        border-color: var(--amber-dim);
    }

    .film-frame::before, .film-frame::after {
        animation: dotPulse 4s ease-in-out infinite;
    }

    .film-frame::after {
        animation-delay: 2s;
    }

    /* ── Landing Cards — Dramatic Stagger ── */
    .landing-card {
        animation: scaleReveal 0.7s cubic-bezier(0.22, 1, 0.36, 1) both;
        transition: all 0.4s cubic-bezier(0.22, 1, 0.36, 1);
    }

    .landing-card:hover {
        border-color: var(--amber);
        transform: translateY(-6px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
    }

    .landing-card .lc-num {
        animation: counterUp 0.8s cubic-bezier(0.22, 1, 0.36, 1) both;
        animation-delay: 0.3s;
        transition: all 0.3s ease;
    }

    .landing-card:hover .lc-num {
        transform: scale(1.15);
        text-shadow: 0 0 20px var(--amber-glow);
    }

    /* ── Section Divider — Animated Draw ── */
    .section-divider {
        animation: shimmerLine 4s ease-in-out infinite;
        background: linear-gradient(90deg, var(--amber) 0%, var(--border) 30%, transparent 100%);
        background-size: 200% 100%;
    }

    /* ── Recommendation Box — Slide + Border Glow ── */
    .recommendation-box {
        animation: slideInLeft 0.7s cubic-bezier(0.22, 1, 0.36, 1) both;
        transition: all 0.4s ease;
    }

    .recommendation-box:hover {
        border-left-color: var(--amber);
        box-shadow: -4px 0 20px var(--amber-glow);
    }

    /* ── AI Terminal — Type Effect ── */
    .ai-terminal {
        animation: fadeUp 0.6s cubic-bezier(0.22, 1, 0.36, 1) both;
        transition: all 0.3s ease;
    }

    .ai-terminal::before {
        animation: dotPulse 2s ease-in-out infinite;
    }

    /* ── Plotly Chart Container — Smooth Entry ── */
    .stPlotlyChart {
        animation: scaleReveal 0.8s cubic-bezier(0.22, 1, 0.36, 1) both;
        animation-delay: 0.2s;
    }

    /* ── Stagger Delays via nth-child ── */
    .stHorizontalBlock > div:nth-child(1) .metric-card,
    .stHorizontalBlock > div:nth-child(1) .landing-card,
    .stHorizontalBlock > div:nth-child(1) .film-frame {
        animation-delay: 0.05s;
    }
    .stHorizontalBlock > div:nth-child(2) .metric-card,
    .stHorizontalBlock > div:nth-child(2) .landing-card,
    .stHorizontalBlock > div:nth-child(2) .film-frame {
        animation-delay: 0.15s;
    }
    .stHorizontalBlock > div:nth-child(3) .metric-card,
    .stHorizontalBlock > div:nth-child(3) .landing-card,
    .stHorizontalBlock > div:nth-child(3) .film-frame {
        animation-delay: 0.25s;
    }
    .stHorizontalBlock > div:nth-child(4) .metric-card,
    .stHorizontalBlock > div:nth-child(4) .landing-card,
    .stHorizontalBlock > div:nth-child(4) .film-frame {
        animation-delay: 0.35s;
    }

    /* ── Scrollbar — Themed ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-deep); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--amber-dim); }
</style>
""", unsafe_allow_html=True)

IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")


# ═══════════════════════════════════════════════════════════════════════
#  FUNGSI UTILITAS
# ═══════════════════════════════════════════════════════════════════════

def load_preset_image(name):
    """Muat gambar preset dari folder images/."""
    for ext in ["png", "tif", "tiff", "jpg", "jpeg", "bmp"]:
        path = os.path.join(IMAGE_DIR, f"{name}.{ext}")
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                if img.shape != (512, 512):
                    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
                return img
    return None


def load_preset_image_color(name):
    """Muat gambar preset dalam format warna (BGR → RGB) untuk ditampilkan."""
    for ext in ["png", "tif", "tiff", "jpg", "jpeg", "bmp"]:
        path = os.path.join(IMAGE_DIR, f"{name}.{ext}")
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if img.shape[:2] != (512, 512):
                    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
                return img
    return None


def uploaded_to_grayscale(uploaded_file):
    """Konversi file upload ke grayscale numpy array 512x512."""
    pil_img = Image.open(uploaded_file)
    pil_gray = pil_img.convert("L")
    pil_resized = pil_gray.resize((512, 512), Image.LANCZOS)
    return np.array(pil_resized)


def process_image(original, noise_density):
    """Proses citra: tambah noise, filter, hitung metrik."""
    np.random.seed(42)
    noisy = add_salt_pepper_noise(original, noise_density)

    noisy_mse = calculate_mse(original, noisy)
    noisy_psnr = calculate_psnr(noisy_mse)

    filters = []
    for filter_type in ["Median", "Gaussian"]:
        for ksize in KERNEL_SIZES:
            if filter_type == "Median":
                filtered = apply_median_filter(noisy, ksize)
            else:
                filtered = apply_gaussian_filter(noisy, ksize, GAUSSIAN_SIGMA)

            mse = calculate_mse(original, filtered)
            psnr = calculate_psnr(mse)

            filters.append({
                "filter_type": filter_type,
                "kernel_size": ksize,
                "label": f"{filter_type} {ksize}×{ksize}",
                "filtered_image": filtered,
                "mse": mse,
                "psnr": psnr,
            })

    best = max(filters, key=lambda r: r["psnr"])
    worst = min(filters, key=lambda r: r["psnr"])

    return {
        "original": original,
        "noisy": noisy,
        "noisy_mse": noisy_mse,
        "noisy_psnr": noisy_psnr,
        "noise_density": noise_density,
        "filters": filters,
        "best": best,
        "worst": worst,
    }


def create_comparison_chart(results):
    """Buat bar chart interaktif perbandingan PSNR dengan animasi."""
    labels = [f["label"] for f in results["filters"]]
    psnr_vals = [f["psnr"] for f in results["filters"]]
    best_label = results["best"]["label"]

    colors = []
    for label in labels:
        if label == best_label:
            colors.append("#D4A843")
        elif "Median" in label:
            colors.append("#5BA3CF")
        else:
            colors.append("#CF5B7C")

    fig = go.Figure(go.Bar(
        y=labels, x=psnr_vals,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(width=1, color='rgba(255,255,255,0.1)'),
        ),
        text=[f"{v:.2f} dB" for v in psnr_vals],
        textposition='outside',
        textfont=dict(family="IBM Plex Mono, monospace", size=12, color="#E8E2D6"),
        hovertemplate="<b>%{y}</b><br>PSNR: %{x:.2f} dB<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text="<b>Perbandingan PSNR</b><br><span style='font-size:11px;color:#8A8275;'>Semakin tinggi = semakin baik</span>",
            font=dict(family="IBM Plex Mono, monospace", size=14, color="#E8E2D6"),
        ),
        xaxis=dict(
            title="PSNR (dB)", color="#8A8275",
            gridcolor="rgba(42, 40, 37, 0.6)", gridwidth=1,
            range=[0, max(psnr_vals) * 1.25],
            zeroline=False,
        ),
        yaxis=dict(color="#E8E2D6", tickfont=dict(size=11)),
        plot_bgcolor="#111110",
        paper_bgcolor="#080807",
        font=dict(family="Outfit, sans-serif", color="#E8E2D6"),
        margin=dict(l=10, r=60, t=60, b=40),
        height=300,
        hoverlabel=dict(bgcolor="#1A1918", font_size=13, font_family="IBM Plex Mono, monospace", bordercolor="#E89B3E"),
    )
    return fig


def create_mse_chart(results):
    """Buat bar chart interaktif perbandingan MSE dengan animasi."""
    labels = [f["label"] for f in results["filters"]]
    mse_vals = [f["mse"] for f in results["filters"]]
    best_label = results["best"]["label"]

    colors = []
    for label in labels:
        if label == best_label:
            colors.append("#D4A843")
        elif "Median" in label:
            colors.append("#5BA3CF")
        else:
            colors.append("#CF5B7C")

    fig = go.Figure(go.Bar(
        y=labels, x=mse_vals,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(width=1, color='rgba(255,255,255,0.1)'),
        ),
        text=[f"{v:.2f}" for v in mse_vals],
        textposition='outside',
        textfont=dict(family="IBM Plex Mono, monospace", size=12, color="#E8E2D6"),
        hovertemplate="<b>%{y}</b><br>MSE: %{x:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text="<b>Perbandingan MSE</b><br><span style='font-size:11px;color:#8A8275;'>Semakin rendah = semakin baik</span>",
            font=dict(family="IBM Plex Mono, monospace", size=14, color="#E8E2D6"),
        ),
        xaxis=dict(
            title="MSE", color="#8A8275",
            gridcolor="rgba(42, 40, 37, 0.6)", gridwidth=1,
            range=[0, max(mse_vals) * 1.3],
            zeroline=False,
        ),
        yaxis=dict(color="#E8E2D6", tickfont=dict(size=11)),
        plot_bgcolor="#111110",
        paper_bgcolor="#080807",
        font=dict(family="Outfit, sans-serif", color="#E8E2D6"),
        margin=dict(l=10, r=60, t=60, b=40),
        height=300,
        hoverlabel=dict(bgcolor="#1A1918", font_size=13, font_family="IBM Plex Mono, monospace", bordercolor="#CF5B7C"),
    )
    return fig


def image_to_png_bytes(img_array):
    """Konversi numpy array ke PNG bytes untuk download."""
    pil_img = Image.fromarray(img_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════

def render_sidebar():
    """Render sidebar dan return konfigurasi."""
    with st.sidebar:
        st.markdown("## ⚙️ Pengaturan")

        # ── Sumber Gambar ──
        st.markdown("### 🖼️ Sumber Citra")
        source = st.radio(
            "Pilih sumber citra:",
            ["Citra Preset", "Upload Citra"],
            index=0,
            label_visibility="collapsed",
        )

        image = None
        image_color = None
        image_name = "uploaded"

        if source == "Citra Preset":
            presets = {
                "Lena": "lena",
                "Cameraman": "cameraman",
                "Baboon": "baboon",
            }
            selected = st.selectbox("Pilih citra uji:", list(presets.keys()))
            image = load_preset_image(presets[selected])
            image_color = load_preset_image_color(presets[selected])
            image_name = presets[selected]
            if image is None:
                st.error(f"Gagal memuat {selected}. Jalankan `download_test_images.py`.")
        else:
            uploaded = st.file_uploader(
                "Upload citra (JPG/PNG/BMP):",
                type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            )
            if uploaded:
                pil_img = Image.open(uploaded)
                pil_color = pil_img.convert("RGB").resize((512, 512), Image.LANCZOS)
                image_color = np.array(pil_color)
                uploaded.seek(0)
                image = uploaded_to_grayscale(uploaded)
                image_name = os.path.splitext(uploaded.name)[0]

        # ── Parameter Noise ──
        st.markdown("---")
        st.markdown("### 🎛️ Parameter Noise")
        noise_density = st.slider(
            "Densitas Noise (%)",
            min_value=5, max_value=50, value=20, step=5,
            help="Persentase piksel yang terkena noise Salt-and-Pepper",
        ) / 100.0

        st.caption(f"Noise density: **{noise_density:.0%}** "
                   f"({int(512*512*noise_density):,} piksel terdampak)")

        # Ambil API Key dari secrets (tanpa UI di sidebar)
        api_key = ""
        try:
            if "GEMINI_API_KEY" in st.secrets:
                api_key = st.secrets["GEMINI_API_KEY"]
        except Exception as e:
            print(f"DEBUG: Exception reading st.secrets: {e}")
            pass
        print(f"DEBUG: GEMINI_API_KEY read status: {'Loaded' if api_key else 'Empty'}, Length: {len(api_key)}")



        # ── Tombol Proses ──
        st.markdown("---")
        process_btn = st.button(
            "🔬 Mulai Proses",
            use_container_width=True,
            type="primary",
            disabled=(image is None),
        )

        # ── Info ──
        st.markdown("---")
        st.markdown(
            "<div style='text-align:center;color:#8A8275;font-size:0.7rem;"
            "font-family:IBM Plex Mono,monospace;letter-spacing:0.1em;'>"
            "PROYEK PENGOLAHAN CITRA DIGITAL<br>"
            "<span style='color:#E89B3E;'>MEDIAN vs GAUSSIAN</span><br>"
            "&copy; 2026"
            "</div>",
            unsafe_allow_html=True,
        )

    return image, image_color, image_name, noise_density, api_key, process_btn


# ═══════════════════════════════════════════════════════════════════════
#  TAMPILAN HASIL
# ═══════════════════════════════════════════════════════════════════════

def display_step1_images(results):
    """Tampilkan Step 1: Citra berwarna → Grayscale → Citra bernoise."""
    st.markdown('<div class="step-badge">STEP 1</div>', unsafe_allow_html=True)
    st.markdown("### Citra Asli vs Citra Ber-noise")

    original_color = results.get("original_color")
    if original_color is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="film-frame">', unsafe_allow_html=True)
            st.image(original_color, caption="Citra Asli (Berwarna)",
                     use_container_width=True, clamp=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="film-frame">', unsafe_allow_html=True)
            st.image(results["original"], caption="Grayscale 512×512",
                     use_container_width=True, clamp=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="film-frame">', unsafe_allow_html=True)
            cap = (f"Noise Salt-and-Pepper {results['noise_density']:.0%}  |  "
                   f"PSNR: {results['noisy_psnr']:.2f} dB")
            st.image(results["noisy"], caption=cap,
                     use_container_width=True, clamp=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="film-frame">', unsafe_allow_html=True)
            st.image(results["original"], caption="Citra Asli (Grayscale 512×512)",
                     use_container_width=True, clamp=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="film-frame">', unsafe_allow_html=True)
            cap = (f"Noise Salt-and-Pepper {results['noise_density']:.0%}  |  "
                   f"PSNR: {results['noisy_psnr']:.2f} dB")
            st.image(results["noisy"], caption=cap,
                     use_container_width=True, clamp=True)
            st.markdown('</div>', unsafe_allow_html=True)


def display_step2_filters(results):
    """Tampilkan Step 2: Hasil semua filter."""
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<div class="step-badge">STEP 2</div>', unsafe_allow_html=True)
    st.markdown("### Hasil Filtering")

    cols = st.columns(4)
    for i, f in enumerate(results["filters"]):
        is_best = (f["label"] == results["best"]["label"])
        with cols[i]:
            if is_best:
                st.markdown(
                    '<div class="film-frame" style="position:relative;">'
                    '<span style="position:absolute;top:8px;right:8px;z-index:10;'
                    'background:#E89B3E;color:#0A0A09;padding:2px 10px;'
                    'font-size:0.7rem;font-weight:700;font-family:IBM Plex Mono,monospace;'
                    'letter-spacing:0.05em;border-radius:3px;">TERBAIK</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div class="film-frame">', unsafe_allow_html=True)
            caption = (f"{f['label']}\n"
                       f"PSNR: {f['psnr']:.2f} dB | MSE: {f['mse']:.2f}")
            st.image(f["filtered_image"], caption=caption,
                     use_container_width=True, clamp=True)
            st.markdown('</div>', unsafe_allow_html=True)


def display_step3_metrics(results):
    """Tampilkan Step 3: Metrik perbandingan."""
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<div class="step-badge">STEP 3</div>', unsafe_allow_html=True)
    st.markdown("### Analisis Metrik")

    # Kartu metrik ringkasan dengan gaya kustom
    best = results["best"]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Filter Terbaik</div>
            <div class="metric-value" style="color: #E89B3E;">{best['label']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card best">
            <div class="metric-label">PSNR Terbaik</div>
            <div class="metric-value highlight">{best['psnr']:.2f} dB</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">MSE Terendah</div>
            <div class="metric-value" style="color: #CF5B7C;">{best['mse']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        improvement = best["psnr"] - results["noisy_psnr"]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Peningkatan PSNR</div>
            <div class="metric-value" style="color: #5BA3CF;">+{improvement:.2f} dB</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabel detail
    st.markdown("#### Tabel Perbandingan Lengkap")
    rows = []
    for f in results["filters"]:
        is_best = (f["label"] == best["label"])
        rows.append({
            "Filter": f["filter_type"],
            "Kernel": f"{f['kernel_size']}×{f['kernel_size']}",
            "MSE": round(f["mse"], 2),
            "PSNR (dB)": round(f["psnr"], 2),
            "Peringkat": "Terbaik" if is_best else "",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Grafik
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig_psnr = create_comparison_chart(results)
        st.plotly_chart(fig_psnr, use_container_width=True)
    with chart_col2:
        fig_mse = create_mse_chart(results)
        st.plotly_chart(fig_mse, use_container_width=True)


def display_step4_recommendation(results):
    """Tampilkan Step 4: Rekomendasi."""
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<div class="step-badge">STEP 4</div>', unsafe_allow_html=True)
    st.markdown("### Rekomendasi")

    best = results["best"]
    worst = results["worst"]
    density_pct = results["noise_density"] * 100

    st.markdown(f"""
<div class="recommendation-box">
    <h4 style="color:#D4A843; margin-top:0;">Rekomendasi Filter</h4>
    <p>Untuk noise <b>Salt-and-Pepper</b> pada densitas <b>{density_pct:.0f}%</b>,
    filter <b style="color:#D4A843;">{best['label']}</b> memberikan restorasi citra paling optimal:</p>
    <ul>
        <li><b>PSNR:</b> {best['psnr']:.2f} dB (vs {worst['label']}: {worst['psnr']:.2f} dB)</li>
        <li><b>MSE:</b> {best['mse']:.2f} (vs {worst['label']}: {worst['mse']:.2f})</li>
        <li><b>Selisih PSNR:</b> {best['psnr'] - worst['psnr']:.2f} dB lebih unggul dibandingkan konfigurasi terburuk.</li>
    </ul>
</div>
""", unsafe_allow_html=True)


def display_step5_ai(results, api_key):
    """Tampilkan Step 5: Penjelasan AI dengan Multi-turn Chat History."""
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<div class="step-badge">STEP 5</div>', unsafe_allow_html=True)
    st.markdown("### 🤖 Penjelasan AI")

    # Inisialisasi riwayat obrolan jika belum ada
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Generate atau tampilkan penjelasan awal
    generate_btn = st.button("✨ Generate Penjelasan", type="secondary")

    if generate_btn or st.session_state.get("explanation"):
        if generate_btn or "explanation" not in st.session_state:
            with st.spinner("Menghasilkan penjelasan..."):
                explanation, source = generate_explanation(results, api_key)
                st.session_state["explanation"] = explanation
                st.session_state["explanation_source"] = source
                # Reset chat history saat generate ulang penjelasan baru
                st.session_state["chat_history"] = []

        explanation = st.session_state["explanation"]
        source = st.session_state.get("explanation_source", "fallback")

        if source == "gemini":
            st.caption("🤖 Dihasilkan oleh Google Gemini AI")
        else:
            st.caption("📝 Penjelasan template otomatis (tanpa API Key)")

        st.markdown(f'<div class="ai-terminal">{explanation}</div>', unsafe_allow_html=True)

        # Tanya jawab lanjutan dengan Riwayat Percakapan
        st.markdown("---")
        st.markdown("#### 💬 Terminal Tanya Jawab Interaktif")

        # Tampilkan riwayat chat yang ada
        if st.session_state["chat_history"]:
            chat_content = ""
            for chat in st.session_state["chat_history"]:
                q_text = chat["question"]
                a_text = chat["answer"]
                chat_content += f'<span style="color: #E89B3E; font-weight: bold;">PERTANYAAN &rsaquo;</span> {q_text}<br><br>'
                chat_content += f'<span style="color: #5BA3CF; font-weight: bold;">JAWABAN &rsaquo;</span> {a_text}<br><hr style="border-color: #2A2825; margin: 15px 0;">'

            st.markdown(f'<div class="ai-terminal" style="max-height: 400px; overflow-y: auto;">{chat_content}</div>', unsafe_allow_html=True)

        # Form / Input untuk pertanyaan baru menggunakan st.chat_input agar responsif dan modern
        question = st.chat_input("Tanyakan detail eksperimen (misal: Mengapa ukuran kernel ganjil?)...")
        if question:
            answer, src = ask_followup(question, results, api_key)
            st.session_state["chat_history"].append({
                "question": question,
                "answer": answer,
                "source": src
            })
            st.rerun()


def display_step6_download(results, image_name):
    """Tampilkan Step 6: Download hasil."""
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<div class="step-badge">STEP 6</div>', unsafe_allow_html=True)
    st.markdown("### 📥 Unduh Hasil")

    col1, col2 = st.columns(2)
    with col1:
        best = results["best"]
        png_data = image_to_png_bytes(best["filtered_image"])
        st.download_button(
            f"🖼️ Download Hasil Terbaik ({best['label']})",
            data=png_data,
            file_name=f"filtered_{image_name}_{best['label'].replace(' ', '_')}.png",
            mime="image/png",
            use_container_width=True,
        )
    with col2:
        noisy_png = image_to_png_bytes(results["noisy"])
        st.download_button(
            "🖼️ Download Citra Ber-noise",
            data=noisy_png,
            file_name=f"noisy_{image_name}_{results['noise_density']:.0%}.png",
            mime="image/png",
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Perbandingan Filter Citra Digital</h1>
        <div class="subtitle">Median vs Gaussian &mdash; Reduksi Noise Salt-and-Pepper</div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    image, image_color, image_name, noise_density, api_key, process_btn = render_sidebar()

    # State init
    if "results" not in st.session_state:
        st.session_state["results"] = None

    # Proses saat tombol diklik
    if process_btn and image is not None:
        # Clear penjelasan lama
        st.session_state.pop("explanation", None)
        st.session_state.pop("explanation_source", None)

        with st.spinner("⏳ Memproses citra..."):
            results = process_image(image, noise_density)
            results["image_name"] = image_name
            results["original_color"] = image_color
            st.session_state["results"] = results
        st.success("Proses selesai!")

    # Tampilkan hasil jika ada
    results = st.session_state.get("results")

    if results is not None:
        display_step1_images(results)
        display_step2_filters(results)
        display_step3_metrics(results)
        display_step4_recommendation(results)
        display_step5_ai(results, api_key)
        display_step6_download(results, results.get("image_name", "image"))

    else:
        # Landing state
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="landing-card" style="animation-delay: 0.1s;">
                <div class="lc-num">1</div>
                <div class="lc-title">Pilih Citra</div>
                <div class="lc-desc">Gunakan citra preset (Lena, Cameraman, Baboon) atau upload citra sendiri.</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="landing-card" style="animation-delay: 0.25s;">
                <div class="lc-num">2</div>
                <div class="lc-title">Atur Noise</div>
                <div class="lc-desc">Pilih densitas noise Salt-and-Pepper yang ingin diuji (5% &ndash; 50%).</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="landing-card" style="animation-delay: 0.4s;">
                <div class="lc-num">3</div>
                <div class="lc-title">Analisis</div>
                <div class="lc-desc">Klik "Mulai Proses" untuk melihat perbandingan filter lengkap.</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.info(
            "Atur parameter di sidebar, lalu klik **Mulai Proses** untuk memulai analisis.",
            icon="💡",
        )


if __name__ == "__main__":
    main()
