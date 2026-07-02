# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- **Install dependencies**: `pip install -r requirements.txt`
- **Run Streamlit app**: `streamlit run app.py` (Interactive web interface with filtering, metrics, AI explanation, and download)
- **Download/prepare test images**: `python download_test_images.py` (Attempts to download Lena, Cameraman, Baboon from public URLs; falls back to generating synthetic geometric patterns in `images/` via `generate_synthetic_image()`.)
- **Run main experiment**: `python filter_comparison.py` (Applies noise, runs filter combinations, calculates MSE/PSNR, prints summary table.)
- **Run all unit tests**: `python -m unittest test_metrics.py`
- **Run a specific unit test**: `python -m unittest test_metrics.TestMetrics.test_calculate_mse_identical_images`

## Code Architecture & Structure

This repository evaluates and compares Median and Gaussian filters for Salt-and-Pepper noise reduction on grayscale images. The project is structured as a research experiment pipeline with reproducible results.

### Data Pipeline Flow

1. Test images are loaded from `images/` and resized to 512×512 (`filter_comparison.py:load_test_images`).
2. Salt-and-Pepper noise is injected via manual NumPy array operations with `np.random.seed(42)` for reproducibility (`filter_comparison.py:add_salt_pepper_noise`; each density gets its own noisy image per source image for fair filter comparison).
3. Noisy images are filtered using OpenCV's `medianBlur` and `GaussianBlur` across 3 noise densities × 2 filter types × 2 kernel sizes = 12 combinations per image (`filter_comparison.py:main`).
4. MSE (`calculate_mse`) and PSNR (`calculate_psnr`) are computed via NumPy array operations (`filter_comparison.py`).
5. Results are displayed interactively in the Streamlit web app with Plotly charts.

### Key Global Parameters (`filter_comparison.py`)

- `NOISE_DENSITIES = [0.10, 0.20, 0.30]` — noise proportions
- `KERNEL_SIZES = [3, 5]` — kernel dimensions
- `GAUSSIAN_SIGMA = 1.0` — Gaussian standard deviation
- `EXPECTED_IMAGES = ["lena", "cameraman", "baboon"]` — expected test image names
- `MAX_PIXEL_VALUE = 255` — 8-bit pixel max

### File Roles

| File | Role |
|------|------|
| `app.py` | Streamlit web interface: upload/preset images, noise injection, filtering, metrics display, AI explanation (Gemini), download |
| `gemini_helper.py` | Google Gemini API integration with fallback explanations in Indonesian |
| `filter_comparison.py` | Core experiment: image loading, noise injection, filtering, metric computation |
| `test_metrics.py` | Unit tests for MSE, PSNR, noise injection, and filters using `unittest` |
| `download_test_images.py` | Downloads standard test images with synthetic fallback if download fails |
| `images/` | Input directory for test images |
