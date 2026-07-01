"""
Script untuk mengunduh citra uji standar (Lena, Cameraman, Baboon)
untuk penelitian perbandingan filter Median dan Gaussian.

Jika gagal mengunduh, akan membuat citra sintetis 512x512 sebagai fallback.
"""

import os
import urllib.request
import numpy as np

# Direktori penyimpanan citra uji
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")

# Sumber citra uji dari repositori publik
IMAGE_SOURCES = {
    "lena": [
        "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png",
        "https://raw.githubusercontent.com/mohammadimtiazz/standard-test-images-for-Image-Processing/master/standard_test_images/lena_gray_512.tif",
    ],
    "cameraman": [
        "https://raw.githubusercontent.com/mohammadimtiazz/standard-test-images-for-Image-Processing/master/standard_test_images/cameraman.tif",
    ],
    "baboon": [
        "https://raw.githubusercontent.com/mohammadimtiazz/standard-test-images-for-Image-Processing/master/standard_test_images/baboon.png",
    ],
}


def download_image(name: str, urls: list, output_dir: str) -> bool:
    """Mengunduh citra dari URL. Mencoba setiap URL sampai berhasil."""
    for url in urls:
        ext = os.path.splitext(url)[1].split("?")[0]  # ambil ekstensi file
        if not ext:
            ext = ".png"
        filepath = os.path.join(output_dir, f"{name}{ext}")
        try:
            print(f"  Mengunduh {name} dari {url} ...")
            urllib.request.urlretrieve(url, filepath)
            print(f"  ✓ Berhasil disimpan: {filepath}")
            return True
        except Exception as e:
            print(f"  ✗ Gagal: {e}")
    return False


def generate_synthetic_image(name: str, output_dir: str, size: int = 512) -> None:
    """
    Membuat citra sintetis grayscale 512x512 sebagai fallback
    jika download gagal. Menggunakan pola geometris yang berbeda
    untuk setiap citra agar dapat dibedakan saat pengujian.
    """
    try:
        import cv2
    except ImportError:
        print("  ✗ OpenCV tidak tersedia untuk menyimpan citra sintetis.")
        return

    img = np.zeros((size, size), dtype=np.uint8)
    y, x = np.mgrid[0:size, 0:size]

    if name == "lena":
        # Pola radial gradient dengan cincin
        cx, cy = size // 2, size // 2
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        img = np.clip(128 + 80 * np.sin(r / 20), 0, 255).astype(np.uint8)
    elif name == "cameraman":
        # Pola kotak-kotak (checkerboard) dengan gradasi
        block = 64
        checker = ((x // block) + (y // block)) % 2
        gradient = (x * 255 / size).astype(np.uint8)
        img = np.where(checker, gradient, 255 - gradient).astype(np.uint8)
    elif name == "baboon":
        # Pola tekstur acak (menyerupai tekstur detail tinggi)
        np.random.seed(42)
        base = np.random.randint(50, 200, (size, size), dtype=np.uint8)
        # Tambahkan variasi frekuensi rendah
        low_freq = np.clip(
            128 + 60 * np.sin(x / 40.0) * np.cos(y / 30.0), 0, 255
        ).astype(np.uint8)
        img = np.clip((base.astype(int) + low_freq.astype(int)) // 2, 0, 255).astype(
            np.uint8
        )
    else:
        # Pola default: gradasi diagonal
        img = ((x + y) * 255 // (2 * size)).astype(np.uint8)

    filepath = os.path.join(output_dir, f"{name}.png")
    cv2.imwrite(filepath, img)
    print(f"  ✓ Citra sintetis dibuat: {filepath}")


def main():
    """Fungsi utama untuk menyiapkan citra uji."""
    print("=" * 60)
    print("  PERSIAPAN CITRA UJI")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nFolder output: {OUTPUT_DIR}\n")

    for name, urls in IMAGE_SOURCES.items():
        print(f"[{name.upper()}]")
        success = download_image(name, urls, OUTPUT_DIR)
        if not success:
            print(f"  → Membuat citra sintetis sebagai pengganti...")
            generate_synthetic_image(name, OUTPUT_DIR)
        print()

    # Verifikasi hasil
    print("=" * 60)
    print("  VERIFIKASI")
    print("=" * 60)
    files = os.listdir(OUTPUT_DIR)
    if files:
        for f in sorted(files):
            fpath = os.path.join(OUTPUT_DIR, f)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {f:30s}  {size_kb:8.1f} KB")
    else:
        print("  ✗ Tidak ada file citra yang tersedia!")
    print()


if __name__ == "__main__":
    main()
