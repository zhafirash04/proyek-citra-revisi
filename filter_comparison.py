"""
==========================================================================
  Perbandingan Filter Median dan Gaussian untuk Reduksi Noise
  Salt-and-Pepper pada Citra Digital Berdasarkan Metrik PSNR dan MSE
==========================================================================

  Program ini membandingkan performa Filter Median dan Filter Gaussian
  dalam mereduksi noise Salt-and-Pepper pada citra digital grayscale.

  Referensi penelitian:
  - R.C. Gonzalez & R.E. Woods, "Digital Image Processing", 4th Ed.
  - Format laporan: IEEE

  Library: OpenCV, NumPy, Matplotlib
  Penulis: [Nama Anda]
  Tanggal: April 2026
==========================================================================
"""

import os
import glob
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ==========================================================================
#  PARAMETER PENELITIAN
#  Ubah variabel di bawah ini untuk menyesuaikan eksperimen
# ==========================================================================

# Variasi noise density (persentase piksel yang terkena noise)
NOISE_DENSITIES = [0.10, 0.20, 0.30]  # 10%, 20%, 30%

# Ukuran kernel filter
KERNEL_SIZES = [3, 5]  # 3x3 dan 5x5

# Sigma untuk Gaussian filter
GAUSSIAN_SIGMA = 1.0  # Kembalikan ke 1.0 sesuai dengan eksperimen awal

# Direktori input dan output
IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Nama citra uji yang diharapkan (tanpa ekstensi)
EXPECTED_IMAGES = ["lena", "cameraman", "baboon"]

# Nilai maksimum piksel (untuk citra 8-bit)
MAX_PIXEL_VALUE = 255


# ==========================================================================
#  FUNGSI-FUNGSI UTAMA
# ==========================================================================


def add_salt_pepper_noise(image, density):
    """
    Menambahkan noise Salt-and-Pepper pada citra secara manual.

    Noise Salt-and-Pepper mengubah piksel secara acak menjadi:
    - Salt (putih): piksel = 255
    - Pepper (hitam): piksel = 0
    Masing-masing memiliki probabilitas density/2.

    Parameter:
        image   : numpy array citra grayscale (2D) atau berwarna (3D, H×W×C)
        density : float, proporsi piksel yang terkena noise (0.0 - 1.0)

    Return:
        noisy_image : numpy array citra berisi noise
    """
    assert 0 < density <= 1, "Noise density harus lebih dari 0 dan maksimal 1"
    noisy_image = image.copy()

    if image.ndim == 3:
        rows, cols, channels = image.shape
    else:
        rows, cols = image.shape
        channels = 0

    total_pixels = rows * cols
    num_salt = int(total_pixels * density / 2)
    num_pepper = int(total_pixels * density / 2)

    if channels > 0:
        for c in range(channels):
            flat_indices = np.random.choice(total_pixels, num_salt + num_pepper, replace=False)
            salt_indices = flat_indices[:num_salt]
            pepper_indices = flat_indices[num_salt:]
            noisy_image[salt_indices // cols, salt_indices % cols, c] = MAX_PIXEL_VALUE
            noisy_image[pepper_indices // cols, pepper_indices % cols, c] = 0
    else:
        flat_indices = np.random.choice(total_pixels, num_salt + num_pepper, replace=False)
        salt_indices = flat_indices[:num_salt]
        pepper_indices = flat_indices[num_salt:]
        noisy_image[salt_indices // cols, salt_indices % cols] = MAX_PIXEL_VALUE
        noisy_image[pepper_indices // cols, pepper_indices % cols] = 0

    return noisy_image


def apply_median_filter(image, ksize):
    """
    Menerapkan Median Filter pada citra.

    Median filter mengganti setiap piksel dengan nilai median
    dari tetangganya dalam jendela ksize × ksize.
    Filter ini efektif untuk noise impulsif (Salt-and-Pepper).

    Parameter:
        image : numpy array citra grayscale (2D) atau berwarna (3D, H×W×C)
        ksize : int, ukuran kernel (harus ganjil, misal 3 atau 5)

    Return:
        filtered : numpy array citra hasil filter
    """
    assert ksize % 2 == 1, "Ukuran kernel (ksize) harus bernilai ganjil"
    filtered = cv2.medianBlur(image, ksize)
    return filtered


def apply_gaussian_filter(image, ksize, sigma=GAUSSIAN_SIGMA):
    """
    Menerapkan Gaussian Filter pada citra.

    Gaussian filter melakukan konvolusi citra dengan kernel Gaussian
    yang menghaluskan citra berdasarkan distribusi normal.
    Filter ini lebih efektif untuk noise Gaussian daripada Salt-and-Pepper.

    Parameter:
        image : numpy array citra grayscale (2D) atau berwarna (3D, H×W×C)
        ksize : int, ukuran kernel (harus ganjil, misal 3 atau 5)
        sigma : float, standar deviasi Gaussian (default: 1.0)

    Return:
        filtered : numpy array citra hasil filter
    """
    assert ksize % 2 == 1, "Ukuran kernel (ksize) harus bernilai ganjil"
    filtered = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    return filtered


def calculate_mse(original, filtered):
    """
    Menghitung Mean Squared Error (MSE) antara citra asli dan citra hasil filter.

    Rumus: MSE = mean((original - filtered)^2)

    MSE mengukur rata-rata kuadrat selisih piksel antara dua citra.
    Semakin kecil MSE, semakin mirip citra hasil filter dengan citra asli.

    Parameter:
        original : numpy array citra asli (2D atau 3D)
        filtered : numpy array citra hasil filter (2D atau 3D)

    Return:
        mse : float, nilai Mean Squared Error
    """
    # Konversi ke float64 untuk menghindari overflow pada operasi kuadrat
    diff = original.astype(np.float64) - filtered.astype(np.float64)
    mse = np.mean(diff ** 2)
    return mse


def calculate_psnr(mse):
    """
    Menghitung Peak Signal-to-Noise Ratio (PSNR) dari nilai MSE.

    Rumus: PSNR = 10 * log10(MAX^2 / MSE)
    di mana MAX = 255 untuk citra 8-bit.

    PSNR mengukur kualitas rekonstruksi citra dalam satuan desibel (dB).
    Semakin tinggi PSNR, semakin baik kualitas citra hasil filter.
    Umumnya: >40 dB = sangat baik, 30-40 dB = baik, <30 dB = kurang baik.

    Parameter:
        mse : float, nilai Mean Squared Error

    Return:
        psnr : float, nilai PSNR dalam dB. Jika MSE=0, return inf.
    """
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10((MAX_PIXEL_VALUE ** 2) / mse)
    return psnr


def load_test_images(image_dir):
    """
    Memuat citra uji grayscale dari direktori.

    Mencari file citra berdasarkan nama yang diharapkan (lena, cameraman, baboon).
    Jika citra berwarna, akan dikonversi ke grayscale.
    Semua citra di-resize ke 512x512 untuk konsistensi.

    Parameter:
        image_dir : str, path ke direktori berisi citra uji

    Return:
        images : dict, {nama_citra: numpy_array_grayscale}
    """
    images = {}
    # Cari semua file gambar di direktori
    all_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"]:
        all_files.extend(glob.glob(os.path.join(image_dir, ext)))

    if not all_files:
        print(f"[ERROR] Tidak ada file citra di folder: {image_dir}")
        print("        Jalankan 'python download_test_images.py' terlebih dahulu.")
        return images

    for filepath in sorted(all_files):
        basename = os.path.splitext(os.path.basename(filepath))[0].lower()

        # Muat citra dalam mode grayscale
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARNING] Gagal memuat: {filepath}")
            continue

        # Resize ke 512x512 jika perlu
        if img.shape != (512, 512):
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
            print(f"  {basename}: di-resize ke 512x512")

        images[basename] = img
        print(f"  [OK] {basename}: {img.shape} loaded dari {filepath}")

    return images


def export_results_to_csv(all_results, output_dir):
    """
    Mengekspor seluruh hasil eksperimen ke dalam file CSV.
    Berguna untuk analisis data lebih lanjut atau memuat data secara dinamis
    pada skrip visualisasi grafik.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "results.csv")
    
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Tulis Header
        writer.writerow(["Image", "Noise Density", "Filter", "Kernel Size", "MSE", "PSNR"])
        
        # Tulis Data
        for image_name in sorted(all_results.keys()):
            for r in all_results[image_name]:
                writer.writerow([
                    image_name,
                    r['noise_density'],
                    r['filter_type'],
                    f"{r['kernel_size']}x{r['kernel_size']}",
                    r['mse'],
                    r['psnr']
                ])
                
    print(f"  [OK] Hasil diekspor ke CSV: {csv_path}")
    return csv_path


def generate_result_table(all_results):
    """
    Mencetak tabel hasil lengkap di terminal dalam format yang rapi.

    Tabel ini dirancang agar mudah disalin ke dokumen laporan IEEE.
    Format: | Citra | Noise % | Filter | Kernel | MSE | PSNR (dB) |

    Parameter:
        all_results : dict, {nama_citra: [list of result dicts]}
    """
    # Header tabel
    print()
    print("=" * 82)
    print("  TABEL HASIL PERBANDINGAN FILTER MEDIAN DAN GAUSSIAN")
    print("  Noise: Salt-and-Pepper | Metrik: MSE dan PSNR")
    print("=" * 82)
    print()

    # Format header kolom
    header = (
        f"{'Citra':<12} | {'Noise %':>7} | {'Filter':<10} | "
        f"{'Kernel':>6} | {'MSE':>10} | {'PSNR (dB)':>10}"
    )
    separator = "-" * len(header)

    print(header)
    print(separator)

    # Cetak data setiap kombinasi
    for image_name in sorted(all_results.keys()):
        results = all_results[image_name]
        first_row = True
        for r in results:
            name_col = image_name.capitalize() if first_row else ""
            noise_pct = f"{r['noise_density'] * 100:.0f}%"
            filter_label = f"{r['filter_type']}"
            kernel_label = f"{r['kernel_size']}x{r['kernel_size']}"
            mse_str = f"{r['mse']:.2f}"
            psnr_str = f"{r['psnr']:.2f}"

            print(
                f"{name_col:<12} | {noise_pct:>7} | {filter_label:<10} | "
                f"{kernel_label:>6} | {mse_str:>10} | {psnr_str:>10}"
            )
            first_row = False
        print(separator)

    print()

    # Tabel ringkasan: rata-rata PSNR per filter dan kernel
    print("=" * 60)
    print("  RINGKASAN: Rata-rata PSNR (dB) per Filter dan Kernel")
    print("=" * 60)
    print()

    summary_header = f"{'Filter':<10} | {'Kernel':>6} | {'Noise 10%':>10} | {'Noise 20%':>10} | {'Noise 30%':>10}"
    print(summary_header)
    print("-" * len(summary_header))

    for filter_type in ["Median", "Gaussian"]:
        for ksize in KERNEL_SIZES:
            row = f"{filter_type:<10} | {ksize}x{ksize:>3} | "
            for density in NOISE_DENSITIES:
                # Kumpulkan PSNR dari semua citra untuk kombinasi ini
                psnr_values = []
                for image_name in all_results:
                    for r in all_results[image_name]:
                        if (
                            r["filter_type"] == filter_type
                            and r["kernel_size"] == ksize
                            and r["noise_density"] == density
                        ):
                            psnr_values.append(r["psnr"])
                avg_psnr = np.mean(psnr_values) if psnr_values else 0
                row += f"{avg_psnr:>10.2f} | "
            print(row)

    print()


def visualize_comparison(image_name, original, results_by_noise, output_dir):
    """
    Membuat figure matplotlib berisi perbandingan visual hasil filter.

    Layout figure:
      Baris : tiap noise density (10%, 20%, 30%)
      Kolom : Asli | Ber-noise | Median 3x3 | Median 5x5 | Gaussian 3x3 | Gaussian 5x5

    Parameter:
        image_name      : str, nama citra untuk judul dan nama file
        original        : numpy array citra asli
        results_by_noise: dict, {noise_density: [list of result dicts]}
        output_dir      : str, direktori untuk menyimpan file PNG
    """
    num_rows = len(NOISE_DENSITIES)
    num_cols = 6  # Asli, Noisy, Median 3x3, Median 5x5, Gaussian 3x3, Gaussian 5x5

    fig, axes = plt.subplots(
        num_rows, num_cols,
        figsize=(14, 3.8 * num_rows)
    )

    fig.suptitle(
        f"Perbandingan Filter pada Citra {image_name.capitalize()}"
        f" | Noise: Salt-and-Pepper | sigma={GAUSSIAN_SIGMA}",
        fontsize=11,
        fontweight="bold"
    )

    # Label kolom
    col_labels = [
        "Asli",
        "Ber-noise (S&P)",
        "Median 3×3",
        "Median 5×5",
        "Gaussian 3×3",
        "Gaussian 5×5",
    ]

    for row_idx, density in enumerate(NOISE_DENSITIES):
        noise_results = results_by_noise.get(density, [])

        # Urutkan: Median 3x3, Median 5x5, Gaussian 3x3, Gaussian 5x5
        ordered_results = []
        for ftype in ["Median", "Gaussian"]:
            for ksize in KERNEL_SIZES:
                for r in noise_results:
                    if r["filter_type"] == ftype and r["kernel_size"] == ksize:
                        ordered_results.append(r)
                        break

        # Kolom 0: Citra asli
        ax = axes[row_idx, 0]
        ax.imshow(original, cmap="gray", vmin=0, vmax=255)
        ax.set_title(col_labels[0], fontsize=8, pad=6)
        ax.set_ylabel(f"Noise {density * 100:.0f}%", fontsize=9, fontweight="bold")
        ax.axis("off")

        # Kolom 1: Citra ber-noise
        if ordered_results:
            noisy = ordered_results[0]["noisy_image"]
            ax = axes[row_idx, 1]
            ax.imshow(noisy, cmap="gray", vmin=0, vmax=255)
            mse_noisy = calculate_mse(original, noisy)
            psnr_noisy = calculate_psnr(mse_noisy)
            ax.set_title(
                f"{col_labels[1]}\nPSNR: {psnr_noisy:.2f} dB",
                fontsize=8,
                pad=6
            )
            ax.axis("off")

        # Kolom 2-5: Hasil filter
        for col_idx, r in enumerate(ordered_results, start=2):
            ax = axes[row_idx, col_idx]
            ax.imshow(r["filtered_image"], cmap="gray", vmin=0, vmax=255)
            ax.set_title(
                f"{col_labels[col_idx]}\n"
                f"MSE: {r['mse']:.2f}\n"
                f"PSNR: {r['psnr']:.2f} dB",
                fontsize=8,
                pad=6,
            )
            ax.axis("off")

    # Simpan figure sebagai PNG
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"perbandingan_{image_name}.png")
    fig.subplots_adjust(hspace=0.35, wspace=0.08, top=0.93)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] Figure disimpan: {output_path}")

    return output_path


def main():
    """
    Fungsi utama program.

    Alur eksekusi:
    1. Muat citra uji dari folder images/
    2. Untuk setiap citra, lakukan semua kombinasi eksperimen:
       3 noise density × 2 filter × 2 kernel = 12 kombinasi per citra
    3. Cetak tabel hasil lengkap di terminal
    4. Buat dan simpan figure perbandingan visual untuk setiap citra
    """

    print()
    print("=" * 70)
    print("  PERBANDINGAN FILTER MEDIAN DAN GAUSSIAN")
    print("  Untuk Reduksi Noise Salt-and-Pepper pada Citra Digital")
    print("=" * 70)
    print()

    # Tetapkan seed sekali saja di main() agar noise di setiap citra berbeda
    # tetapi reproduksibel pada iterasi berikutnya
    np.random.seed(42)

    # ── 1. Muat citra uji ──────────────────────────────────────────────
    print("[1/4] Memuat citra uji...")
    images = load_test_images(IMAGE_DIR)

    if not images:
        print("\n[ERROR] Tidak ada citra yang berhasil dimuat.")
        print("        Jalankan 'python download_test_images.py' terlebih dahulu.")
        return

    print(f"\n     Total citra dimuat: {len(images)}")
    print()

    # ── 2. Proses semua kombinasi eksperimen ───────────────────────────
    print("[2/4] Memproses semua kombinasi eksperimen...")
    print(f"     Noise density : {[f'{d*100:.0f}%' for d in NOISE_DENSITIES]}")
    print(f"     Kernel sizes  : {KERNEL_SIZES}")
    print(f"     Filter types  : Median, Gaussian (sigma={GAUSSIAN_SIGMA})")
    print(f"     Total kombinasi per citra: "
          f"{len(NOISE_DENSITIES)} x 2 x {len(KERNEL_SIZES)} = "
          f"{len(NOISE_DENSITIES) * 2 * len(KERNEL_SIZES)}")
    print()

    # Dictionary untuk menyimpan semua hasil
    all_results = {}       # {image_name: [list of result dicts]}
    results_by_noise = {}  # {image_name: {density: [list of result dicts]}}

    for image_name, original in images.items():
        print(f"  Memproses: {image_name.capitalize()} ({original.shape})")
        all_results[image_name] = []
        results_by_noise[image_name] = {}

        for density in NOISE_DENSITIES:
            results_by_noise[image_name][density] = []

            # Buat satu citra noisy per density (agar perbandingan fair)
            noisy = add_salt_pepper_noise(original, density)

            for filter_type in ["Median", "Gaussian"]:
                for ksize in KERNEL_SIZES:
                    # Terapkan filter pada citra noisy yang sama
                    if filter_type == "Median":
                        filtered = apply_median_filter(noisy, ksize)
                    else:
                        filtered = apply_gaussian_filter(noisy, ksize, GAUSSIAN_SIGMA)

                    # Hitung metrik
                    mse = calculate_mse(original, filtered)
                    psnr = calculate_psnr(mse)

                    result = {
                        "noisy_image": noisy,
                        "filtered_image": filtered,
                        "mse": mse,
                        "psnr": psnr,
                        "noise_density": density,
                        "filter_type": filter_type,
                        "kernel_size": ksize,
                    }

                    all_results[image_name].append(result)
                    results_by_noise[image_name][density].append(result)

        # Hitung progress
        total_done = len(all_results[image_name])
        print(f"    -> {total_done} kombinasi selesai")

    print()

    # ── 3. Cetak tabel hasil & Ekspor CSV ──────────────────────────────
    print("[3/4] Menampilkan tabel hasil dan mengekspor CSV...")
    generate_result_table(all_results)
    export_results_to_csv(all_results, OUTPUT_DIR)
    print()

    # ── 4. Buat visualisasi ────────────────────────────────────────────
    print("[4/4] Membuat visualisasi perbandingan...")
    saved_figures = []
    for image_name, original in images.items():
        path = visualize_comparison(
            image_name, original, results_by_noise[image_name], OUTPUT_DIR
        )
        saved_figures.append(path)

    # ── Selesai ────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  SELESAI!")
    print("=" * 70)
    print(f"  Total citra diuji     : {len(images)}")
    print(f"  Total kombinasi       : {sum(len(v) for v in all_results.values())}")
    print(f"  Figure tersimpan di   : {OUTPUT_DIR}")
    for fig_path in saved_figures:
        print(f"    - {os.path.basename(fig_path)}")
    print()
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
