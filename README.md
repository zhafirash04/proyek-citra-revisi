# Perbandingan Filter Median dan Gaussian untuk Reduksi Noise Salt-and-Pepper pada Citra Digital

**Proyek Pengolahan Citra Digital** — Aplikasi web interaktif untuk mengevaluasi performa Filter Median (non-linier) dan Filter Gaussian (linier) dalam mereduksi noise Salt-and-Pepper pada citra grayscale, diukur dengan metrik **PSNR** dan **MSE**.

> **Live Demo:** [Buka Aplikasi di Streamlit Cloud](https://proyek-citra-revisi.streamlit.app)

---

## Daftar Isi

- [Latar Belakang](#latar-belakang)
- [Fitur Utama](#fitur-utama)
- [Arsitektur Sistem](#arsitektur-sistem)
- [Struktur Repositori](#struktur-repositori)
- [Prasyarat](#prasyarat)
- [Instalasi dan Cara Menjalankan](#instalasi-dan-cara-menjalankan)
- [Alur Kerja Aplikasi](#alur-kerja-aplikasi)
- [Penjelasan Teknis](#penjelasan-teknis)
  - [Noise Salt-and-Pepper](#noise-salt-and-pepper)
  - [Filter Median](#filter-median)
  - [Filter Gaussian](#filter-gaussian)
  - [Metrik Evaluasi: MSE dan PSNR](#metrik-evaluasi-mse-dan-psnr)
  - [Mengapa Median Lebih Unggul?](#mengapa-median-lebih-unggul-untuk-noise-salt-and-pepper)
  - [Pengaruh Ukuran Kernel](#pengaruh-ukuran-kernel-3x3-vs-5x5)
- [Integrasi AI (Google Gemini)](#integrasi-ai-google-gemini)
- [Unit Testing](#unit-testing)
- [Deployment](#deployment)
- [Reproduksibilitas](#reproduksibilitas)
- [Teknologi yang Digunakan](#teknologi-yang-digunakan)
- [Referensi Akademis](#referensi-akademis)

---

## Latar Belakang

Noise Salt-and-Pepper adalah jenis derau impulsif yang sering terjadi pada citra digital akibat kesalahan transmisi data, gangguan sensor kamera, atau kerusakan memori penyimpanan. Noise ini memiliki karakteristik unik: piksel yang terdampak berubah secara ekstrem menjadi nilai **0 (hitam/Pepper)** atau **255 (putih/Salt)**, sementara piksel lainnya tidak terpengaruh.

Pertanyaan penelitian: **Filter mana yang paling efektif mereduksi noise Salt-and-Pepper — Filter Median yang bersifat non-linier, atau Filter Gaussian yang bersifat linier?**

Proyek ini menjawab pertanyaan tersebut melalui eksperimen kuantitatif menggunakan tiga citra uji standar (Lena, Cameraman, Baboon) dengan variasi densitas noise dan ukuran kernel filter.

---

## Fitur Utama

| Fitur | Deskripsi |
|-------|-----------|
| **Aplikasi Web Interaktif** | Antarmuka Streamlit dengan tema Darkroom Film Lab, animasi CSS cinematic, dan visualisasi real-time |
| **Grafik Interaktif Plotly** | Bar chart PSNR dan MSE dengan hover tooltip, zoom, dan pan — bukan gambar statis |
| **Integrasi AI (Gemini)** | Penjelasan otomatis hasil eksperimen oleh Google Gemini dalam Bahasa Indonesia akademis |
| **Multi-turn Q&A** | Tanya jawab interaktif dengan AI, riwayat percakapan tersimpan selama sesi berlangsung |
| **Fallback Offline** | Jika API Gemini tidak tersedia, penjelasan otomatis berbasis teori Gonzalez & Woods tetap ditampilkan |
| **Upload Citra Custom** | Selain citra preset, pengguna bisa upload citra sendiri (JPG/PNG/BMP/TIFF) |
| **Download Hasil** | Unduh metrik CSV, citra hasil filter terbaik, dan citra ber-noise dalam format PNG |
| **Unit Testing** | 8 test case memvalidasi kebenaran MSE, PSNR, noise injection, dan kedua filter |
| **Reproduksibilitas** | Random seed tetap (`np.random.seed(42)`) menjamin hasil identik setiap eksekusi |

---

## Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT WEB APP                        │
│                          (app.py)                               │
│                                                                 │
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌────────────┐  │
│  │ Sidebar  │  │  Visualisasi │  │  Grafik  │  │  Download  │  │
│  │ Controls │  │  Citra       │  │  Plotly  │  │  Hasil     │  │
│  └────┬─────┘  └──────┬───────┘  └────┬─────┘  └────────────┘  │
│       │               │               │                         │
│  ┌────▼───────────────▼───────────────▼──────────────────────┐  │
│  │              FILTER COMPARISON ENGINE                      │  │
│  │              (filter_comparison.py)                        │  │
│  │                                                           │  │
│  │  add_salt_pepper_noise() → apply_median_filter()          │  │
│  │                          → apply_gaussian_filter()        │  │
│  │                          → calculate_mse()                │  │
│  │                          → calculate_psnr()               │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              GEMINI AI INTEGRATION                         │  │
│  │              (gemini_helper.py)                            │  │
│  │                                                           │  │
│  │  generate_explanation() → Model Fallback Chain            │  │
│  │  ask_followup()         → Multi-turn Chat                 │  │
│  │  _get_fallback_explanation() → Offline Template           │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Struktur Repositori

```
proyek-citra-revisi/
├── app.py                    # Aplikasi web Streamlit (UI, visualisasi, integrasi)
├── filter_comparison.py      # Inti logika: noise injection, filtering, metrik
├── gemini_helper.py          # Integrasi Google Gemini API + fallback offline
├── test_metrics.py           # 8 unit test (MSE, PSNR, noise, filter)
├── download_test_images.py   # Download citra uji standar + fallback sintetis
├── requirements.txt          # Dependensi Python
├── NASKAH_PRESENTASI.txt     # Naskah presentasi untuk demo
├── CLAUDE.md                 # Panduan untuk Claude Code
├── .gitignore                # Konfigurasi git ignore
├── .streamlit/
│   └── config.toml           # Konfigurasi tema Streamlit
└── images/                   # Citra uji (Lena, Cameraman, Baboon)
    ├── lena.tif
    ├── cameraman.tif
    └── baboon.png
```

### Peran Setiap File

| File | Peran |
|------|-------|
| `app.py` | Antarmuka web utama: memuat citra, mengatur parameter noise, menampilkan hasil filtering, grafik Plotly interaktif, metrik perbandingan, penjelasan AI, dan fitur download |
| `filter_comparison.py` | Modul inti pengolahan citra: fungsi noise injection manual dengan NumPy, penerapan filter via OpenCV (`medianBlur`, `GaussianBlur`), dan perhitungan metrik MSE/PSNR |
| `gemini_helper.py` | Modul integrasi Google Gemini API dengan mekanisme fallback bertingkat (4 model) dan penjelasan offline berbasis teori jika API tidak tersedia |
| `test_metrics.py` | 8 unit test menggunakan `unittest` untuk memvalidasi kebenaran seluruh fungsi komputasi |
| `download_test_images.py` | Mengunduh citra uji standar dari URL publik; jika gagal, membuat citra sintetis geometris sebagai fallback |

---

## Prasyarat

- **Python** 3.8 atau lebih baru
- **pip** (package manager Python)
- **Google Gemini API Key** (opsional — untuk fitur penjelasan AI)

---

## Instalasi dan Cara Menjalankan

### 1. Clone Repositori

```bash
git clone https://github.com/zhafirash04/proyek-citra-revisi.git
cd proyek-citra-revisi
```

### 2. Install Dependensi

```bash
pip install -r requirements.txt
```

### 3. Siapkan Citra Uji

```bash
python download_test_images.py
```

Ini akan mengunduh citra Lena, Cameraman, dan Baboon ke folder `images/`. Jika download gagal (misalnya tidak ada koneksi internet), program otomatis membuat citra sintetis geometris sebagai pengganti.

### 4. (Opsional) Konfigurasi API Key Gemini

Buat file `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "ISI_API_KEY_ANDA_DI_SINI"
```

> Dapatkan API key gratis di [Google AI Studio](https://aistudio.google.com/apikey). Model `gemini-1.5-flash` memiliki kuota 1.500 request/hari pada tier gratis.

### 5. Jalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`.

### 6. Jalankan Unit Test

```bash
python -m unittest test_metrics.py -v
```

Untuk menjalankan test spesifik:

```bash
python -m unittest test_metrics.TestMetrics.test_calculate_mse_identical_images
```

---

## Alur Kerja Aplikasi

Aplikasi mengikuti alur 6 langkah (step) yang berurutan:

### Step 1 — Muat Citra dan Tambahkan Noise

Pengguna memilih citra uji (preset atau upload) dan mengatur densitas noise (5%–50%) di sidebar. Setelah klik **Mulai Proses**, noise Salt-and-Pepper diinjeksikan ke citra. Tampilan menunjukkan citra asli vs citra ber-noise secara berdampingan.

### Step 2 — Hasil Filtering

Sistem otomatis menguji 4 konfigurasi filter:
1. Median Filter 3×3
2. Median Filter 5×5
3. Gaussian Filter 3×3 (σ = 1.0)
4. Gaussian Filter 5×5 (σ = 1.0)

Keempat hasil ditampilkan berdampingan untuk perbandingan visual.

### Step 3 — Analisis Metrik

Kartu metrik menampilkan: filter terbaik, PSNR tertinggi, MSE terendah, dan peningkatan PSNR dari citra ber-noise. Tabel perbandingan lengkap dan grafik bar interaktif (Plotly) menyajikan data secara kuantitatif.

### Step 4 — Rekomendasi

Sistem secara otomatis merekomendasikan filter dengan PSNR tertinggi. Rekomendasi dihitung dinamis berdasarkan data eksperimen — bukan hardcoded.

### Step 5 — Penjelasan AI

Tombol **Generate Penjelasan** mengirim data eksperimen ke Google Gemini API dan menampilkan penjelasan akademis dalam Bahasa Indonesia. Fitur tanya jawab multi-turn memungkinkan pengguna bertanya lebih lanjut dengan riwayat percakapan yang tersimpan.

### Step 6 — Download Hasil

Pengguna dapat mengunduh:
- Metrik dalam format CSV
- Citra hasil filter terbaik (PNG)
- Citra ber-noise (PNG)

---

## Penjelasan Teknis

### Noise Salt-and-Pepper

Noise Salt-and-Pepper (juga disebut noise impulsif) adalah jenis derau di mana piksel secara acak berubah menjadi nilai ekstrem:

- **Salt (putih):** piksel → 255
- **Pepper (hitam):** piksel → 0

Penyebab di dunia nyata:
- Kesalahan transmisi data pada saluran komunikasi
- Gangguan sensor kamera (dead pixel, hot pixel)
- Kerusakan memori penyimpanan
- Interferensi elektromagnetik

**Implementasi dalam kode** (`filter_comparison.py:add_salt_pepper_noise`):

```python
# Setiap piksel memiliki probabilitas density/2 menjadi Salt dan density/2 menjadi Pepper
num_salt = int(total_pixels * density / 2)
num_pepper = int(total_pixels * density / 2)

# Pilih posisi acak tanpa pengembalian (replace=False) untuk presisi matematis
flat_indices = np.random.choice(total_pixels, num_salt + num_pepper, replace=False)

# Salt → 255, Pepper → 0
noisy_image[salt_rows, salt_cols] = 255
noisy_image[pepper_rows, pepper_cols] = 0
```

Catatan penting: `np.random.choice(..., replace=False)` menjamin tidak ada piksel yang terkena noise dua kali, sehingga jumlah piksel noise persis sesuai densitas yang ditetapkan.

### Filter Median

Filter Median adalah **filter non-linier** berbasis statistik urutan (order-statistic filter).

**Cara kerja:**
1. Ambil semua piksel di dalam jendela kernel (misalnya 3×3 = 9 piksel)
2. Urutkan nilai piksel dari terkecil ke terbesar
3. Pilih nilai tengah (median) sebagai nilai piksel baru

**Contoh dengan kernel 3×3:**
```
Piksel tetangga: [10, 12, 11, 255, 10, 13, 10, 11, 12]
Setelah diurutkan: [10, 10, 10, 11, 11, 12, 12, 13, 255]
Median (posisi ke-5): 11
```

Perhatikan bahwa nilai noise 255 berada di ujung urutan dan **tidak pernah terpilih** sebagai median. Inilah mengapa filter Median sangat efektif untuk noise Salt-and-Pepper.

**Implementasi:** `cv2.medianBlur(image, ksize)`

### Filter Gaussian

Filter Gaussian adalah **filter linier** berbasis konvolusi dengan kernel distribusi Gaussian.

**Cara kerja:**
1. Buat kernel berisi bobot yang mengikuti distribusi Gaussian (bell curve)
2. Geser kernel di atas setiap piksel citra
3. Hitung rata-rata tertimbang (weighted average) piksel tetangga
4. Hasilnya menggantikan piksel di posisi tengah kernel

**Kernel Gaussian 3×3 (σ = 1.0):**
```
[ 0.075  0.124  0.075 ]
[ 0.124  0.204  0.124 ]
[ 0.075  0.124  0.075 ]
```

Piksel di tengah mendapat bobot terbesar (0.204), dan bobot menurun seiring jarak.

**Masalah untuk noise Salt-and-Pepper:** Karena ini operasi rata-rata tertimbang, piksel noise bernilai 0 atau 255 **tetap ikut dihitung**. Akibatnya, noise tidak hilang melainkan menyebar ke piksel sekitarnya, menghasilkan efek kabur (blurring).

**Implementasi:** `cv2.GaussianBlur(image, (ksize, ksize), sigma)`

**Parameter Sigma (σ):** Standar deviasi distribusi Gaussian yang menentukan seberapa lebar pengaruh pembobotan. Sigma kecil (1.0) → smoothing ringan. Sigma besar → blur lebih kuat.

### Metrik Evaluasi: MSE dan PSNR

#### MSE (Mean Squared Error)

MSE menghitung rata-rata kuadrat selisih antara citra asli dan citra hasil filter:

```
MSE = (1 / N) × Σ (piksel_asli - piksel_hasil)²
```

di mana N = jumlah total piksel (512 × 512 = 262.144).

- **MSE = 0** → citra identik sempurna
- **Semakin kecil MSE** → semakin mirip dengan citra asli → semakin baik

**Implementasi:**
```python
diff = original.astype(np.float64) - filtered.astype(np.float64)
mse = np.mean(diff ** 2)
```

> Konversi ke `float64` mencegah integer overflow. Tanpa konversi: `(0 - 255)²` pada `uint8` akan menghasilkan nilai salah karena overflow aritmetika unsigned.

#### PSNR (Peak Signal-to-Noise Ratio)

PSNR mengukur kualitas restorasi citra dalam satuan desibel (dB):

```
PSNR = 10 × log₁₀(MAX² / MSE)
```

di mana MAX = 255 (nilai maksimum piksel 8-bit).

Interpretasi:
| PSNR | Kualitas |
|------|----------|
| > 40 dB | Sangat baik (perbedaan hampir tidak terlihat) |
| 30–40 dB | Baik (perbedaan minor) |
| 20–30 dB | Cukup (perbedaan terlihat) |
| < 20 dB | Kurang baik |

**Kasus khusus:** Jika MSE = 0 (citra identik), PSNR = ∞ (infinity). Kode menangani ini secara eksplisit:

```python
def calculate_psnr(mse):
    if mse == 0:
        return float('inf')
    return 10 * np.log10((MAX_PIXEL_VALUE ** 2) / mse)
```

### Mengapa Median Lebih Unggul untuk Noise Salt-and-Pepper?

Jawabannya terletak pada sifat fundamental kedua filter:

**Filter Median (non-linier):**
- Bekerja dengan **statistik urutan** — memilih nilai tengah
- Noise Salt-and-Pepper bernilai 0 atau 255 → selalu di ujung urutan
- Nilai ekstrem **tidak pernah terpilih** sebagai median
- Hasilnya: noise tereliminasi sempurna, tepi citra tetap tajam

**Filter Gaussian (linier):**
- Bekerja dengan **rata-rata tertimbang** — semua piksel ikut dihitung
- Noise bernilai 0 atau 255 **tetap berkontribusi** pada rata-rata
- Hasilnya: noise menyebar ke piksel sekitar, muncul artefak keabuan (ghosting)
- Citra juga menjadi lebih kabur secara keseluruhan

**Analogi:**
> Bayangkan sebuah ruangan dengan 9 orang berbicara normal, lalu 1 orang berteriak sangat keras (noise).
> - **Median Filter:** Mengabaikan teriakan, mendengarkan suara mayoritas → jernih
> - **Gaussian Filter:** Merata-ratakan semua suara termasuk teriakan → semua jadi sedikit bising

Referensi: Gonzalez, R.C. & Woods, R.E., *Digital Image Processing*, 4th Ed., Chapter 5 (Image Restoration and Reconstruction).

### Pengaruh Ukuran Kernel (3×3 vs 5×5)

| Aspek | Kernel 3×3 | Kernel 5×5 |
|-------|-----------|-----------|
| Area | 9 piksel tetangga | 25 piksel tetangga |
| Detail tepi | Lebih terjaga | Lebih banyak hilang |
| Tahan noise tinggi | Kurang (9 sampel kecil) | Lebih (25 sampel besar) |
| Kecepatan | Lebih cepat | Lebih lambat |
| Cocok untuk | Noise rendah–sedang | Noise tinggi |

**Trade-off klasik:** Semakin besar kernel → semakin kuat reduksi noise → semakin banyak detail halus yang dikorbankan.

Pada noise densitas rendah (≤ 20%), Median 3×3 biasanya sudah cukup karena dari 9 piksel, mayoritas masih bersih. Pada densitas tinggi (> 30%), Median 5×5 lebih stabil karena area referensi 25 piksel memberikan sampling yang lebih representatif.

---

## Integrasi AI (Google Gemini)

### Arsitektur Fallback Bertingkat

Modul `gemini_helper.py` menggunakan sistem percobaan bertingkat untuk keandalan maksimal:

```
Urutan percobaan model:
1. gemini-1.5-flash      (1.500 RPD, paling stabil)
2. gemini-2.5-flash       (cadangan)
3. gemini-2.5-flash-lite  (20 RPD, kuota kecil)
4. gemini-flash-lite-latest (cadangan terakhir)
           ↓
    Jika semua gagal:
5. Penjelasan offline (template berbasis teori Gonzalez & Woods)
```

> **RPD** = Requests Per Day pada tier gratis Google AI.

### Fitur AI

1. **Generate Penjelasan** — Mengirim data eksperimen (PSNR, MSE, densitas noise, filter terbaik) ke Gemini dan meminta penjelasan akademis dalam Bahasa Indonesia
2. **Tanya Jawab Multi-turn** — Pengguna bisa bertanya berkali-kali, riwayat percakapan tersimpan di `st.session_state`
3. **Fallback Offline** — Jika API tidak tersedia (kuota habis, tanpa internet, API key tidak dikonfigurasi), penjelasan template tetap ditampilkan berdasarkan data eksperimen aktual

---

## Unit Testing

8 test case memvalidasi seluruh fungsi komputasi:

```
Test 1: test_calculate_mse_identical_images
        → MSE citra identik harus = 0.0

Test 2: test_calculate_mse_different_images
        → MSE citra berbeda dihitung benar (selisih 10 → MSE = 100.0)

Test 3: test_calculate_mse_overflow_handling
        → Overflow uint8 ditangani (0 vs 255 → MSE = 65025.0, bukan nilai salah)

Test 4: test_calculate_psnr_zero_mse
        → PSNR untuk MSE = 0 harus = infinity

Test 5: test_calculate_psnr_normal
        → PSNR dihitung sesuai rumus 10 × log₁₀(255² / MSE)

Test 6: test_add_salt_pepper_noise_density
        → Jumlah piksel noise akurat sesuai densitas (toleransi ±2 piksel)

Test 7: test_apply_median_filter_odd_kernel
        → Median 3×3 berhasil menghapus single noise pixel (255 → 10)

Test 8: test_apply_gaussian_filter_smoothing
        → Gaussian smoothing mengurangi nilai piksel tengah (255 → < 255)
```

Jalankan dengan: `python -m unittest test_metrics.py -v`

---

## Deployment

### Streamlit Cloud

Aplikasi di-deploy di Streamlit Cloud dari repositori GitHub ini. Setiap push ke branch `main` otomatis memicu re-deploy.

**Konfigurasi API Key di Streamlit Cloud:**
1. Buka dashboard Streamlit Cloud
2. Klik **Manage app** → **Settings** → **Secrets**
3. Tambahkan:
   ```toml
   GEMINI_API_KEY = "ISI_API_KEY_ANDA"
   ```

**Catatan teknis:** Repositori ini menggunakan `opencv-python-headless` (bukan `opencv-python`) karena Streamlit Cloud (Linux) tidak menyediakan `libGL.so` yang dibutuhkan oleh versi non-headless. Versi headless memiliki fungsionalitas identik untuk pemrosesan citra — hanya menghilangkan modul GUI (`cv2.imshow`, dll.) yang memang tidak diperlukan di web app.

---

## Reproduksibilitas

Setiap eksekusi menggunakan random seed tetap:

```python
np.random.seed(42)
```

Ini menjamin:
- Pola noise Salt-and-Pepper **identik** setiap kali program dijalankan
- Nilai MSE dan PSNR **konsisten** dan dapat diverifikasi ulang
- Hasil eksperimen **dapat direproduksi** oleh siapapun yang menjalankan kode ini

Seed 42 dipilih sebagai konvensi umum dalam komunitas ilmiah (referensi ke *The Hitchhiker's Guide to the Galaxy*).

---

## Teknologi yang Digunakan

| Teknologi | Kegunaan |
|-----------|----------|
| **Python 3** | Bahasa pemrograman utama |
| **Streamlit** | Framework web app untuk data science |
| **OpenCV** (`opencv-python-headless`) | Penerapan filter Median dan Gaussian |
| **NumPy** | Operasi array untuk noise injection dan perhitungan metrik |
| **Plotly** | Grafik interaktif (bar chart PSNR/MSE) |
| **Pandas** | Manipulasi data tabel dan ekspor CSV |
| **Pillow** | Konversi format gambar dan resize |
| **Google Generative AI** | Integrasi Gemini API untuk penjelasan otomatis |
| **Matplotlib** | Digunakan oleh `filter_comparison.py` untuk visualisasi standalone |

---

## Referensi Akademis

1. **Gonzalez, R.C. & Woods, R.E.** (2018). *Digital Image Processing*, 4th Edition. Pearson. — Teori filter spasial, noise model, dan metrik kualitas citra (Chapter 3, 5).

2. **Bovik, A.C.** (2009). *The Essential Guide to Image Processing*. Academic Press. — Evaluasi performa filter dengan PSNR dan MSE.

3. **IEEE Standard** — Format penulisan dan struktur laporan teknis.

---

<p align="center">
  <b>Proyek Pengolahan Citra Digital</b><br>
  Median vs Gaussian — Reduksi Noise Salt-and-Pepper<br>
  &copy; 2026
</p>
