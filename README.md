# Perbandingan Filter Median dan Gaussian untuk Reduksi Noise Salt-and-Pepper

Proyek ini adalah alat riset berbasis Python untuk mengevaluasi performa **Filter Median** dan **Filter Gaussian** dalam mereduksi noise Salt-and-Pepper pada citra grayscale. Pengukuran performa dilakukan dengan metrik Mean Squared Error (MSE) dan Peak Signal-to-Noise Ratio (PSNR), dan didesain agar kompatibel dengan standar laporan teknis dan jurnal IEEE.

## Fitur Utama
- Penambahan noise Salt-and-Pepper yang terautomasi secara manual (tanpa bergantung pada pustaka eksternal) untuk visibilitas matematis yang jelas.
- Dukungan variasi tingkat noise (density): 10%, 20%, 30%.
- Filter Median dan Gaussian menggunakan library OpenCV.
- Pengujian menggunakan ukuran kernel 3x3 dan 5x5.
- Validasi input (ukuran kernel ganjil, density antara 0 dan 1).
- Perhitungan MSE dan PSNR yang ditangani langsung dengan array NumPy.
- Output berupa perbandingan visual (baris dan kolom), tabel ringkasan di konsol, dan ekspor data eksperimen ke dalam format `.csv` untuk reproduksibilitas.
- Modul pembuatan grafik otomatis (Line Chart dan Bar Chart).
- Pengujian Unit Test menggunakan `unittest`.

## Struktur Repositori
- `filter_comparison.py`: Skrip utama untuk menjalankan eksperimen noise, filtering, perhitungan metrik, serta pembuatan tabel CLI dan ekspor CSV.
- `visualisasi_grafik.py`: Skrip untuk membaca `output/results.csv` dan membuat visualisasi berupa Line Chart dan Bar Chart untuk perbandingan rata-rata PSNR.
- `test_metrics.py`: Skrip Unit Test untuk memvalidasi fungsi kalkulasi MSE dan PSNR.
- `images/`: Folder tempat menempatkan citra uji standar (seperti lena, cameraman, baboon).
- `output/`: Folder tempat menyimpan output grafis dan CSV eksperimen.

## Prasyarat (Requirements)
Proyek ini membutuhkan Python 3.x dan pustaka berikut:
- `numpy`
- `opencv-python` (`cv2`)
- `matplotlib`

Anda bisa menginstalnya dengan menjalankan:
```bash
pip install numpy opencv-python matplotlib
```

## Cara Menjalankan Program

### 1. Persiapan Citra Uji
Pastikan folder `images/` memiliki citra berformat `.png`, `.jpg`, atau `.tif`. Nama yang disarankan untuk pengujian ini: `lena`, `cameraman`, dan `baboon`. 

### 2. Jalankan Skrip Eksperimen Utama
Jalankan file `filter_comparison.py`:
```bash
python filter_comparison.py
```
**Apa yang terjadi?**
- Program akan menambahkan noise, memproses filter, mencetak tabel di terminal, dan menyimpan gambar hasil (side-by-side) serta tabel CSV di dalam folder `output/`.

### 3. Hasilkan Grafik
Setelah CSV berhasil dibuat, hasilkan visualisasi grafik (Bar Chart & Line Chart) dengan menjalankan:
```bash
python visualisasi_grafik.py
```
**Apa yang terjadi?**
- Program akan membaca file `output/results.csv`, memproses rata-ratanya, dan menyimpan file `kurva_rata_rata_psnr.png` serta `bar_rata_rata_psnr.png` di folder `output/`.

### 4. Menjalankan Unit Test
Untuk memastikan fungsi MSE dan PSNR berjalan akurat:
```bash
python -m unittest test_metrics.py
```

## Reproduksibilitas (Reproducibility)
Setiap eksekusi modul noise menggunakan *random seed* tetap (`np.random.seed(42)`). Ini berarti hasil perbandingan dari noise yang di-generate selalu konsisten setiap kali skrip dijalankan, mempermudah replikasi hasil dalam riset atau jurnal teknis.
