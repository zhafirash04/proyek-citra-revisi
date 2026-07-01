"""
Script untuk memvisualisasikan data hasil perbandingan (MSE & PSNR) 
ke dalam bentuk grafik Bar dan kurva (Line Chart).
"""

import matplotlib.pyplot as plt
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# MEMBACA DATA DARI CSV
# =====================================================================
import csv
from collections import defaultdict

CSV_PATH = os.path.join(OUTPUT_DIR, "results.csv")

noise_levels = []
x_pos = []
width = 0.2

# Data Rata-rata PSNR
psnr_median_3x3 = []
psnr_median_5x5 = []
psnr_gaussian_3x3 = []
psnr_gaussian_5x5 = []

# Data Rata-rata MSE
mse_median_3x3 = []
mse_median_5x5 = []
mse_gaussian_3x3 = []
mse_gaussian_5x5 = []

def load_data():
    global noise_levels, x_pos
    global psnr_median_3x3, psnr_median_5x5, psnr_gaussian_3x3, psnr_gaussian_5x5
    global mse_median_3x3, mse_median_5x5, mse_gaussian_3x3, mse_gaussian_5x5
    
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} tidak ditemukan.")
        print("Jalankan filter_comparison.py terlebih dahulu.")
        exit(1)
        
    data_psnr = defaultdict(lambda: defaultdict(list))
    data_mse = defaultdict(lambda: defaultdict(list))
    
    with open(CSV_PATH, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            density = float(row['Noise Density'])
            filter_type = row['Filter']
            kernel = row['Kernel Size']
            # Hindari nilai string "inf"
            psnr_val = float(row['PSNR']) if row['PSNR'] != 'inf' else 0.0
            mse_val = float(row['MSE'])
            
            filter_kernel = f"{filter_type} {kernel}"
            data_psnr[density][filter_kernel].append(psnr_val)
            data_mse[density][filter_kernel].append(mse_val)
            
    # Calculate means
    densities = sorted(data_psnr.keys())
    noise_levels = [f"{int(d*100)}%" for d in densities]
    x_pos = np.arange(len(noise_levels))
    
    for d in densities:
        psnr_median_3x3.append(np.mean(data_psnr[d]["Median 3x3"]) if data_psnr[d]["Median 3x3"] else 0)
        psnr_median_5x5.append(np.mean(data_psnr[d]["Median 5x5"]) if data_psnr[d]["Median 5x5"] else 0)
        psnr_gaussian_3x3.append(np.mean(data_psnr[d]["Gaussian 3x3"]) if data_psnr[d]["Gaussian 3x3"] else 0)
        psnr_gaussian_5x5.append(np.mean(data_psnr[d]["Gaussian 5x5"]) if data_psnr[d]["Gaussian 5x5"] else 0)

        mse_median_3x3.append(np.mean(data_mse[d]["Median 3x3"]) if data_mse[d]["Median 3x3"] else 0)
        mse_median_5x5.append(np.mean(data_mse[d]["Median 5x5"]) if data_mse[d]["Median 5x5"] else 0)
        mse_gaussian_3x3.append(np.mean(data_mse[d]["Gaussian 3x3"]) if data_mse[d]["Gaussian 3x3"] else 0)
        mse_gaussian_5x5.append(np.mean(data_mse[d]["Gaussian 5x5"]) if data_mse[d]["Gaussian 5x5"] else 0)



def plot_psnr_line_chart():
    """Membuat Line Chart untuk Rata-rata PSNR."""
    plt.figure(figsize=(10, 6))
    
    plt.plot(noise_levels, psnr_median_3x3, marker='o', linestyle='-', linewidth=2, label='Median 3x3', color='blue')
    plt.plot(noise_levels, psnr_median_5x5, marker='s', linestyle='--', linewidth=2, label='Median 5x5', color='cyan')
    plt.plot(noise_levels, psnr_gaussian_3x3, marker='^', linestyle='-', linewidth=2, label='Gaussian 3x3', color='red')
    plt.plot(noise_levels, psnr_gaussian_5x5, marker='D', linestyle='--', linewidth=2, label='Gaussian 5x5', color='orange')
    
    plt.title('Perbandingan Rata-rata PSNR (Semua Citra) terhadap Densitas Noise', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Densitas Noise Salt-and-Pepper (%)', fontsize=12)
    plt.ylabel('Rata-rata PSNR (dB)', fontsize=12)
    plt.ylim(15, 35)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize=10)
    
    # Tambahkan nilai di setiap titik data
    for i, txt in enumerate(psnr_median_3x3):
        plt.annotate(f"{txt:.2f}", (noise_levels[i], psnr_median_3x3[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    filepath = os.path.join(OUTPUT_DIR, 'kurva_rata_rata_psnr.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Grafik disimpan di: {filepath}")

def plot_psnr_bar_chart():
    """Membuat Bar Chart untuk Rata-rata PSNR."""
    plt.figure(figsize=(12, 7))
    
    rects1 = plt.bar(x_pos - 1.5*width, psnr_median_3x3, width, label='Median 3x3', color='#1f77b4')
    rects2 = plt.bar(x_pos - 0.5*width, psnr_median_5x5, width, label='Median 5x5', color='#4fc1e8')
    rects3 = plt.bar(x_pos + 0.5*width, psnr_gaussian_3x3, width, label='Gaussian 3x3', color='#d62728')
    rects4 = plt.bar(x_pos + 1.5*width, psnr_gaussian_5x5, width, label='Gaussian 5x5', color='#ff9896')
    
    plt.title('Diagram Batang Rata-rata PSNR (Semua Citra)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Densitas Noise Salt-and-Pepper (%)', fontsize=12)
    plt.ylabel('Rata-rata PSNR (dB)', fontsize=12)
    plt.xticks(x_pos, noise_levels)
    plt.ylim(0, 35)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Fungsi untuk menambahkan text/label di atas bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    
    filepath = os.path.join(OUTPUT_DIR, 'bar_rata_rata_psnr.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Grafik disimpan di: {filepath}")

def plot_mse_line_chart():
    """Membuat Line Chart untuk Rata-rata MSE."""
    plt.figure(figsize=(10, 6))
    
    plt.plot(noise_levels, mse_median_3x3, marker='o', linestyle='-', linewidth=2, label='Median 3x3', color='blue')
    plt.plot(noise_levels, mse_median_5x5, marker='s', linestyle='--', linewidth=2, label='Median 5x5', color='cyan')
    plt.plot(noise_levels, mse_gaussian_3x3, marker='^', linestyle='-', linewidth=2, label='Gaussian 3x3', color='red')
    plt.plot(noise_levels, mse_gaussian_5x5, marker='D', linestyle='--', linewidth=2, label='Gaussian 5x5', color='orange')
    
    plt.title('Perbandingan Rata-rata MSE (Semua Citra) terhadap Densitas Noise', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Densitas Noise Salt-and-Pepper (%)', fontsize=12)
    plt.ylabel('Rata-rata MSE', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', fontsize=10)
    
    # Tambahkan nilai di setiap titik data untuk Median 3x3 (karena nilai paling kecil seringnya)
    for i, txt in enumerate(mse_median_3x3):
        plt.annotate(f"{txt:.2f}", (noise_levels[i], mse_median_3x3[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    filepath = os.path.join(OUTPUT_DIR, 'kurva_rata_rata_mse.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Grafik disimpan di: {filepath}")

def plot_mse_bar_chart():
    """Membuat Bar Chart untuk Rata-rata MSE."""
    plt.figure(figsize=(12, 7))
    
    rects1 = plt.bar(x_pos - 1.5*width, mse_median_3x3, width, label='Median 3x3', color='#1f77b4')
    rects2 = plt.bar(x_pos - 0.5*width, mse_median_5x5, width, label='Median 5x5', color='#4fc1e8')
    rects3 = plt.bar(x_pos + 0.5*width, mse_gaussian_3x3, width, label='Gaussian 3x3', color='#d62728')
    rects4 = plt.bar(x_pos + 1.5*width, mse_gaussian_5x5, width, label='Gaussian 5x5', color='#ff9896')
    
    plt.title('Diagram Batang Rata-rata MSE (Semua Citra)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Densitas Noise Salt-and-Pepper (%)', fontsize=12)
    plt.ylabel('Rata-rata MSE', fontsize=12)
    plt.xticks(x_pos, noise_levels)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Fungsi untuk menambahkan text/label di atas bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    
    filepath = os.path.join(OUTPUT_DIR, 'bar_rata_rata_mse.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Grafik disimpan di: {filepath}")

if __name__ == "__main__":
    print("Membuat visualisasi grafik...")
    load_data()
    plot_psnr_line_chart()
    plot_psnr_bar_chart()
    plot_mse_line_chart()
    plot_mse_bar_chart()
    print("Selesai!")
