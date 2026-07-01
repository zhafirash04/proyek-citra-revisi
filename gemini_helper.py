"""
Modul integrasi Google Gemini API untuk penjelasan hasil filtering.
Menyediakan fallback otomatis jika API tidak tersedia.
"""

import google.generativeai as genai

GEMINI_MODELS = [
    "gemini-1.5-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-flash-lite-latest"
]

def _build_prompt(results_data):
    """Bangun prompt untuk Gemini berdasarkan data eksperimen."""
    d = results_data
    density_pct = d["noise_density"] * 100

    # Format tabel hasil filter
    filter_rows = ""
    for f in d["filters"]:
        label = f"{f['filter_type']} {f['kernel_size']}x{f['kernel_size']}"
        filter_rows += f"  - {label}: PSNR = {f['psnr']:.2f} dB, MSE = {f['mse']:.2f}\n"

    best = d["best"]
    best_label = f"{best['filter_type']} {best['kernel_size']}x{best['kernel_size']}"

    prompt = f"""Kamu adalah dosen Pengolahan Citra Digital di universitas.
Jelaskan hasil eksperimen filtering noise berikut dalam Bahasa Indonesia
yang akademis namun mudah dipahami mahasiswa.

DATA EKSPERIMEN:
- Citra: grayscale 512x512 piksel
- Jenis noise: Salt-and-Pepper
- Densitas noise: {density_pct:.0f}%
- Hasil filtering:
{filter_rows}
- Citra ber-noise (tanpa filter): PSNR = {d['noisy_psnr']:.2f} dB, MSE = {d['noisy_mse']:.2f}
- Filter terbaik: {best_label} (PSNR tertinggi: {best['psnr']:.2f} dB)

Berikan penjelasan yang mencakup:
1. Mengapa {best_label} memberikan hasil terbaik untuk noise Salt-and-Pepper?
   Jelaskan secara teori (Gonzalez & Woods).
2. Bagaimana perbedaan kernel 3x3 dan 5x5 mempengaruhi hasil?
3. Interpretasi nilai PSNR dan MSE yang didapat.
4. Kesimpulan dan rekomendasi filter untuk kasus ini.

Gunakan format yang rapi dengan heading dan bullet points.
Panjang sekitar 300-400 kata."""

    return prompt


def _get_fallback_explanation(results_data):
    """Penjelasan fallback jika Gemini tidak tersedia."""
    d = results_data
    density_pct = d["noise_density"] * 100
    best = d["best"]
    best_label = f"{best['filter_type']} {best['kernel_size']}x{best['kernel_size']}"

    # Cari data per filter untuk perbandingan
    median_results = [f for f in d["filters"] if f["filter_type"] == "Median"]
    gaussian_results = [f for f in d["filters"] if f["filter_type"] == "Gaussian"]
    best_median = max(median_results, key=lambda x: x["psnr"])
    best_gaussian = max(gaussian_results, key=lambda x: x["psnr"])

    explanation = f"""## Analisis Hasil Filtering

### Ringkasan
Pada densitas noise **{density_pct:.0f}%**, filter **{best_label}** memberikan
hasil terbaik dengan **PSNR {best['psnr']:.2f} dB** dan **MSE {best['mse']:.2f}**.

### Mengapa {best['filter_type']} Lebih Baik?

Noise Salt-and-Pepper bersifat **impulsif** — piksel berubah secara ekstrem
ke nilai 0 (hitam) atau 255 (putih). Karakteristik ini membuat:

- **Median Filter** sangat efektif karena operasi median secara natural
  mengabaikan nilai-nilai ekstrem (outlier). Piksel noise yang bernilai 0
  atau 255 tidak mempengaruhi hasil median dari tetangganya.
- **Gaussian Filter** kurang efektif karena operasi rata-rata tertimbang
  tetap memasukkan piksel noise ke dalam perhitungan, sehingga noise
  "menyebar" ke piksel sekitarnya (*blurring*).

### Pengaruh Ukuran Kernel

| Kernel | Kelebihan | Kekurangan |
|--------|-----------|------------|
| **3×3** | Detail citra lebih terjaga | Kurang efektif pada noise tinggi |
| **5×5** | Lebih efektif pada noise tinggi | Mengurangi detail/ketajaman citra |

Pada densitas {density_pct:.0f}%:
- Median terbaik: **{best_median['kernel_size']}x{best_median['kernel_size']}** \
(PSNR: {best_median['psnr']:.2f} dB)
- Gaussian terbaik: **{best_gaussian['kernel_size']}x{best_gaussian['kernel_size']}** \
(PSNR: {best_gaussian['psnr']:.2f} dB)

### Interpretasi Metrik

- **PSNR > 30 dB**: Kualitas baik, noise tereduksi secara signifikan
- **PSNR 20-30 dB**: Kualitas cukup, noise masih terlihat
- **PSNR < 20 dB**: Kualitas kurang, filter tidak cukup efektif

### Kesimpulan

Untuk noise Salt-and-Pepper pada densitas {density_pct:.0f}%, disarankan
menggunakan **{best_label}** karena menghasilkan PSNR tertinggi
({best['psnr']:.2f} dB) dengan MSE terendah ({best['mse']:.2f}).

*Referensi: R.C. Gonzalez & R.E. Woods, "Digital Image Processing", 4th Ed.*
"""
    return explanation


def generate_explanation(results_data, api_key):
    """
    Generate penjelasan hasil eksperimen.
    Menggunakan Gemini API jika tersedia, fallback jika tidak.

    Returns:
        tuple: (explanation_text, source) where source is 'gemini' or 'fallback'
    """
    if not api_key or not api_key.strip():
        return _get_fallback_explanation(results_data), "fallback"

    try:
        genai.configure(api_key=api_key.strip())
        prompt = _build_prompt(results_data)
        
        last_error = None
        for model_name in GEMINI_MODELS:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt, request_options={"timeout": 30})
                return response.text, "gemini"
            except Exception as e:
                print(f"[Gemini Error with {model_name}] {e}")
                last_error = e
                
        print(f"[Gemini All Models Failed] Last error: {last_error}")
        with open("gemini_error.txt", "w", encoding="utf-8") as f:
            f.write(f"All models failed. Last error: {str(last_error)}")
        return _get_fallback_explanation(results_data), "fallback"
    except Exception as e:
        print(f"[Gemini Configuration/Execution Error] {e}")
        with open("gemini_error.txt", "w", encoding="utf-8") as f:
            f.write(f"Configuration/Execution error: {str(e)}")
        return _get_fallback_explanation(results_data), "fallback"



def _get_offline_followup_answer(question, results_data):
    """Memberikan jawaban cerdas secara lokal jika Gemini tidak tersedia."""
    q = question.lower()
    d = results_data
    best = d["best"]
    best_label = f"{best['filter_type']} {best['kernel_size']}x{best['kernel_size']}"
    
    # 1. Gaussian Filter
    if any(k in q for k in ["gaussian", "gaus", "rata-rata", "average"]):
        return f"""**Filter Gaussian** kurang efektif untuk noise Salt-and-Pepper karena filter ini merupakan filter linier yang bekerja dengan menghitung rata-rata tertimbang (*weighted average*) dari nilai piksel di dalam kernel. 

Karena noise Salt-and-Pepper memiliki nilai ekstrem (0 untuk hitam/pepper dan 255 untuk putih/salt), nilai-nilai ekstrem ini akan ikut dihitung dalam proses rata-rata. Akibatnya, noise tidak tereduksi melainkan justru menyebar ke piksel tetangga terdekat, menciptakan efek kekaburan (*blurring*) di sekitar noise."""

    # 2. Median Filter
    if any(k in q for k in ["median", "nilai tengah", "lebih baik", "efektif"]):
        return f"""**Filter Median** sangat efektif untuk mereduksi noise Salt-and-Pepper karena merupakan penapis statistik non-linier (*order-statistic filter*). 

Filter ini bekerja dengan mengurutkan nilai piksel di dalam kernel dan memilih nilai tengahnya. Karena noise Salt-and-Pepper bernilai ekstrem (0 atau 255), nilai noise tersebut akan selalu berada di ujung kiri atau kanan setelah diurutkan. Dengan demikian, nilai median yang terpilih hampir pasti adalah nilai piksel asli yang bersih, sehingga noise tereliminasi sepenuhnya tanpa merusak ketajaman tepi citra secara berlebihan."""

    # 3. Kernel Size
    if any(k in q for k in ["kernel", "ukuran", "3x3", "5x5", "dimensi"]):
        return f"""Ukuran kernel mempengaruhi keseimbangan antara kemampuan mereduksi noise dan ketajaman citra:
        
- **Kernel 3x3**: Lebih baik dalam menjaga detail halus dan ketajaman citra, namun kurang efektif jika densitas noise cukup tinggi karena jumlah piksel bersih di dalam kernel terlalu sedikit.
- **Kernel 5x5**: Lebih kuat dalam mereduksi noise tinggi karena memiliki area referensi lebih luas (25 piksel), tetapi memiliki efek samping membuat citra hasil filter terlihat lebih kabur (*blur*) karena detail halus ikut tereliminasi.

Pada eksperimen ini, dengan densitas noise {d['noise_density']*100:.0f}%, filter terbaik adalah **{best_label}**."""

    # 4. PSNR & MSE
    if any(k in q for k in ["psnr", "mse", "metrik", "db", "pengukuran"]):
        return f"""Metrik **PSNR** dan **MSE** digunakan untuk mengukur kualitas pemulihan citra:

- **MSE (Mean Squared Error)**: Mengukur rata-rata kuadrat selisih antara citra asli dan citra hasil pemulihan. Semakin kecil nilai MSE (mendekati 0), semakin sedikit perbedaan antara citra hasil dengan citra asli.
- **PSNR (Peak Signal-to-Noise Ratio)**: Dinyatakan dalam desibel (dB), menunjukkan perbandingan antara kekuatan sinyal maksimum citra terhadap derau yang merusaknya. Semakin besar nilai PSNR (biasanya > 30 dB dianggap sangat baik), semakin tinggi kualitas visual citra hasil pemulihan."""

    # 5. Salt-and-Pepper Noise
    if any(k in q for k in ["salt", "pepper", "garam", "merica", "derau"]):
        return f"""**Noise Salt-and-Pepper** (derau impuls) adalah bentuk kerusakan citra yang ditandai dengan munculnya piksel-piksel hitam (nilai 0 - *pepper*) dan putih (nilai 255 - *salt*) secara acak. 

Noise ini sering terjadi karena adanya kesalahan transmisi data digital, gangguan pada sensor kamera, atau masalah penyimpanan. Karakteristiknya yang ekstrem (impulsif) membuat filter linier seperti Gaussian gagal mengatasinya, sedangkan filter non-linier seperti Median sangat ideal."""

    # 6. Gonzalez / Woods / Teori
    if any(k in q for k in ["gonzalez", "woods", "teori", "buku", "literatur"]):
        return f"""Menurut teori pengolahan citra dari **R.C. Gonzalez & R.E. Woods**, filter median tergolong sebagai penapis statistik peringkat (*order-statistic filter*). 

Teori menyatakan bahwa untuk noise yang bersifat impulsif (seperti Salt-and-Pepper), filter median memberikan kemampuan reduksi derau yang jauh lebih unggul dengan distorsi visual yang jauh lebih minimal dibandingkan filter rata-rata linier (seperti filter Gaussian) pada ukuran kernel yang setara."""

    # 7. Cara Kerja / Algoritma
    if any(k in q for k in ["cara kerja", "bagaimana", "proses", "langkah", "algoritma"]):
        return f"""**Langkah-langkah algoritma Filter Median**:
1. Tempatkan kernel (misal 3x3) pada posisi piksel target.
2. Ambil semua nilai intensitas piksel tetangga yang tercakup oleh kernel.
3. Urutkan nilai-nilai piksel tersebut dari yang terkecil hingga terbesar.
4. Pilih nilai yang berada tepat di tengah urutan (nilai median).
5. Ganti nilai piksel target dengan nilai median tersebut.
6. Geser kernel ke piksel berikutnya dan ulangi proses hingga seluruh citra terproses."""

    # Default fallback
    return f"""Berdasarkan data eksperimen dengan densitas noise **{d['noise_density']*100:.0f}%**, filter terbaik adalah **{best_label}** dengan PSNR **{best['psnr']:.2f} dB** dan MSE **{best['mse']:.2f}**. 

Secara umum, filter Median selalu mengungguli Gaussian untuk noise impulsif Salt-and-Pepper karena filter Median mampu mengeliminasi nilai ekstrem (0 dan 255) secara tuntas melalui pengurutan nilai tengah, sedangkan Gaussian cenderung merata-ratakan dan menyebarkan derau tersebut."""


def ask_followup(question, results_data, api_key):
    """
    Tanya pertanyaan lanjutan tentang hasil eksperimen.

    Returns:
        tuple: (answer_text, source)
    """
    if not api_key or not api_key.strip():
        return _get_offline_followup_answer(question, results_data), "fallback"

    d = results_data
    density_pct = d["noise_density"] * 100
    best = d["best"]
    best_label = f"{best['filter_type']} {best['kernel_size']}x{best['kernel_size']}"

    filter_rows = ""
    for f in d["filters"]:
        label = f"{f['filter_type']} {f['kernel_size']}x{f['kernel_size']}"
        filter_rows += f"  - {label}: PSNR={f['psnr']:.2f} dB, MSE={f['mse']:.2f}\n"

    prompt = f"""Kamu adalah dosen Pengolahan Citra Digital.
Konteks eksperimen:
- Noise Salt-and-Pepper densitas {density_pct:.0f}%
- Hasil filtering:
{filter_rows}
- Filter terbaik: {best_label}

Pertanyaan mahasiswa: {question}

Jawab dalam Bahasa Indonesia, singkat dan jelas (maks 200 kata)."""

    try:
        genai.configure(api_key=api_key.strip())
        
        last_error = None
        for model_name in GEMINI_MODELS:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt, request_options={"timeout": 30})
                return response.text, "gemini"
            except Exception as e:
                print(f"[Gemini Error in ask_followup with {model_name}] {e}")
                last_error = e

        print(f"[Gemini Error in ask_followup] Fallback to offline answer due to: {last_error}")
        return _get_offline_followup_answer(question, results_data), "fallback"
    except Exception as e:
        print(f"[Gemini Configuration/Execution Error in ask_followup] Fallback to offline answer due to: {e}")
        return _get_offline_followup_answer(question, results_data), "fallback"

