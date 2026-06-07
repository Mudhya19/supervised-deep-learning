# 🏀 Prediksi Trajektori Bola Basket (Deep Learning)

Proyek ini dibangun untuk memenuhi tugas Ujian Tengah Semester (UTS) mata kuliah Deep Learning. Fokus utama dari proyek ini adalah membangun *pipeline end-to-end* yang memanfaatkan arsitektur jaringan saraf berulang (RNN, LSTM, dan GRU) untuk memprediksi lintasan trajektori bola basket di masa depan berdasarkan rekaman koordinat pergerakan bola di masa lalu.

## 🌟 Fitur Utama

1. **Object Tracking Otomatis**: Menggunakan **YOLOv8** dan algoritma pelacakan BoT-SORT untuk mendeteksi dan melacak tiga bola basket secara presisi dari rekaman video asli.
2. **Ekstraksi & Transformasi Data**: Membangun dataset koordinat (x, y) spasio-temporal dari hasil pelacakan, serta memprosesnya menggunakan metode *Sliding Window* untuk menghasilkan deret waktu (*time-series sequences*).
3. **Komparasi Model Deep Learning**: Melatih dan membandingkan performa tiga arsitektur *Recurrent Neural Network* yang berbeda:
   - **Simple RNN**
   - **LSTM** (Long Short-Term Memory)
   - **GRU** (Gated Recurrent Unit)
4. **Visualisasi Komprehensif**: Menampilkan perbandingan pergerakan bola hasil prediksi versus aktual, lengkap dengan grafik kurva evaluasi *Loss* (*Training* dan *Validation*).

---

## 📂 Struktur Repositori

Proyek ini diorganisasikan dengan standar yang rapi dan terstruktur agar mempermudah proses reproduksi dan eksperimen:

```text
supervised_deep_learning/
├── app/                  # (Opsional) Kode untuk deployment aplikasi 
├── config/               # File konfigurasi hyperparameter dan arsitektur (jika diperlukan)
├── data/
│   ├── processed/        # Tensor yang telah diolah (X_train.npy, y_train.npy, dsb.)
│   └── raw/              # Dataset mentah berupa CSV dari hasil tracking YOLO
├── docs/                 # Dokumen soal UTS, referensi, serta video mentah (.mp4)
├── images/
│   └── output/           # Gambar grafik loss, komparasi trajektori, dan video hasil tracking
├── logs/                 # Catatan proses eksperimen model
├── models/               # Model PyTorch terlatih (.pth) dan objek Scaler (.pkl)
├── notebooks/            # Jupyter Notebook utama tempat pipeline dieksekusi
├── src/                  # Kumpulan skrip Python, termasuk generator notebook
├── test/                 # Skrip pengujian kode
└── README.md             # Dokumentasi utama proyek
```

---

## ⚙️ Persyaratan Sistem & Instalasi

Pastikan sistem Anda telah memiliki instalasi Python 3.8+ (disarankan 3.10+). Eksekusi perintah berikut untuk mempersiapkan lingkungan:

```bash
# 1. Buat virtual environment
python -m venv .venv

# 2. Aktivasi environment (Windows)
.\.venv\Scripts\activate

# 3. Instal dependensi utama (YOLO, PyTorch, dsb.)
pip install -r requirements.txt
# (Catatan: yt-dlp dan pustaka lain yang dibutuhkan otomatis akan ditarik jika menggunakan skrip generator)
```

---

## 🚀 Cara Menjalankan Pipeline

Proyek ini didesain agar sangat interaktif menggunakan arsitektur Jupyter Notebook tunggal yang merangkum seluruh tahapan eksperimen.

### 1. Menghasilkan Notebook (Generate)
Jika Anda ingin mereset atau membangun ulang kerangka file Jupyter Notebook beserta kode sumber terbarunya, jalankan generator yang kami siapkan:

```bash
python src/generate_notebook.py
```
Perintah ini akan secara otomatis membuat/memperbarui file `notebooks/Ball Trajectory Prediction.ipynb`.

### 2. Mengeksekusi Eksperimen
Buka notebook yang telah di-*generate* menggunakan Jupyter Notebook atau JupyterLab:

```bash
jupyter notebook "notebooks/Ball Trajectory Prediction.ipynb"
```

Ikuti eksekusi seluler secara berurutan sesuai Bab yang telah dirancang:
1. **Tahap 0: Persiapan Dataset** - Unduh video dari YouTube dan potong menjadi sampel 3 menit di bagian tengah.
2. **Tahap 1: Ekstraksi & Tracking** - Lacak bola menggunakan YOLO dan validasi hasilnya.
3. **Tahap 2 & 3: Data Engineering** - Konversi hasil ke CSV dan lakukan pembagian *Train/Test Split* secara kronologis.
4. **Tahap 4: Sliding Window** - Transformasi ke bentuk tensor 3D untuk *sequence modeling*.
5. **Tahap 5: Arsitektur Model** - Bangun dan latih Simple RNN, LSTM, serta GRU.
6. **Tahap 6 & 7: Evaluasi & Visualisasi** - Hitung skor *loss* gabungan dan gambarkan prediksi trajektori pada ruang 2D.

---

## 📊 Hasil Evaluasi Singkat

Model diuji kemampuannya untuk mengantisipasi letak koordinat `X_center` dan `Y_center` bola pada urutan *frame* yang belum pernah dilihat sebelumnya (1 menit terakhir video). Secara umum, model dengan memori jangka panjang seperti **LSTM dan GRU menghasilkan kurva prediksi yang lebih mulus dan akurat** dibandingkan Simple RNN dalam menangani dinamika trajektori lambungan bola basket.

## 📝 Referensi Akademik
* Ultralytics YOLOv8 Documentation
* PyTorch Recurrent Neural Networks Documentation
* Konsep UTS Deep Learning Universitas
