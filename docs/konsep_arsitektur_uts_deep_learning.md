# Prompt Konsep Arsitektur dan Rancangan UTS Deep Learning v2

Gunakan project folder **supervised deep learning** yang sudah tersedia. Fokus tugas ini adalah menyusun **rancangan konseptual arsitektur notebook dengan source code**, agar implementasi nantinya konsisten dengan seluruh arahan soal UTS, struktur folder project, serta output yang sudah/telah disiapkan.

## Tujuan Utama

Bangun sebuah notebook utama bernama **Ball Trajectory Prediction.ipynb** yang berfungsi sebagai pipeline end-to-end untuk tugas UTS Deep Learning tentang **prediksi trajektori bola basket** berdasarkan koordinat frame sebelumnya. Notebook ini harus memuat penjelasan markdown akademik, narasi teknis, dan source code pada setiap bagian, sehingga sekaligus dapat menjadi dasar laporan analitis.

## Konteks Soal

Tugas UTS meminta pembangunan sistem supervised deep learning untuk:
- mengunduh dan menyiapkan video sumber,
- memotong video menjadi 3 menit dari bagian tengah,
- mendeteksi dan melacak tiga objek bola basket pada video 3 menit,
- mengekstrak koordinat titik tengah setiap bola beserta ID uniknya,
- membagi data secara kronologis menjadi train dan test,
- membentuk data sekuensial dengan sliding window,
- melatih tiga model: Simple RNN, LSTM, dan GRU,
- mengevaluasi proses training melalui loss curve,
- membandingkan performa prediksi trajektori menggunakan metrik MSE/RMSE dan visualisasi lintasan.

## Struktur Folder yang Harus Dipatuhi

Gunakan path yang sudah tersedia berikut ini secara konsisten:

- `src/generate_notebook.py` → script generator notebook
- `notebooks/Ball Trajectory Prediction.ipynb` → notebook utama hasil generate
- `docs/3 Basketball Bouncing.mp4` → video sumber awal
- `docs/3 Basketball Bouncing_3min.mp4` → video 3 menit untuk proses tracking
- `docs/Soal UTS Deep Learning(B).pdf` → dokumen referensi soal
- `data/raw/dataset_koordinat_bola.csv` → dataset koordinat hasil tracking / input data tabular utama
- `data/processed/tensors/` → hasil preprocessing tensor (`X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`)
- `models/` → penyimpanan model terlatih (`Simple_RNN.pth`, `LSTM.pth`, `GRU.pth`, `scaler.pkl`)
- `images/output/loss_curves.png` → grafik loss training
- `images/output/trajectory_comparison.png` → grafik komparasi trajektori
- `images/output/tracking_demo.mp4` → video hasil tracking
- `images/output/tracking_samples/` → contoh frame hasil tracking
- `logs/` → catatan proses dan eksperimen
- `config/` → konfigurasi hyperparameter, path, dan eksperimen bila diperlukan
- `app/` dan `test/` → biarkan tersedia, tidak menjadi fokus utama notebook

## Prinsip Perancangan Notebook

Notebook harus dirancang sebagai dokumen ilmiah-terstruktur, bukan sekadar tempat eksekusi kode. Setiap bab harus memiliki:
- judul bab sesuai butir soal,
- subbab bernomor seperti `1.1`, `1.2`, dan seterusnya,
- markdown penjelasan tentang tujuan, konsep, dan interpretasi hasil,
- source code pada bagian implementasi,
- komentar konseptual pada bagian kode,
- alur yang mengikuti urutan logis data engineering → modeling → evaluasi.

Gunakan gaya penulisan akademik formal, jelas, sistematis, dan sesuai level Magister Informatika.

## 0. Persiapan Dataset

Sebelum masuk ke Bab 1, notebook harus memiliki satu bagian khusus tentang **Persiapan Dataset** karena instruksi dosen mewajibkan tahapan ini dilakukan terlebih dahulu.

### 0.1 Unduh Video Sumber
Jelaskan bahwa video sumber berasal dari tautan YouTube yang diberikan pada soal. Bagian ini perlu menegaskan bahwa sumber data awal masih berupa video mentah dan belum siap digunakan untuk tracking.

### 0.2 Pemotongan Video Menjadi 3 Menit
Rancang subbagian yang menjelaskan bahwa video harus dipotong menjadi durasi 3 menit dengan mengambil bagian tengah video, yaitu segmen ketika bola terlihat jelas dan aktif bergerak. Ini penting agar data yang diproses memiliki kualitas visual yang cukup baik untuk object tracking.

### 0.3 Verifikasi Video 3 Menit
Notebook perlu menjelaskan bahwa hasil pemotongan disimpan sebagai artefak tersendiri di folder `docs/`, misalnya `docs/3 Basketball Bouncing_3min.mp4`. Bagian ini dapat menampilkan ringkasan durasi, frame rate, dan alasan pemilihan segmen tengah.

### 0.4 Posisi dalam Pipeline
Tekankan bahwa persiapan dataset bukan bagian dari tracking itu sendiri, melainkan tahap pra-pemrosesan awal sebelum Bab 1. Dengan demikian, urutan notebook yang benar adalah:
1. Persiapan Dataset,
2. Ekstraksi & Tracking Objek,
3. Data Engineering CSV Exporter,
4. Train/Test Split,
5. Sliding Window,
6. Arsitektur Model,
7. Evaluasi,
8. Pengujian & Komparasi Trajektori.

## Rancangan Besar Notebook

# 1. Ekstraksi & Tracking Objek

## 1.1 Latar Belakang
Jelaskan bahwa tahap pertama bertujuan memperoleh data koordinat bola dari video menggunakan pendekatan object detection dan multi-object tracking. Tegaskan bahwa inti tahap ini adalah menghasilkan identitas bola yang konsisten antar frame.

## 1.2 Input dan Output Tahap Tracking
Input utama mengarah ke file video `docs/3 Basketball Bouncing_3min.mp4`. Output tahap ini diarahkan ke:
- video hasil tracking di `images/output/tracking_demo.mp4`,
- frame sampling hasil tracking di `images/output/tracking_samples/`,
- data koordinat terstruktur yang nantinya tersimpan sebagai `data/raw/dataset_koordinat_bola.csv`.

## 1.3 Konsep Metode
Jelaskan konsep YOLO tracking (`model.track()`) untuk mendeteksi dan melacak tiga bola secara simultan. Bahas bahwa setiap objek harus memiliki identitas unik agar trajektori masing-masing bola dapat dipelajari sebagai data sekuensial.

## 1.4 Validasi Hasil Tracking
Rancang bagian untuk menjelaskan bagaimana hasil tracking diverifikasi secara visual melalui video annotasi dan beberapa frame contoh. Notebook harus menekankan bahwa kualitas tracking sangat menentukan kualitas dataset downstream.

# 2. Data Engineering – CSV Exporter

## 2.1 Struktur Dataset Koordinat
Jelaskan bahwa hasil tracking perlu ditransformasikan menjadi dataset tabular terintegrasi. Kolom minimum yang disarankan:
- `frame`
- `ball_id`
- `x_center`
- `y_center`
- `confidence` (opsional tetapi direkomendasikan)

## 2.2 Konsolidasi Tiga Bola
Notebook harus menjelaskan bahwa ketiga bola disimpan dalam satu file CSV/DataFrame terpadu, bukan dipisah per bola, agar analisis lebih rapi dan fleksibel.

## 2.3 Penyimpanan Dataset
Arahkan hasil akhir tahap ini ke `data/raw/dataset_koordinat_bola.csv`. Jika file tersebut sudah tersedia, notebook dapat tetap menjelaskan bahwa file tersebut merupakan keluaran tahap tracking/exporter yang menjadi input utama tahap berikutnya.

## 2.4 Pemeriksaan Kualitas Data
Rancang subbab untuk pembahasan:
- missing frame,
- missing coordinate,
- ketidakkonsistenan ID,
- potensi noise tracking,
- distribusi koordinat tiap bola.

# 3. Data Engineering – Train/Test Split

## 3.1 Prinsip Split Kronologis
Notebook harus menegaskan bahwa pembagian data **tidak boleh random**, karena tugas ini merupakan masalah time series / sequence prediction. Data dua menit pertama digunakan sebagai train, dan satu menit terakhir sebagai test.

## 3.2 Strategi Pemotongan Waktu
Jelaskan bahwa unit pemisahan dilakukan secara temporal berdasarkan urutan frame. Rancangan ini harus menjaga kausalitas waktu sehingga model hanya belajar dari masa lalu untuk memprediksi masa depan.

## 3.3 Normalisasi
Sertakan bagian konsep bahwa koordinat perlu dinormalisasi sebelum masuk model RNN/LSTM/GRU. Notebook harus menjelaskan bahwa scaler di-fit menggunakan data train, lalu diaplikasikan ke data test untuk menghindari data leakage. Artefak scaler disimpan ke `models/scaler.pkl`.

# 4. Data Engineering – Sliding Window

## 4.1 Tujuan Transformasi Sekuensial
Jelaskan bahwa model recurrent tidak menerima data mentah per baris, melainkan urutan observasi masa lalu. Karena itu, dataset harus diubah menjadi pasangan input-output berbentuk sequence.

## 4.2 Desain Look-back Window
Rancang agar notebook menjelaskan pilihan look-back window, misalnya 10, 20, atau nilai lain yang argumentatif. Rekomendasi konseptual: gunakan satu nilai utama yang dijustifikasi, misalnya 20 frame ke belakang untuk memprediksi 1 frame ke depan.

## 4.3 Bentuk Tensor
Notebook harus menjelaskan interpretasi dimensi tensor akhir:
- `X_train`: `(jumlah_sampel, window_size, jumlah_fitur)`
- `y_train`: `(jumlah_sampel, jumlah_target)`
- `X_test`, `y_test` dengan struktur serupa

Semua keluaran tensor diarahkan ke folder `data/processed/tensors/` dengan nama:
- `X_train.npy`
- `X_test.npy`
- `y_train.npy`
- `y_test.npy`

## 4.4 Catatan Penting
Jelaskan bahwa sliding window harus mempertahankan urutan temporal dan idealnya memproses trajektori per `ball_id`, agar tidak mencampur urutan antar objek secara keliru.

# 5. Arsitektur Model

## 5.1 Tujuan Perbandingan Model
Notebook harus menjelaskan bahwa eksperimen bertujuan membandingkan tiga arsitektur supervised deep learning berbasis sequence modeling:
- Simple RNN
- LSTM
- GRU

## 5.2 Rancangan Input Model
Setiap model menerima input tensor urutan koordinat hasil sliding window. Fitur utama adalah koordinat posisi, yaitu `x_center` dan `y_center`.

## 5.3 Rancangan Model Simple RNN
Subbab ini harus menjelaskan konsep dasar Simple RNN sebagai baseline. Tekankan kelebihan kesederhanaan arsitektur dan kelemahannya pada dependensi temporal panjang.

## 5.4 Rancangan Model LSTM
Subbab ini harus menjelaskan bahwa LSTM dirancang untuk menangani long-term dependency melalui mekanisme gate. Notebook perlu membahas alasan teoretis mengapa LSTM cocok untuk trajektori sekuensial.

## 5.5 Rancangan Model GRU
Subbab ini harus menjelaskan bahwa GRU merupakan penyederhanaan LSTM dengan jumlah parameter lebih ringkas, tetapi sering tetap kompetitif pada tugas sequence regression.

## 5.6 Strategi Training
Notebook harus merancang pembahasan mencakup:
- loss function regresi (MSE),
- metrik evaluasi tambahan (RMSE),
- optimizer (misalnya Adam),
- epoch, batch size, validation split,
- early stopping / checkpoint bila digunakan.

## 5.7 Penyimpanan Model
Semua model hasil training diarahkan ke folder `models/` dengan nama konsisten:
- `models/Simple_RNN.pth`
- `models/LSTM.pth`
- `models/GRU.pth`

# 6. Evaluasi Proses Training

## 6.1 Loss Curve
Notebook harus memiliki bagian yang menampilkan pergerakan training loss dan validation loss untuk ketiga model. Output visual diarahkan ke `images/output/loss_curves.png`.

## 6.2 Interpretasi Kurva
Rancang penjelasan untuk menilai apakah model:
- belajar dengan baik,
- mengalami underfitting,
- mengalami overfitting,
- atau memerlukan penyesuaian hyperparameter.

Penjelasan harus bersifat analitis, bukan hanya deskriptif.

# 7. Pengujian & Komparasi Trajektori

## 7.1 Evaluasi pada Data Uji
Notebook harus menguji ketiga model pada test set hasil split kronologis. Gunakan metrik utama:
- MSE
- RMSE

## 7.2 Tabel Komparasi
Rancang satu tabel komparatif yang menampilkan performa akhir ketiga model. Fokus utama adalah menunjukkan model mana yang paling baik dalam memprediksi koordinat bola.

## 7.3 Visualisasi Ground Truth vs Prediksi
Notebook harus memvisualisasikan trajektori asli dan hasil prediksi dalam plot 2D untuk **satu ID bola saja** (misalnya `Ball_1`). Plot diarahkan ke `images/output/trajectory_comparison.png`.

## 7.4 Analisis Akhir
Rancang bagian kesimpulan analitis yang menjawab:
- model mana yang terbaik,
- mengapa model tersebut unggul,
- bagaimana karakteristik arsitektur berpengaruh pada prediksi trajektori,
- serta keterbatasan eksperimen.

## Rancangan Isi Markdown per Bab

Pada setiap bab, notebook harus memiliki markdown dengan pola berikut:
- **Tujuan bagian**
- **Konsep inti**
- **Input dan output bagian**
- **Alasan metodologis**
- **Interpretasi hasil**
- **Catatan risiko / kendala**

Dengan pola ini, notebook tidak hanya menjadi file eksekusi, tetapi juga menjadi dokumen naratif akademik yang siap mendukung laporan PDF akhir.

## Rancangan Khusus untuk `src/generate_notebook.py`

File `src/generate_notebook.py` berfungsi sebagai **pembangun notebook otomatis**. Secara konseptual, file ini harus dirancang untuk:
- membentuk notebook dari daftar markdown cell dan source code cell,
- menyusun bab dan subbab sesuai nomor soal,
- memastikan seluruh path output sesuai struktur folder project,
- menghasilkan file akhir ke `notebooks/Ball Trajectory Prediction.ipynb`,
- menjaga konsistensi urutan pipeline dari tracking sampai evaluasi.

Secara konseptual, `generate_notebook.py` sebaiknya memiliki komponen berikut:
- definisi metadata notebook,
- generator markdown cell untuk setiap bab,
- generator source code cell untuk setiap pipeline,
- penyusun urutan cell,
- writer/exporter notebook ke folder `notebooks/`.

## Penyesuaian dengan Kondisi Folder Saat Ini (Reset)

Berdasarkan struktur folder terbaru yang baru saja di-reset dan dibersihkan dari artefak lama, kondisi saat ini adalah **bersih (kosong)** pada direktori output. Oleh karena itu, rancangan notebook harus:
- **Mengeksekusi ulang seluruh pipeline dari awal**, termasuk ekstraksi koordinat untuk menghasilkan `data/raw/dataset_koordinat_bola.csv`;
- Membangun dan mengekspor tensor ulang ke dalam `data/processed/tensors/`;
- Melatih model dari awal (dari nol) karena direktori `models/` dalam keadaan kosong, lalu menyimpan bobot `Simple_RNN.pth`, `LSTM.pth`, dan `GRU.pth` beserta `scaler.pkl` setelah training selesai;
- Memproduksi ulang seluruh visualisasi baik itu *tracking demo*, *loss curve*, maupun plot *trajectory comparison* dan menyimpannya ke `images/output/`.

Notebook harus benar-benar ditulis dan dieksekusi secara **End-to-End reproducible pipeline**, di mana setiap sel yang dijalankan akan mengisi kembali kekosongan pada struktur folder yang telah disiapkan.

## Gaya Penulisan yang Diinginkan

Gunakan gaya bahasa Indonesia akademik formal. Penjelasan harus:
- sistematis,
- kritis,
- menjelaskan alasan pemilihan metode,
- menjelaskan hubungan antar tahap pipeline,
- menonjolkan pemahaman supervised deep learning,
- serta relevan untuk kebutuhan laporan UTS tingkat magister.

## Checklist Final Rancangan

Pastikan notebook hasil rancangan nantinya memenuhi semua poin berikut:
- mengikuti urutan soal 1 sampai 7,
- persiapan dataset muncul sebelum Bab 1,
- setiap soal diubah menjadi bab dan subbab bernomor,
- memiliki markdown penjelasan di setiap bagian,
- menyertakan source code pada bagian implementasi,
- menggunakan path sesuai folder project yang tersedia,
- mengarahkan semua output ke folder yang tepat,
- mendukung reproduksibilitas pipeline,
- memisahkan jelas tahap tracking, data engineering, modeling, evaluasi, dan analisis,
- siap dijadikan dasar penyusunan laporan PDF akhir.

## Instruksi Akhir

Bangun notebook dengan orientasi **reproducible research pipeline**. Fokus bukan hanya agar program berjalan, tetapi agar struktur notebook memperlihatkan pemahaman konseptual, metodologis, dan eksperimen supervised deep learning secara utuh. Seluruh narasi harus menguatkan jawaban atas soal UTS, bukan sekadar menampilkan hasil.