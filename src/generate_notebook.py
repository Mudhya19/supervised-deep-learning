import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3"
}

cells = []

# CELL 0: JUDUL & SETUP
cells.append(nbf.v4.new_markdown_cell("""\
# 🏀 Ball Trajectory Prediction using Deep Learning
## Ujian Tengah Semester (UTS) - Deep Learning (Kelas B)

**Deskripsi Tugas:**  
Membangun model Deep Learning berbasis Supervised Learning untuk memprediksi trajektori bola basket.

**Fase 1: Persiapan Dataset & Ekstraksi Tracking**
Pada fase awal ini, kita akan:
1. Mengiris (Crop) video mentah menjadi berdurasi ~6000 frame (sekitar 3 menit 20 detik) pada bagian aktivitas tertinggi.
2. Mendeteksi dan melacak pergerakan objek bola secara simultan menggunakan YOLOv8.
"""))

# CELL 1: IMPORT LIBRARY & VIDEO CROPPER
cells.append(nbf.v4.new_code_cell("""\
import cv2
import pandas as pd
import numpy as np
import os
import shutil
import warnings
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from collections import deque
from ultralytics import YOLO

warnings.filterwarnings('ignore')

# Memastikan Working Directory
if os.path.basename(os.path.abspath('.')) == 'notebooks':
    os.chdir('..')

# Membuat direktori yang dibutuhkan
os.makedirs('data/raw', exist_ok=True)
os.makedirs('images/output/tracking_samples', exist_ok=True)

print("✅ Setup Lingkungan Selesai.")

# ============================================
# PRA-PEMROSESAN: PEMOTONGAN VIDEO 6000 FRAME
# ============================================
raw_video_path = 'docs/3 Basketball Bouncing.mp4'
cropped_video_path = 'docs/3 Basketball Bouncing_3min.mp4'

# Kita buat selalu regenerate jika belum ada, atau jika frame-nya belum 6000
need_crop = True
if os.path.exists(cropped_video_path):
    cap_check = cv2.VideoCapture(cropped_video_path)
    if int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT)) >= 6000:
        need_crop = False
    cap_check.release()

if need_crop:
    print("⏳ Memotong video menjadi durasi 6000 Frame (Mulai dari Menit 08:30)...")
    cap = cv2.VideoCapture(raw_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    start_frame = int(8.5 * 60 * fps) # Menit 08:30
    end_frame   = start_frame + 6000  # Total 6000 frame (~3 menit 20 detik)
    
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(cropped_video_path, fourcc, fps, (width, height))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    
    while cap.isOpened() and current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret: break
        out.write(frame)
        current_frame += 1
        
    cap.release()
    out.release()
    print("✅ Video berhasil dipotong menjadi 6000 Frame!")
else:
    print("✅ Video berdurasi 6000 Frame sudah siap digunakan!")
"""))

# CELL 2: BAGIAN 1 MARKDOWN
cells.append(nbf.v4.new_markdown_cell("""\
---
## 📌 Bagian 1 — Ekstraksi & Tracking Objek

### 1.1 YOLO TRACKING — Deteksi & Lacak 3 Bola Basket
Untuk mengekstraksi titik kordinat, kami mendeploy **YOLOv8** dengan algoritma *Multi-Object Tracking* **BoT-SORT** untuk melacak hingga **3 bola simultan** (dan objek keempat jika masuk *frame*).
* **Targeting (`classes=[32]`)**: Memblokir noise, fokus pada bola olahraga.
* **Persistent Identity (`persist=True`)**: Mencegah ID tertukar saat bola bersilangan.
* **Trajectory Tailing**: Menggambar "ekor lintasan" grafik gerak bola untuk validasi Trajektori.
"""))

# CELL 3: SOAL 1 CODE (TRACKING)
cells.append(nbf.v4.new_code_cell("""\
# ============================================
# IMPLEMENTASI 1.1: ADVANCED YOLO TRACKING
# ============================================

model_yolo = YOLO('yolov8n.pt')
video_path = 'docs/3 Basketball Bouncing_3min.mp4'

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

tracking_data = []
confidence_scores = []
frame_id = 0

# Bersihkan direktori tracking_samples sebelumnya agar tidak bentrok dengan sisa file lama
for f in os.listdir('images/output/tracking_samples'):
    os.remove(os.path.join('images/output/tracking_samples', f))

# Target frame untuk diambil sebagai sample (mengikuti contoh 4 gambar)
target_frames = [100, 1350, 2700, 4050]

# Memori Dinamis Ekor Trajektori
trajectories = {} 
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)] # 4 Warna

print("⏳ Memulai eksekusi Computer Vision Tracking...")

with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Deteksi dengan YOLO
        results = model_yolo.track(frame, persist=True, classes=[32], tracker="botsort.yaml", conf=0.3, iou=0.5, verbose=False)
        annotated_frame = results[0].plot()

        num_balls = len(results[0].boxes) if results[0].boxes is not None else 0
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            for box, track_id, conf in zip(boxes, track_ids, confs):
                t_id = int(track_id)
                confidence_scores.append(float(conf))
                
                center_x = int((box[0] + box[2]) / 2.0)
                center_y = int((box[1] + box[3]) / 2.0)
                
                if t_id not in trajectories:
                    trajectories[t_id] = deque(maxlen=30)
                trajectories[t_id].append((center_x, center_y))
                
                # Menggambar ekor
                color = colors[(t_id-1) % len(colors)]
                pts = list(trajectories[t_id])
                for i in range(1, len(pts)):
                    thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
                    cv2.line(annotated_frame, pts[i - 1], pts[i], color, thickness)
                
                tracking_data.append({
                    'Frame_ID': frame_id, 
                    'Ball_ID': t_id,
                    'Center_X': center_x,
                    'Center_Y': center_y,
                    'BBox_X1': int(box[0]),
                    'BBox_Y1': int(box[1]),
                    'BBox_X2': int(box[2]),
                    'BBox_Y2': int(box[3]),
                    'Confidence': float(conf)
                })

        # --- SMART FRAME CAPTURE (Menangkap Frame Spesifik) ---
        if frame_id in target_frames:
            # Gunakan format padding angka nol agar file berurutan
            out_path = f'images/output/tracking_samples/frame_{frame_id:04d}_balls_{num_balls}.jpg'
            cv2.imwrite(out_path, annotated_frame) 
            tqdm.write(f"   📸 [SMART CAPTURE] Frame {frame_id} ditangkap dengan {num_balls} bola terdeteksi!")

        frame_id += 1
        pbar.update(1)

cap.release()

# Ekspor Dataset CSV untuk keperluan training Deep Learning (Soal 2 & 3)
csv_path = 'data/raw/dataset_koordinat_bola.csv'
df_tracking = pd.DataFrame(tracking_data)
df_tracking.to_csv(csv_path, index=False)

print(f"\\n✅ Dataset koordinat tersimpan di: {csv_path}")
print(f"✅ Tahap Tracking (Bagian 1.1) Selesai!")
"""))

# CELL 4: MARKDOWN ANALYSIS
cells.append(nbf.v4.new_markdown_cell("""\
### 1.2 Analisis Hasil Tracking
Validasi statistik untuk mengukur tingkat kepercayaan (*confidence level*) model dalam membedakan identitas bola.
"""))

# CELL 5: CODE ANALYSIS
cells.append(nbf.v4.new_code_cell("""\
# ============================================
# IMPLEMENTASI 1.2: ANALISIS STATISTIK
# ============================================

total_detections = len(tracking_data)
unique_ids = sorted(list(set([d['Ball_ID'] for d in tracking_data])))
avg_conf = np.mean(confidence_scores) * 100 if confidence_scores else 0

print("📊 ANALISIS STATISTIK TRACKING YOLOv8:")
print(f"  ▪ Total Titik Deteksi     : {total_detections:,} bounding boxes")
print(f"  ▪ Rata-Rata Confidence    : {avg_conf:.2f}%")
print(f"  ▪ ID Bola Terdaftar       : {unique_ids} (Sebanyak {len(unique_ids)} Objek)")
"""))

# CELL 6: MARKDOWN VISUALIZATION
cells.append(nbf.v4.new_markdown_cell("""\
### 1.3 Visualisasi: Sample Frame dengan Tracking
Merupakan bukti empiris (visual) penangkapan frame secara spesifik. Keempat gambar tersebut disatukan secara simetris ke dalam sebuah *grid* visualisasi.
"""))

# CELL 7: CODE VISUALIZATION
cells.append(nbf.v4.new_code_cell("""\
# ============================================
# IMPLEMENTASI 1.3: VISUALISASI 4 GAMBAR DISATUKAN
# ============================================
sample_dir = 'images/output/tracking_samples'
if os.path.exists(sample_dir):
    sample_files = sorted([f for f in os.listdir(sample_dir) if f.startswith('frame_') and 'balls' in f])
    
    if len(sample_files) > 0:
        print("📷 Visualisasi 4 Gambar Sample Bukti (Disatukan):")
        # Membuat grid 2x2 untuk menampung maksimal 4 gambar
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Sample Frame dengan YOLO Tracking', fontsize=18, fontweight='bold')
        axes = axes.flatten()
        
        for idx, file in enumerate(sample_files):
            if idx >= 4: break # Batasi maksimal 4 grid
            
            img_rgb = cv2.cvtColor(cv2.imread(os.path.join(sample_dir, file)), cv2.COLOR_BGR2RGB)
            axes[idx].imshow(img_rgb)
            axes[idx].axis('off')
            
            # Parsing nama file: frame_0100_balls_1.jpg
            # string.split('_') -> ['frame', '0100', 'balls', '1.jpg']
            parts = file.split("_")
            f_id = int(parts[1])
            b_count = parts[3].split(".")[0]
            
            axes[idx].set_title(f'Frame {f_id} — {b_count} bola terdeteksi', fontweight='bold', fontsize=14)
        
        # Kosongkan sisa axis jika objek terdeteksi < 4
        for i in range(len(sample_files), 4):
            axes[i].axis('off')
            
        plt.subplots_adjust(hspace=0.3, wspace=0.05, top=0.92)
        plt.show()
    else:
        print("Belum ada sample frame yang berhasil ditangkap.")
"""))

cells.append(nbf.v4.new_markdown_cell("""\
### Bagian 2 — Data Engineering: CSV Exporter
Tahap ini bertujuan untuk mengekstrak titik tengah koordinat (X, Y) dari hasil tracking beserta ID uniknya, lalu menyimpannya dalam satu Dataframe terpadu sesuai mandat soal nomor 2.

**Inspeksi Data Mentah (Raw Data):** Memastikan tabel berisikan kolom esensial seperti `Frame_ID`, `Ball_ID`, `Center_X`, dan `Center_Y`.
"""))

cells.append(nbf.v4.new_code_cell("""\
# ============================================
# IMPLEMENTASI SOAL 2: CSV EXPORTER
# ============================================
import pandas as pd
import numpy as np

csv_path = 'data/raw/dataset_koordinat_bola.csv'
try:
    df_tracking = pd.read_csv(csv_path)
    print(f"✅ Berhasil memuat dataset: {csv_path}")
    print(f"Total Baris Data: {len(df_tracking):,} records")
    display(df_tracking.head())
except Exception as e:
    print(f"⚠️ Gagal memuat dataset, pastikan sel pelacakan sebelumnya sudah dijalankan! Error: {e}")
"""))

cells.append(nbf.v4.new_markdown_cell("""\
### Bagian 3 — Data Engineering: Train/Test Split
Sesuai soal nomor 3, dataset sekuensial **tidak boleh** dibagi secara acak (random split) untuk mencegah *data leakage* antar waktu. Pembagian dilakukan secara **kronologis temporal**:
- **Data Latih (Train)**: 66.6% frame awal (Siklus pergerakan masa lalu)
- **Data Uji (Test)**: 33.3% frame akhir (Siklus pergerakan masa depan)

Serta penerapan normalisasi spasial menggunakan `MinMaxScaler`.
"""))

cells.append(nbf.v4.new_code_cell("""\
# ============================================
# IMPLEMENTASI SOAL 3: TRAIN/TEST SPLIT KRONOLOGIS
# ============================================
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

os.makedirs('models', exist_ok=True)

# 1. Pastikan terurut berdasarkan Ball_ID dan Kronologi Waktu (Frame_ID)
df_tracking = df_tracking.sort_values(by=['Ball_ID', 'Frame_ID']).reset_index(drop=True)

# 2. Split 66.6% Train, 33.3% Test
train_ratio = 0.666
split_idx = int(len(df_tracking) * train_ratio)

train_df = df_tracking.iloc[:split_idx].copy()
test_df = df_tracking.iloc[split_idx:].copy()

print("--- Pembagian Kronologis Temporal ---")
print(f"Data Latih : {len(train_df):,} observasi")
print(f"Data Uji   : {len(test_df):,} observasi")

# 3. Normalisasi Koordinat (X, Y)
features = ['Center_X', 'Center_Y']
scaler = MinMaxScaler()

# Fit hanya pada Data Latih untuk mencegah kebocoran informasi masa depan (Data Leakage)
train_df[features] = scaler.fit_transform(train_df[features])
test_df[features] = scaler.transform(test_df[features])

# Simpan scaler
joblib.dump(scaler, 'models/scaler.pkl')
print("\\n✅ Scaler berhasil disimpan di 'models/scaler.pkl'")
"""))

cells.append(nbf.v4.new_markdown_cell("""\
### Bagian 4 — Data Engineering: Sliding Window
Model *Neural Network* berarsitektur *Sequence Modeling* memerlukan *input* berbentuk format 3D: `(batch_size, sequence_length, features)`.
Kita menggunakan algoritma **Sliding Window** dengan **look-back window = 10**, artinya model akan mengobservasi *history* 10 frame ke belakang untuk memprediksi probabilitas posisi 1 frame berikutnya.

Transformasi ini diproses secara terisolasi **per objek bola (Ball_ID)** agar trajektori bola yang satu tidak memengaruhi *sequence* bola lainnya.
"""))

cells.append(nbf.v4.new_code_cell("""\
# ============================================
# IMPLEMENTASI SOAL 4: SLIDING WINDOW
# ============================================
def create_sequences(data, seq_length=10):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i+seq_length)][features].values
        y = data.iloc[i+seq_length][features].values
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 10
X_train_list, y_train_list = [], []
X_test_list, y_test_list = [], []

# Transformasi per Ball_ID (Mencegah overlapping lintasan)
for ball_id in train_df['Ball_ID'].unique():
    b_df = train_df[train_df['Ball_ID'] == ball_id]
    if len(b_df) > seq_length:
        x, y = create_sequences(b_df, seq_length)
        X_train_list.append(x)
        y_train_list.append(y)

for ball_id in test_df['Ball_ID'].unique():
    b_df = test_df[test_df['Ball_ID'] == ball_id]
    if len(b_df) > seq_length:
        x, y = create_sequences(b_df, seq_length)
        X_test_list.append(x)
        y_test_list.append(y)

X_train = np.vstack(X_train_list)
y_train = np.vstack(y_train_list)
X_test = np.vstack(X_test_list)
y_test = np.vstack(y_test_list)

os.makedirs('data/processed/tensors', exist_ok=True)
np.save('data/processed/tensors/X_train.npy', X_train)
np.save('data/processed/tensors/y_train.npy', y_train)
np.save('data/processed/tensors/X_test.npy', X_test)
np.save('data/processed/tensors/y_test.npy', y_test)

print(f"🎯 Dimensi Tensor Latih : X={X_train.shape}, y={y_train.shape}")
print(f"🎯 Dimensi Tensor Uji   : X={X_test.shape}, y={y_test.shape}")
print("✅ Tensor sequence berhasil diekspor secara permanen!")
"""))

cells.append(nbf.v4.new_markdown_cell("""\
### Bagian 5 — Arsitektur Model (Simple RNN, LSTM, GRU)
Membangun dan mengeksekusi proses pelatihan (*training loop*) untuk 3 arsitektur fundamental JST Rekuren menggunakan **PyTorch**:
1. **Simple RNN**: Pendekatan dasar, efisien namun rentan terhadap *Vanishing Gradient*.
2. **LSTM**: Memiliki mekanisme *gates* untuk memori jangka panjang (*Long-Term Dependency*).
3. **GRU**: Varian penyederhanaan LSTM dengan performa komputasi yang lebih lincah dan kompetitif.
"""))

cells.append(nbf.v4.new_code_cell("""\
# ============================================
# IMPLEMENTASI SOAL 5: PEMBUATAN 3 MODEL & TRAINING
# ============================================
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Komputasi diproses menggunakan: {device}")

# Konversi NumPy -> PyTorch Tensor
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=64, shuffle=False)

# Kelas Pembungkus Arsitektur (Multi-Architecture Support)
class TrajectoryNet(nn.Module):
    def __init__(self, rnn_type, input_size=2, hidden_size=64, output_size=2):
        super(TrajectoryNet, self).__init__()
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :] # Ambil output probabilitas dari timestep terakhir
        out = self.fc(out)
        return out

models = {
    'Simple_RNN': TrajectoryNet('RNN').to(device),
    'LSTM': TrajectoryNet('LSTM').to(device),
    'GRU': TrajectoryNet('GRU').to(device)
}

criterion = nn.MSELoss()
epochs = 50
history = {name: {'train_loss': [], 'val_loss': []} for name in models}

print("\\n🚀 Memulai Proses Pelatihan Model...")
for name, model in models.items():
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
        history[name]['train_loss'].append(epoch_loss / len(train_loader))
        history[name]['val_loss'].append(val_loss / len(test_loader))
        
    torch.save(model.state_dict(), f'models/{name}.pth')
    print(f"✅ {name} Selesai dilatih! (Final Val Loss: {val_loss/len(test_loader):.4f}) -> models/{name}.pth")
"""))

cells.append(nbf.v4.new_markdown_cell("""\
### Bagian 6 — Evaluasi Proses Training (Loss Curves)
Menganalisis performa model selama masa *training* dengan melukiskan grafik metrik kesalahan (*Training vs Validation Loss*). Hal ini krusial untuk mengidentifikasi konvergensi fitur model atau terjebak dalam *overfitting/underfitting*.
"""))

cells.append(nbf.v4.new_code_cell("""\
# ============================================
# IMPLEMENTASI SOAL 6: KURVA LOSS TRAINING
# ============================================
import matplotlib.pyplot as plt

plt.figure(figsize=(18, 5))
for idx, name in enumerate(models.keys(), 1):
    plt.subplot(1, 3, idx)
    plt.plot(history[name]['train_loss'], label='Train Loss', color='blue')
    plt.plot(history[name]['val_loss'], label='Val Loss', color='orange')
    plt.title(f'Kurva Konvergensi - {name}', fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
os.makedirs('images/output', exist_ok=True)
plt.savefig('images/output/loss_curves.png', dpi=300)
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""\
### Bagian 7 — Pengujian & Komparasi Trajektori
Langkah pamungkas: Mengevaluasi performa numerik komparatif pada Data Uji tak terlihat (MSE dan RMSE), serta memvisualisasikan *Spatial Predictive Power* model dibandingkan dengan trajektori *Ground Truth*.

Sebagai validasi empiris sesuai arahan, kita akan membedah plot plot 2D (Sumbu X dan Y) spesifik untuk **ID Bola Dominan** (misalnya Bola 1).
"""))

cells.append(nbf.v4.new_code_cell("""\
# ============================================
# IMPLEMENTASI SOAL 7: KOMPARASI & VISUALISASI TRAJEKTORI
# ============================================
import math

results_list = []
for name, model in models.items():
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t)
        mse = criterion(preds, y_test_t).item()
        rmse = math.sqrt(mse)
        results_list.append({'Model': name, 'MSE': mse, 'RMSE': rmse})

df_results = pd.DataFrame(results_list)
print("🏆 TABEL KOMPARASI PERFORMA EVALUASI AKHIR:")
display(df_results.sort_values(by='RMSE').reset_index(drop=True))

# ----------------------------------------------------
# PLOT VISUALISASI TRAJEKTORI (2D SPATIAL X vs Y)
# ----------------------------------------------------
# Ambil entitas bola yang eksis dan paling stabil (dominan) di data test
ball_id_target = test_df['Ball_ID'].mode()[0]
ball_df = test_df[test_df['Ball_ID'] == ball_id_target]

if len(ball_df) > seq_length:
    X_b, y_b = create_sequences(ball_df, seq_length)
    X_b_t = torch.tensor(X_b, dtype=torch.float32).to(device)
    
    plt.figure(figsize=(10, 8))
    
    # Transformasi balik (Inverse) Ground Truth agar pixel kembali normal
    y_b_inv = scaler.inverse_transform(y_b)
    plt.plot(y_b_inv[:, 0], y_b_inv[:, 1], 'k-', label='Ground Truth (Asli)', linewidth=4, alpha=0.5)
    
    colors_plt = ['red', 'green', 'blue']
    for idx, (name, model) in enumerate(models.items()):
        model.eval()
        with torch.no_grad():
            preds_b = model(X_b_t).cpu().numpy()
            preds_b_inv = scaler.inverse_transform(preds_b)
            plt.plot(preds_b_inv[:, 0], preds_b_inv[:, 1], '--', color=colors_plt[idx], label=f'Prediksi {name}', linewidth=2)

    plt.title(f'Komparasi Lintasan Prediksi vs Aktual (Bola ID: {ball_id_target})', fontweight='bold', fontsize=16)
    plt.xlabel('Sumbu Resolusi X (Pixel)', fontsize=12)
    plt.ylabel('Sumbu Resolusi Y (Pixel)', fontsize=12)
    
    # Sangat Penting: Plot Y di-invert karena titik 0,0 pada OpenCV image mapping berada di pojok kiri atas
    plt.gca().invert_yaxis() 
    
    plt.legend(fontsize=12, shadow=True)
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig('images/output/trajectory_comparison.png', dpi=300)
    plt.show()
else:
    print(f"⚠️ Data Bola ID {ball_id_target} tidak mencukupi untuk render plot 2D.")
"""))

# PENULISAN FILE NOTEBOOK
nb.cells = cells
output_path = "notebooks/Ball Trajectory Prediction.ipynb"

with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"[OK] Notebook End-To-End berhasil dibangun dan disimpan ke {output_path}")
