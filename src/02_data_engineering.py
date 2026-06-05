import pandas as pd
import numpy as np
import os
import joblib
import warnings
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')

print("[INFO] Membaca dataset CSV hasil tracking...")
df = pd.read_csv('data/raw/dataset_koordinat_bola.csv')

# Mengurutkan data berdasarkan urutan waktu (Frame)
df = df.sort_values(by=['Ball_ID', 'Frame_ID'])

# Menghapus ID Bola yang durasinya terlalu pendek (noise/false positive YOLO)
# Mengambil 3 Bola utama dengan kemunculan (frame) paling banyak
top_balls = df['Ball_ID'].value_counts().nlargest(3).index
df = df[df['Ball_ID'].isin(top_balls)]

if len(df) == 0:
    print("[ERROR] Tidak ada data bola yang valid di dataset.")
    exit(1)

# ==========================================
# SOAL 3: Train/Test Split (Kronologis)
# ==========================================
print("\n[INFO] Mengeksekusi Soal 3: Train/Test Split secara kronologis...")
# Mencari frame maksimum untuk mencari titik potong 66.6% waktu
max_frame = df['Frame_ID'].max()
split_frame = int(max_frame * 0.666)

# Membagi berdasarkan waktu (Data Latih = 2 menit awal, Data Uji = 1 menit akhir)
train_df = df[df['Frame_ID'] <= split_frame].copy()
test_df = df[df['Frame_ID'] > split_frame].copy()

print(f"Batas Frame Split: {split_frame} dari total {max_frame} frames")
print(f"-> Data Latih: {len(train_df)} koordinat (66.6% awal)")
print(f"-> Data Uji  : {len(test_df)} koordinat (33.3% akhir)")

# Normalisasi MinMax (Sangat disarankan untuk stabilitas gradien Neural Network)
scaler = MinMaxScaler()
train_df[['Center_X', 'Center_Y']] = scaler.fit_transform(train_df[['Center_X', 'Center_Y']])
test_df[['Center_X', 'Center_Y']] = scaler.transform(test_df[['Center_X', 'Center_Y']])

# ==========================================
# SOAL 4: Sliding Window
# ==========================================
def create_sliding_window(dataset, look_back=10):
    """
    Fungsi transformasi deret waktu spasial menjadi Tensor Input/Output.
    X: Koordinat 10 frame sebelumnya
    y: Koordinat target 1 frame ke depan
    """
    X, Y = [], []
    # Dikelompokkan per ID bola agar tidak ada data window lintas bola
    for ball_id, group in dataset.groupby('Ball_ID'):
        coords = group[['Center_X', 'Center_Y']].values
        for i in range(len(coords) - look_back):
            X.append(coords[i:(i + look_back)])
            Y.append(coords[i + look_back])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

LOOK_BACK = 10 # Menentukan ukuran memori frame ke belakang
print(f"\n[INFO] Mengeksekusi Soal 4: Sliding Window dengan look-back = {LOOK_BACK} frame...")

X_train, y_train = create_sliding_window(train_df, look_back=LOOK_BACK)
X_test, y_test = create_sliding_window(test_df, look_back=LOOK_BACK)

print(f"\n[PEMBAHASAN SOAL 4: BENTUK DIMENSI TENSOR AKHIR]")
print("Format Tensor: (Jumlah Sampel, Time-Steps/Look-Back, Fitur)")
print(f"-> Tensor X_train (Input Latih)  : {X_train.shape}")
print(f"-> Tensor y_train (Target Latih) : {y_train.shape}")
print(f"-> Tensor X_test (Input Uji)     : {X_test.shape}")
print(f"-> Tensor y_test (Target Uji)    : {y_test.shape}")

# Menyimpan hasil tensor agar bisa diload oleh PyTorch di tahap berikutnya
os.makedirs('data/processed/tensors', exist_ok=True)
np.save('data/processed/tensors/X_train.npy', X_train)
np.save('data/processed/tensors/y_train.npy', y_train)
np.save('data/processed/tensors/X_test.npy', X_test)
np.save('data/processed/tensors/y_test.npy', y_test)

# Menyimpan scaler untuk kebutuhan un-scaling saat plot visualisasi Soal 7
joblib.dump(scaler, 'models/scaler.pkl')
print("\n[SUCCESS] Seluruh Tensor dan Scaler berhasil disimpan di disk lokal!")
