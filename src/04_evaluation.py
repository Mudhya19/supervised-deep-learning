import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.metrics import mean_squared_error

# ==========================================
# Definisi Ulang Arsitektur Model (Untuk Dimuat)
# ==========================================
class SimpleRNN(nn.Module):
    def __init__(self):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(2, 32, batch_first=True)
        self.fc = nn.Linear(32, 2)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(2, 32, batch_first=True)
        self.fc = nn.Linear(32, 2)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class GRUModel(nn.Module):
    def __init__(self):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(2, 32, batch_first=True)
        self.fc = nn.Linear(32, 2)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

print("[INFO] Memuat Tensor Data Uji dan Model terlatih...")
X_test = np.load('data/processed/tensors/X_test.npy')
y_test = np.load('data/processed/tensors/y_test.npy')

X_test_t = torch.from_numpy(X_test)
y_test_t = torch.from_numpy(y_test)

# Memuat Scaler untuk mengembalikan (un-scale) nilai ke piksel layar asli
scaler = joblib.load('models/scaler.pkl')

models = {
    'Simple RNN': SimpleRNN(),
    'LSTM': LSTMModel(),
    'GRU': GRUModel()
}

# Memuat bobot (weights)
for name, model in models.items():
    file_name = 'Simple_RNN' if name == 'Simple RNN' else name
    model.load_state_dict(torch.load(f'models/{file_name}.pth'))
    model.eval() # Set mode evaluasi

# ==========================================
# SOAL 7: Pengujian, Tabel Komparasi, Visualisasi 2D
# ==========================================

results = []
predictions = {}

# Konversi Target Asli ke skala piksel riil
y_test_original = scaler.inverse_transform(y_test)

for name, model in models.items():
    with torch.no_grad():
        preds_scaled = model(X_test_t).numpy()
    
    # Konversi hasil prediksi ke skala piksel riil
    preds_original = scaler.inverse_transform(preds_scaled)
    predictions[name] = preds_original
    
    # Hitung MSE dan RMSE
    mse = mean_squared_error(y_test_original, preds_original)
    rmse = np.sqrt(mse)
    results.append({'Arsitektur Model': name, 'MSE': mse, 'RMSE': rmse})

df_results = pd.DataFrame(results).sort_values(by='RMSE')
print("\n" + "="*60)
print("TABEL KOMPARASI EVALUASI (Berbasis Data Uji / Pixel Asli)")
print("="*60)
print(df_results.to_string(index=False))

# Simpulan Analitis Otomatis
best_model = df_results.iloc[0]['Arsitektur Model']
print(f"\n[KESIMPULAN ANALITIS]")
print(f"Arsitektur yang memiliki performa paling unggul adalah {best_model}.")
print("Analisis Mengapa Terjadi: Simple RNN seringkali gagal menangkap pola momentum jarak panjang ")
print("(vanishing gradient) pada rekaman video sekuensial. Sebaliknya, LSTM/GRU memiliki ")
print("gerbang (gates/memory cell) yang menjaga 'ingatan' terhadap kecepatan dan arah bola ")
print("dari frame terdahulu, sehingga prediksinya jauh lebih presisi.\n")

# Visualisasi Trajektori Sumbu X vs Sumbu Y (Plot 2D)
print("[INFO] Membuat Plot 2D Perbandingan Trajektori Koordinat...")
plt.figure(figsize=(10, 6))

# Plot Asli
plt.plot(y_test_original[:, 0], y_test_original[:, 1], 'k--', label='Ground Truth (Lintasan Asli)', linewidth=3)

# Plot Prediksi
colors = {'Simple RNN': 'blue', 'LSTM': 'green', 'GRU': 'red'}
for name in models.keys():
    plt.plot(predictions[name][:, 0], predictions[name][:, 1], label=f'Prediksi {name}', color=colors[name], alpha=0.8, linewidth=2)

plt.title('Komparasi Lintasan Bola (Koordinat X vs Koordinat Y)')
plt.xlabel('Koordinat Horizontal (X)')
plt.ylabel('Koordinat Vertikal (Y)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('images/output/trajectory_comparison.png', dpi=300)
print("[SUCCESS] Grafik 2D Trajektori berhasil disimpan di: images/output/trajectory_comparison.png")
