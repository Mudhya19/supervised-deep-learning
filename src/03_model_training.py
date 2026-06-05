import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

print("[INFO] Memuat Tensor Latih dan Uji dari disk...")
X_train = np.load('data/processed/tensors/X_train.npy')
y_train = np.load('data/processed/tensors/y_train.npy')
X_test = np.load('data/processed/tensors/X_test.npy')
y_test = np.load('data/processed/tensors/y_test.npy')

# Konversi numpy array ke PyTorch Tensor Float
X_train_t = torch.from_numpy(X_train)
y_train_t = torch.from_numpy(y_train)
X_val_t = torch.from_numpy(X_test)  # Menggunakan Data Uji sebagai Set Validasi untuk memantau kurva
y_val_t = torch.from_numpy(y_test)

# Membuat DataLoader untuk kemudahan perulangan batch
from torch.utils.data import TensorDataset, DataLoader
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=16, shuffle=True)

# ==========================================
# SOAL 5: Arsitektur Model (Simple RNN, LSTM, GRU)
# ==========================================

class SimpleRNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=2):
        super(SimpleRNN, self).__init__()
        # batch_first=True artinya dimensi tensor [Batch, Sequence_length, Features]
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :]) # Hanya ambil output di frame/time-step terakhir
        return out

class LSTMModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class GRUModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

# ==========================================
# SOAL 6: Proses Training & Evaluasi Kurva
# ==========================================

def train_model(model, name, epochs=150, lr=0.01):
    criterion = nn.MSELoss() # Menggunakan fungsi Loss Regresi (MSE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    print(f"\n[INFO] Memulai Proses Latihan (Training) untuk Model: {name}")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
            
        train_loss = epoch_loss / len(X_train_t)
        train_losses.append(train_loss)
        
        # Mode Validasi (Memantau Loss tanpa melatih bobot model)
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
            val_losses.append(val_loss)
            
        # Tampilkan metrik per 50 epoch
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss (MSE): {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
            
    # Menyimpan model (Weight/Bobot) ke disk
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), f'models/{name}.pth')
    return train_losses, val_losses

# Inisialisasi ketiga model
rnn_model = SimpleRNN()
lstm_model = LSTMModel()
gru_model = GRUModel()

# Melatih ketiga model dan menyimpan catatan (history) loss
loss_history = {}
loss_history['Simple_RNN'] = train_model(rnn_model, "Simple_RNN")
loss_history['LSTM'] = train_model(lstm_model, "LSTM")
loss_history['GRU'] = train_model(gru_model, "GRU")

# Membuat Visualisasi Kurva Evaluasi
print("\n[INFO] Menyiapkan Grafik Evaluasi Kurva Pergerakan Loss...")
os.makedirs('images/output', exist_ok=True)
plt.figure(figsize=(15, 5))

models_list = ['Simple_RNN', 'LSTM', 'GRU']
for i, name in enumerate(models_list):
    plt.subplot(1, 3, i+1)
    train_l, val_l = loss_history[name]
    
    plt.plot(train_l, label='Training Loss', color='blue')
    plt.plot(val_l, label='Validation Loss', color='orange')
    plt.title(f'Kurva {name}')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig('images/output/loss_curves.png', dpi=300)
print("[SUCCESS] Grafik berhasil disimpan ke: images/output/loss_curves.png")
