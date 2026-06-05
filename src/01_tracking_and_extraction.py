import cv2
import pandas as pd
from ultralytics import YOLO
import os

# Memastikan direktori output tersedia
os.makedirs('data/raw', exist_ok=True)

print("[INFO] Memuat model YOLOv8n...")
# Menggunakan model yolov8n.pt (akan diunduh otomatis jika belum ada)
model = YOLO('yolov8n.pt')

video_path = 'docs/3 Basketball Bouncing.mp4'
if not os.path.exists(video_path):
    print(f"[ERROR] Video tidak ditemukan di lokasi: {video_path}")
    exit(1)

cap = cv2.VideoCapture(video_path)

tracking_data = []
frame_id = 0

print("[INFO] Memulai pemrosesan video. Proses ini mungkin memakan waktu beberapa menit tergantung durasi video...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break # Video selesai
        
    # YOLO Tracking dengan algoritma bawaan (BoT-SORT)
    # classes=[32] digunakan agar hanya mendeteksi "sports ball" (class id 32 di COCO dataset)
    results = model.track(frame, persist=True, classes=[32], tracker="botsort.yaml", verbose=False)
    
    # Ekstraksi Bounding Box dan ID jika ada objek yang terdeteksi
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy()
        
        for box, track_id in zip(boxes, track_ids):
            x_min, y_min, x_max, y_max = box
            
            # Hitung titik koordinat pusat bola (X, Y)
            center_x = (x_min + x_max) / 2.0
            center_y = (y_min + y_max) / 2.0
            
            tracking_data.append({
                'Frame_ID': frame_id,
                'Ball_ID': int(track_id),
                'Center_X': center_x,
                'Center_Y': center_y
            })
            
    frame_id += 1
    if frame_id % 500 == 0:
        print(f"[INFO] Telah memproses {frame_id} frame...")

cap.release()
print("[INFO] Pemrosesan video selesai. Menyimpan hasil ke file CSV...")

# Konversi ke Pandas DataFrame dan simpan ke CSV
df_tracking = pd.DataFrame(tracking_data)
csv_filename = 'data/raw/dataset_koordinat_bola.csv'
df_tracking.to_csv(csv_filename, index=False)

print(f"[SUCCESS] Berhasil mengekstrak {len(df_tracking)} titik koordinat!")
print(f"[SUCCESS] File CSV tersimpan di: {csv_filename}")
print("\nPratinjau Data (5 Baris Pertama):")
print(df_tracking.head())
