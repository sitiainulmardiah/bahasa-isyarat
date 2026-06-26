import os
import numpy as np
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# TAMBAHAN: Import untuk metrik evaluasi
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import dari folder src
from src.config import DYNAMIC_DATA_DIR, MODEL_DIR, DYNAMIC_MODEL_PATH
from src.model_architectures import build_dynamic_model

def train():
    print("Memuat data dinamis...")
    X_path = os.path.join(DYNAMIC_DATA_DIR, 'X_dynamic.npy')
    y_path = os.path.join(DYNAMIC_DATA_DIR, 'y_dynamic.npy')
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print(f"Error: Data tidak ditemukan di {DYNAMIC_DATA_DIR}. Jalankan extract_features.py dulu!")
        return

    X = np.load(X_path)
    y = np.load(y_path)

    # Encode label teks ('Halo', 'Tolong') menjadi angka (0, 1, dst)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)

    # Simpan mapping label ke JSON
    os.makedirs(MODEL_DIR, exist_ok=True)
    label_mapping = {int(index): str(label) for index, label in enumerate(le.classes_)}
    with open(os.path.join(MODEL_DIR, 'label_dynamic.json'), 'w') as f:
        json.dump(label_mapping, f)

    # Split data training dan testing (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    print(f"Jumlah kelas: {num_classes}")
    print(f"Data latih: {X_train.shape}, Data uji: {X_test.shape}")

    # Bangun dan latih model
    model = build_dynamic_model(num_classes)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    print("\nMulai pelatihan model dinamis...")
    model.fit(X_train, y_train, epochs=150, batch_size=16, validation_data=(X_test, y_test), callbacks=[early_stop])

    # Simpan model
    model.save(DYNAMIC_MODEL_PATH)
    print(f"\nModel dinamis berhasil disimpan di: {DYNAMIC_MODEL_PATH}")

    # =========================================================================
    # TAMBAHAN: FITUR EVALUASI AKURASI & CONFUSION MATRIX
    # =========================================================================
    print("\n" + "="*50)
    print("Memulai Evaluasi Model Dinamis...")
    print("="*50)

    # 1. Lakukan prediksi probabilitas pada data uji (X_test)
    y_pred_probs = model.predict(X_test)
    
    # 2. Ubah probabilitas menjadi indeks kelas (label angka)
    if num_classes > 2 or y_pred_probs.shape[-1] > 1:
        y_pred = np.argmax(y_pred_probs, axis=1)
    else:
        # Penanganan khusus jika klasifikasi biner menggunakan 1 output node
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    # 3. Hitung & Tampilkan Akurasi Keseluruhan
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTingkat Akurasi Model Dinamis pada Data Uji: {acc * 100:.2f}%")

    # 4. Tampilkan Laporan Klasifikasi (Precision, Recall, F1-Score per kata)
    print("\nLaporan Klasifikasi (Classification Report):")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 5. Tampilkan Confusion Matrix dalam bentuk teks/matriks angka
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix (Teks):")
    print(cm)
    
    # 6. OPSI VISUALISASI: Gambar Grafik Confusion Matrix Heatmap
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(8, 6))
        # Menggunakan warna 'Purples' untuk membedakan visualisasi dari model statis
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title('Confusion Matrix - Model Dinamis')
        plt.ylabel('Label Sebenarnya (Actual Labels)')
        plt.xlabel('Label Prediksi (Predicted Labels)')
        
        # Simpan hasil grafik heatmap ke dalam folder models
        plot_path = os.path.join(MODEL_DIR, 'confusion_matrix_dynamic.png')
        plt.savefig(plot_path)
        print(f"\nGrafik Confusion Matrix berhasil disimpan di: {plot_path}")
        plt.show()
    except ImportError:
        print("\nInfo: Silakan install library 'matplotlib' dan 'seaborn' jika ingin melihat grafik gambar dari Confusion Matrix.")

if __name__ == "__main__":
    train()