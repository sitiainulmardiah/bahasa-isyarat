import os
import sys
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =====================================================
# PATH SETUP
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.config import MODEL_DIR, SEQUENCE_LENGTH
from src.post_processing import PredictionSmoother
# PERBAIKAN: Import fungsi penyelarasan tangan kiri/kanan agar logika web sama dengan lokal
from src.preprocessing import extract_two_hands

# =====================================================
# FASTAPI APP SETUP
# =====================================================
app = FastAPI(
    title="VisiSign Production API",
    version="2.2",
    description="API yang diselaraskan untuk inferensi web dengan pelacakan dual-handedness."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=2)

# =====================================================
# MODEL LOADING
# =====================================================
STATIC_MODEL_PATH = os.path.join(MODEL_DIR, "static_model.h5")
DYNAMIC_MODEL_PATH = os.path.join(MODEL_DIR, "dynamic_model.h5")

static_model = None
dynamic_model = None

def load_labels(filename: str) -> Dict[int, str]:
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}

labels_static = {}
labels_dynamic = {}
smoother = PredictionSmoother()

def safe_load_model(model_path: str):
    if not os.path.exists(model_path):
        print(f"[WARN] Model tidak ditemukan di: {model_path}")
        return None
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as exc:
        print(f"[ERROR] Load model gagal: {model_path}\n{exc}")
        return None

print("\n=============================================")
print("Memuat Jaringan Jurnal VisiSign Production API...")
print("=============================================")
static_model = safe_load_model(STATIC_MODEL_PATH)
dynamic_model = safe_load_model(DYNAMIC_MODEL_PATH)

try:
    labels_static = load_labels("label_static.json")
    labels_dynamic = load_labels("label_dynamic.json")
    print("[INFO] Sukses memuat file label mapping mapping.")
except Exception as exc:
    print(f"[ERROR] Gagal memuat berkas label JSON: {exc}")

@app.on_event("startup")
async def warmup_models():
    loop = asyncio.get_event_loop()
    if static_model is not None:
        dummy_static = np.zeros((1, 42, 3), dtype=np.float32)
        await loop.run_in_executor(executor, lambda: static_model.predict(dummy_static, verbose=0))
        print("[INFO] Warmup model statis selesai.")
    if dynamic_model is not None:
        dummy_dynamic = np.zeros((1, SEQUENCE_LENGTH, 42, 3), dtype=np.float32)
        await loop.run_in_executor(executor, lambda: dynamic_model.predict(dummy_dynamic, verbose=0))
        print("[INFO] Warmup model dinamis selesai.")

# =====================================================
# SCHEMAS (STRUKTUR PAYLOAD BARU)
# =====================================================
class HandLandmarkInput(BaseModel):
    # Skema koordinat individual per tangan dari MediaPipe Web Client
    label: str  # 'Left' atau 'Right'
    landmarks: List[List[float]]  # Array berukuran (21, 3) -> x, y, z

class PredictionRequest(BaseModel):
    # Frontend harus mengirimkan list tangan yang terdeteksi pada frame saat ini
    detected_hands: List[HandLandmarkInput]
    mode: str = "STATIC"

class PredictionResponse(BaseModel):
    prediction: str
    confidence: str
    confidence_value: float
    mode: str

# Objek penampung sekuensial dinamis global untuk client web tunggal
web_sequence_buffer = []

# =====================================================
# LOGIKA UTAMA PIPELINE PREDIKSI
# =====================================================
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    global web_sequence_buffer, smoother
    loop = asyncio.get_event_loop()
    mode = request.mode.upper()

    # Mocking struktur MediaPipe Python agar bisa dibaca oleh src/preprocessing.py
    class MockClass:
        def __init__(self, label): self.label = label
    class MockHandedness:
        def __init__(self, label): self.classification = [MockClass(label)]
    class MockLandmark:
        def __init__(self, x, y, z): self.x, self.y, self.z = x, y, z
    class MockHandLandmarks:
        def __init__(self, lms): self.landmark = [MockLandmark(pt[0], pt[1], pt[2]) for pt in lms]

    multi_hand_landmarks = []
    multi_handedness = []

    # Rekonstruksi struktur data JSON web ke standarisasi internal src
    for hand in request.detected_hands:
        multi_hand_landmarks.append(MockHandLandmarks(hand.landmarks))
        multi_handedness.append(MockHandedness(hand.label))

    # Ekstrak & Selaraskan koordinat menggunakan fungsi internal (Shape: 42, 3) yang sudah dikunci Kiri/Kanan
    combined_landmarks = extract_two_hands(multi_hand_landmarks, multi_handedness)

    if mode == "STATIC":
        if static_model is None:
            raise HTTPException(status_code=500, detail="Model statis h5 belum termuat di server")
        
        input_data = np.expand_dims(combined_landmarks, axis=0) # Shape: (1, 42, 3)
        predictions = await loop.run_in_executor(
            executor, lambda: static_model.predict(input_data, verbose=0)[0]
        )
        idx = int(np.argmax(predictions))
        confidence = float(predictions[idx])
        label = labels_static.get(idx, "UNKNOWN")

    elif mode == "DYNAMIC":
        if dynamic_model is None:
            raise HTTPException(status_code=500, detail="Model dinamis h5 belum termuat di server")
        
        # Simpan ke buffer sekuensial internal server
        web_sequence_buffer.append(combined_landmarks)
        if len(web_sequence_buffer) > SEQUENCE_LENGTH:
            web_sequence_buffer.pop(0) # Batasi sliding window sesuai konstanta config (30)

        # Jika sekuens belum terpenuhi, berikan respons instruktif awal
        if len(web_sequence_buffer) < SEQUENCE_LENGTH:
            return PredictionResponse(
                prediction="Mengumpulkan data gerakan...",
                confidence="0.00%",
                confidence_value=0.0,
                mode=mode
            )

        input_data = np.expand_dims(np.array(web_sequence_buffer), axis=0) # Shape: (1, 30, 42, 3)
        predictions = await loop.run_in_executor(
            executor, lambda: dynamic_model.predict(input_data, verbose=0)[0]
        )
        idx = int(np.argmax(predictions))
        confidence = float(predictions[idx])
        label = labels_dynamic.get(idx, "UNKNOWN")
    else:
        raise HTTPException(status_code=400, detail="Mode operasi harus STATIC atau DYNAMIC")

    # Terapkan penyaringan distorsi lewat Majority Voting PredictionSmoother
    smoothed_label = smoother.process(label, confidence)

    return PredictionResponse(
        prediction=smoothed_label,
        confidence=f"{confidence * 100:.2f}%",
        confidence_value=confidence,
        mode=mode,
    )

@app.post("/smoother/clear")
def clear_smoother():
    global web_sequence_buffer
    web_sequence_buffer.clear()
    smoother.clear_buffer()
    return {"status": "ok", "message": "Buffer penampung sekuensial web berhasil di-reset."}

@app.get("/health")
def health():
    return {"status": "healthy", "static_model": static_model is not None, "dynamic_model": dynamic_model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8001, reload=True)