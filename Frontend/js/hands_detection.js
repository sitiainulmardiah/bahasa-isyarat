document.addEventListener('DOMContentLoaded', async () => {

    console.log('VisiSign MediaPipe Hands Module Loaded');

    const API_HOSTS = [
        'http://127.0.0.1:8001',
        'http://127.0.0.1:8000'
    ];
    let apiHost = API_HOSTS[0];
    let apiOnline = false;

    const getApiUrl = () => `${apiHost}/predict`;

    // Fungsi otomatis mendeteksi server API yang aktif
    const checkApiHost = async () => {
        for (const host of API_HOSTS) {
            try {
                const response = await fetch(`${host}/health`, { method: 'GET' });
                if (response.ok) {
                    apiHost = host;
                    apiOnline = true;
                    console.log(`FastAPI Server terdeteksi aktif pada: ${host}`);
                    return;
                }
            } catch (error) {
                // Abaikan dan coba port cadangan berikutnya
            }
        }
        apiOnline = false;
    };

    await checkApiHost();

    const webcamElement = document.getElementById('webcam');
    const canvasElement = document.getElementById('canvas');

    const predictionText = document.getElementById('prediction-text');
    const confidenceText = document.getElementById('confidence-text');

    const btnClearHistory = document.getElementById('btn-clear-history');
    const btnResetPrediction = document.getElementById('btn-reset-prediction');
    const btnToggleMode = document.getElementById('btn-toggle-mode');

    const historyLog = document.getElementById('history-log');

    if (!webcamElement || !canvasElement) {
        console.error('Komponen DOM Webcam atau Canvas tidak ditemukan!');
        return;
    }

    const canvasCtx = canvasElement.getContext('2d');

    // =====================================================
    // STATE MANAGEMENT
    // =====================================================
    let currentMode = "DYNAMIC"; // Default mode sekuensial dinamis (Kata)
    
    // Cooldown penanganan banjir request HTTP POST
    const PREDICT_COOLDOWN_MS = 150;   
    let lastPredictTime = 0;

    // Auto-reset label display jika tidak ada gerakan baru dalam 2.5 detik
    const PREDICTION_RESET_MS = 2500;  
    let predictionResetTimer = null;

    let lastLoggedWord = "";
    let lastLoggedTime = 0;

    // =====================================================
    // INITIALIZE MEDIAPIPE HANDS
    // =====================================================
    const hands = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });

    hands.setOptions({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.7,
        minTrackingConfidence: 0.5
    });

    window.hands = hands;

    // =====================================================
    // TRANSMISI PIPELINE KE FASTAPI
    // =====================================================
    const sendPredictionRequest = async (detectedHandsPayload) => {
        const now = Date.now();

        // Blokir pengiriman jika request dikirim terlalu cepat sebelum cooldown selesai
        if (now - lastPredictTime < PREDICT_COOLDOWN_MS) return;

        if (!apiOnline) {
            await checkApiHost();
        }

        if (!apiOnline) {
            if (predictionText) predictionText.textContent = 'PREDIKSI: ERROR API';
            if (confidenceText) {
                confidenceText.className = 'text-sm font-semibold text-red-500 flex items-center justify-center gap-2';
                confidenceText.innerHTML = `<i class="fa-solid fa-server"></i> FastAPI Backend offline (Nyalakan api.py di port 8001)`;
            }
            return;
        }

        lastPredictTime = now;

        try {
            const response = await fetch(getApiUrl(), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    detected_hands: detectedHandsPayload, // Payload terstruktur Pydantic baru
                    mode: currentMode
                })
            });

            if (!response.ok) {
                const errorText = await response.text();
                console.error("API Response Error:", response.status, errorText);
                return;
            }

            const data = await response.json();
            updateUI(data.prediction, data.confidence);

        } catch (error) {
            console.error("Gagal melakukan fetch request ke backend:", error);
            if (predictionText) predictionText.textContent = 'PREDIKSI: CONN_TIMEOUT';
        }
    };

    // =====================================================
    // UI OPERATIONS & AUTO-RESET CONTROL
    // =====================================================
    const resetPredictionDisplay = () => {
        if (predictionText) predictionText.textContent = 'PREDIKSI: -';
        if (confidenceText) {
            confidenceText.className = 'text-sm font-semibold text-amber-500 flex items-center justify-center gap-2';
            confidenceText.innerHTML = `<i class="fa-solid fa-hand text-amber-400"></i> Menunggu gerakan isyarat...`;
        }
    };

    const updateUI = (label, confidence) => {
        if (!predictionText || !confidenceText) return;

        const ignoredLabels = [
            "Menunggu...",
            "Tidak ada tangan",
            "Menunggu gerakan...",
            "Mengumpulkan data gerakan..."
        ];

        if (!ignoredLabels.includes(label) && label !== "") {
            predictionText.textContent = `PREDIKSI: ${label.toUpperCase()}`;
            confidenceText.className = 'text-sm font-semibold text-emerald-600 flex items-center justify-center gap-2';
            confidenceText.innerHTML = `<i class="fa-solid fa-circle-check"></i> Akurasi Terfilter: ${confidence}`;

            const now = Date.now();
            if (label !== lastLoggedWord || now - lastLoggedTime > 3000) {
                addToHistory(label);
                lastLoggedWord = label;
                lastLoggedTime = now;
            }

            // Jalankan auto-reset timer baru
            clearTimeout(predictionResetTimer);
            predictionResetTimer = setTimeout(resetPredictionDisplay, PREDICTION_RESET_MS);

        } else {
            predictionText.textContent = 'PREDIKSI: -';
            confidenceText.className = 'text-sm font-semibold text-amber-500 flex items-center justify-center gap-2';
            confidenceText.innerHTML = `<i class="fa-solid fa-hand animate-pulse"></i> ${label}`;
            
            clearTimeout(predictionResetTimer);
        }
    };

    const addToHistory = (word) => {
        if (!historyLog) return;

        const placeholder = historyLog.querySelector('.italic');
        if (placeholder) placeholder.remove();

        const now = new Date();
        const time = now.toLocaleTimeString('id-ID');

        const item = document.createElement('div');
        item.className = 'py-2.5 flex justify-between items-center text-slate-900 border-b border-slate-100 animate-fade-in';
        item.innerHTML = `
            <span class="flex items-center gap-2 font-bold text-indigo-600">
                <i class="fa-solid fa-clock text-slate-300"></i> ${word.toUpperCase()}
            </span>
            <span class="text-[10px] font-mono text-slate-400">${time}</span>
        `;

        historyLog.insertBefore(item, historyLog.firstChild);

        while (historyLog.children.length > 10) {
            historyLog.removeChild(historyLog.lastChild);
        }
    };

    // =====================================================
    // CORE CALLBACK: MEDIAPIPE FRAME PROCESSING
    // =====================================================
    hands.onResults((results) => {
        // Sesuaikan resolusi canvas koordinat gambar diam
        if (canvasElement.width !== webcamElement.videoWidth || canvasElement.height !== webcamElement.videoHeight) {
            canvasElement.width = webcamElement.videoWidth;
            canvasElement.height = webcamElement.videoHeight;
        }

        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

        // Siapkan struktur array kosong untuk menampung data frame saat ini
        let detectedHandsPayload = [];

        if (results.multiHandLandmarks && results.multiHandedness) {
            // 1. Gambar Visual Skala Garis Landmark Sendi di Layar
            results.multiHandLandmarks.forEach((landmarks) => {
                drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: '#6366f1', lineWidth: 3 });
                drawLandmarks(canvasCtx, landmarks, { color: '#3b82f6', radius: 2 });
            });

            // 2. PARSING STRUKTUR DATA UTUH KIRI/KANAN UNTUK PYDANTIC BACKEND
            for (let i = 0; i < results.multiHandLandmarks.length; i++) {
                const rawLandmarks = results.multiHandLandmarks[i];
                const handLabel = results.multiHandedness[i].label; // Ambil nilai string 'Left' atau 'Right'

                // Mapping objek koordinat {x, y, z} MediaPipe menjadi Array murni [[x,y,z], ...]
                const landmarksMapped = rawLandmarks.map(lm => [lm.x, lm.y, lm.z]);

                detectedHandsPayload.push({
                    label: handLabel,
                    landmarks: landmarksMapped
                });
            }
        }

        canvasCtx.restore();

        // Jalankan fungsi pengiriman data frame, logika handling kosong diatur di internal server api.py
        sendPredictionPayload(detectedHandsPayload);
    });

    // Helper fungsi pengirim wrapper payload
    const sendPredictionPayload = (payload) => {
        sendPredictionRequest(payload);
    };

    // =====================================================
    // CONTROL COMPONENT INTERACTIONS
    // =====================================================
    if (btnToggleMode) {
        btnToggleMode.addEventListener('click', async (e) => {
            e.preventDefault();
            clearTimeout(predictionResetTimer);
            resetPredictionDisplay();

            // Beritahu server untuk membersihkan memori sliding window sekuens
            try {
                await fetch(`${apiHost}/smoother/clear`, { method: 'POST' });
            } catch(err) { console.log(err); }

            if (currentMode === "DYNAMIC") {
                currentMode = "STATIC";
                btnToggleMode.innerHTML = `
                    <span class="w-8 h-8 bg-pink-600 text-white rounded-xl flex items-center justify-center text-[11px] shadow-sm">
                        <i class="fa-solid fa-font"></i>
                    </span>
                    <div>
                        <p class="text-slate-900">Mode: Huruf/Angka</p>
                        <p class="text-[9px] text-pink-500 font-medium">Ubah ke kata dinamis</p>
                    </div>
                `;
            } else {
                currentMode = "DYNAMIC";
                btnToggleMode.innerHTML = `
                    <span class="w-8 h-8 bg-indigo-600 text-white rounded-xl flex items-center justify-center text-[11px] shadow-sm">
                        <i class="fa-solid fa-exchange-alt"></i>
                    </span>
                    <div>
                        <p class="text-slate-900">Mode: Kata</p>
                        <p class="text-[9px] text-indigo-500 font-medium">Ubah ke huruf/angka</p>
                    </div>
                `;
            }
            console.log("Mode Operasional Sistem Diubah Ke:", currentMode);
        });
    }

    if (btnResetPrediction) {
        btnResetPrediction.addEventListener('click', async (e) => {
            e.preventDefault();
            lastLoggedWord = "";
            lastPredictTime = 0;
            clearTimeout(predictionResetTimer);
            resetPredictionDisplay();
            try {
                await fetch(`${apiHost}/smoother/clear`, { method: 'POST' });
            } catch(err) { console.log(err); }
            console.log("Buffer inferensi dan sekuensial dibersihkan.");
        });
    }

    if (btnClearHistory) {
        btnClearHistory.addEventListener('click', (e) => {
            e.preventDefault();
            historyLog.innerHTML = `<div class="py-4 text-center text-slate-400 text-[11px] italic">Belum ada isyarat yang divalidasi</div>`;
            lastLoggedWord = "";
        });
    }
});