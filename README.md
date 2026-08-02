# 🌾 ZAAR-001: AI-Based Crop Disease Early Detection

> **Ground-Up Hyper-Local Agri-Intelligence Platform Built for Smallholder Farmers**

---

## 📌 Executive Summary

**ZAAR-001** is a hybrid, offline-first mobile application engineered to solve the acute agricultural advisory gap across India (particularly South India and Tamil Nadu). By combining lightweight on-device computer vision, native voice and regional language interfaces, crowdsourced peer-farmer wisdom, and government advisory linkages (mKisan/Kisan Call Centres), ZAAR-001 empowers low-literacy farmers to diagnose crop diseases and execute corrective actions in seconds—even without an active internet connection.

---

## 💡 Key Differentiators (ZAAR-001 vs. Generic Tools)

Unlike broad-spectrum tools like Google Lens or cloud-only commercial apps, ZAAR-001 is built specifically for rural field realities:

* **🌾 Hyper-Regional Data Integration:** Fine-tuned on local Indian crop varieties, regional disease patterns, and real field conditions (messy backgrounds, variable lighting, multi-leaf conditions) across Tamil Nadu and South India.
* **⚡ Hybrid Offline-First Architecture:** Runs on-device local AI inference via quantized TensorFlow Lite (`.tflite`). Works fully without internet; automatically queues metadata for background cloud sync when network connectivity is detected.
* **🎙️ Voice-First Regional UX:** Supports native voice-input and text-to-speech output in **Tamil, Telugu, Kannada, Malayalam, and Hindi**, enabling non-literate and low-literate farmers to interact without typing.
* **🤝 Community & Peer-Validated Remedies:** Combines organic-first cures (e.g., Neem oil mixtures) with verified local agricultural wisdom contributed by peer farmers and validated by KVK agronomists.
* **🗺️ State-Level Outbreak Mapping:** Syncs localized telemetry to generate anonymized district-level heatmap intelligence for agricultural departments and extension officers.

---

## ⚙️ Architecture & Data Workflow

```
[ Field Camera Capture ]
          │
          ▼
[ On-Device TFLite Model ] ──(Inference)──► [ Local Diagnosis & Confidence Score ]
          │                                                    │
          │                                                    ▼
          ├────────────────────────────────► [ Local DB: Localized Remedies & Audio ]
          │
          ▼ (When Network Available)
[ Offline Background Queue ] ──────────────► [ Central Cloud Sync & Outbreak Heatmap ]
```

### 🔄 Step-by-Step Processing Pipeline
1. **Field Data Capture (Offline):** On-device camera preview guides the farmer to position the leaf. Automatic capture records GPS, lighting, and timestamp metadata.
2. **Local AI Inference (Mobile CPU):** Quantized MobileNetV3 / EfficientNet-Lite model classifies the image on-device in under **300ms**.
3. **Localized Advisory:** The app queries an embedded SQLite database to pull regional remedy cards (Organic + Chemical fallback) and provides voice-playback in the chosen local language.
4. **Queue Management:** Diagnostics are serialized locally. If confidence falls below 75%, the case is marked for expert escalation.
5. **Data Sync & Outbreak Heatmap:** Telemetry uploads silently upon network detection to update state-wide disease outbreak maps for government interventions.

---

## 🛠️ Technology Stack

| Layer | Recommended Technology / Framework |
| :--- | :--- |
| **Frontend / App** | Flutter / React Native (Cross-platform, low-end Android support) |
| **On-Device Machine Learning** | TensorFlow Lite (TFLite) / ONNX Runtime Mobile |
| **Model Architectures** | MobileNetV3-Small / EfficientNet-Lite0 / YOLOv8-cls |
| **Data & Local Storage** | SQLite / Room DB / Hive |
| **Language & Audio Layer** | AI4Bharat IndicTrans2 & IndicTTS / Google ML Kit Voice |
| **Backend & APIs** | FastAPI (Python) / Node.js + PostgreSQL |
| **Geospatial & Analytics** | Mapbox / Leaflet.js (Outbreak Heatmaps) |

---

## 📂 Repository Structure

```
zaar-001/
├── assets/
│   ├── models/            # Quantized .tflite model binaries
│   ├── audio/             # Pre-rendered local voice prompts
│   └── images/            # UI icons and static workflow diagrams
├── docs/
│   ├── workflow_diagram.png
│   └── architecture_spec.md
├── mobile_app/            # Flutter cross-platform mobile codebase
│   ├── lib/
│   │   ├── camera/        # Camera overlay and frame processor
│   │   ├── tflite/        # On-device model runner
│   │   ├── db/            # Local SQLite database handlers
│   │   └── ui/            # High-accessibility regional UI screens
├── server/                # Cloud fallback API and Sync engine
│   ├── api/               # FastAPI endpoints
│   ├── analytics/         # District outbreak map generator
│   └── database/          # Global disease & remedy catalog
└── README.md              # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
* Flutter SDK (`>=3.10.0`) or Android Studio (NDK enabled)
* Python `3.10+` (for backend & model quantizer scripts)
* Physical Android Test Device (recommended for camera & TFLite hardware acceleration testing)

### Mobile App Setup
```bash
# Clone the repository
git clone https://github.com/your-org/zaar-001.git
cd zaar-001/mobile_app

# Get dependencies
flutter pub get

# Run on connected Android device
flutter run --release
```

### Model Quantization & Export (Python)
```bash
cd zaar-001/server
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Convert trained MobileNet model to TFLite
python scripts/export_tflite.py --model_path models/crop_mobilenet.h5 --output_path assets/models/crop_v1.tflite
```

---

## 🏆 Pitch & Hackathon Highlights

* **Real-World Ready:** Built specifically to overcome the **1:1,162 extension officer shortage** in India.
* **Zero Barrier to Entry:** Designed for low-literacy users with single-tap interactions, audio feedback, and regional language toggles.
* **Actionable Output:** Provides immediate organic remedies first, preventing pesticide overuse while offering chemical fallbacks for severe infections.

---
