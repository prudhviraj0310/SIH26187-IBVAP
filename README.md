# 🔍 IBVAP — AI-Based Intelligent Video Analytics Platform for Border Surveillance

> **SIH Problem Statement:** SIH26187  
> **Ministry Sponsor:** Ministry of Home Affairs / Sashastra Seema Bal (SSB)  
> **Theme:** Smart Automation / Security  
> **Track:** SOFTWARE  
> **Smart India Hackathon 2026**

---

## 🎯 Problem Statement

India's international borders are monitored by thousands of CCTV cameras, but human operators cannot physically watch all feeds 24/7. Critical intrusion events go undetected until it's too late. Current systems lack AI-powered real-time analytics to detect, classify, and alert on suspicious activities automatically.

## 🔬 Our Solution

IBVAP is an **AI-powered intelligent video analytics platform** that augments existing CCTV infrastructure with real-time computer vision to detect border intrusions, track suspects, recognize vehicles, and alert command centers automatically.

### Core Capabilities

- **Tripwire & Intrusion Detection**: Virtual perimeter fencing with instant alerts on boundary crossing
- **ANPR (Automatic Number Plate Recognition)**: Real-time vehicle identification at border checkpoints
- **Behavioral Analytics**: Anomaly detection for loitering, unusual movement patterns, and crowd formation
- **Face Watchlist Matching**: Real-time face detection against known suspect databases
- **Night/Thermal Enhancement**: Low-light image enhancement for nocturnal surveillance

## 🏗️ Architecture

```
┌──────────────────────┐    ┌──────────────────────────┐
│  CCTV/IP Camera      │───▶│  YOLOv8 Object Detection │
│  Feed (RTSP/HTTP)    │    │  Person/Vehicle/Animal   │
└──────────────────────┘    └──────────┬───────────────┘
                                       ▼
┌──────────────────────┐    ┌──────────────────────────┐
│  Virtual Tripwire    │───▶│  Zone Intrusion Engine   │
│  Boundary Polygons   │    │  Direction-Aware Alerts  │
└──────────────────────┘    └──────────┬───────────────┘
                                       ▼
┌──────────────────────┐    ┌──────────────────────────┐
│  ANPR + Face Match   │───▶│  Suspect Identification  │
│  OCR + Embeddings    │    │  Watchlist Cross-Check   │
└──────────────────────┘    └──────────┬───────────────┘
                                       ▼
                            ┌──────────────────────────┐
                            │  Command Center Alert    │
                            │  + Incident Report PDF   │
                            └──────────────────────────┘
```

## 🚀 Quick Start

```bash
# 1. Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install ultralytics opencv-python-headless numpy easyocr

# 3. Run detection on a video
python engine.py --source video.mp4 --mode tripwire
```

## 📁 Project Structure

```
├── engine.py          # Main detection orchestrator
├── detector.py        # YOLOv8 object detector wrapper
├── tripwire.py        # Virtual fence / intrusion detection
├── anpr.py            # Automatic Number Plate Recognition
├── behavior.py        # Behavioral anomaly analytics
├── face_watch.py      # Face detection & watchlist matching
├── enhancer.py        # Low-light / thermal image enhancement
└── __init__.py        # Package metadata
```

## 🧪 Detection Modules

| Module | Description |
|:---|:---|
| **Tripwire** | Virtual perimeter fencing with direction-aware alerts |
| **ANPR** | License plate detection + OCR for vehicle identification |
| **Behavior** | Loitering detection, unusual trajectory analysis |
| **Face Watch** | Real-time face matching against suspect database |
| **Enhancer** | CLAHE + dehazing for night/fog conditions |

## 📜 License

MIT License — developed for Smart India Hackathon 2026

## 👥 Team

Built for SIH 2026 | Problem Statement SIH26187
