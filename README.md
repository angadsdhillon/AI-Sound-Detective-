# AI Sound Detective

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Next.js 14](https://img.shields.io/badge/next.js-14-black)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)](https://fastapi.tiangolo.com/)

> **Advanced ML-Powered Sound Analysis Platform**

A production-grade full-stack application that uses deep learning to analyze audio files, providing comprehensive insights through sound classification, anomaly detection, spectrogram visualization, and audio fingerprinting.

## 🎯 Features

### Core Capabilities

- **🎵 Sound Classification** - Dual CNN architecture (2D spectrogram + 1D waveform) with ensemble predictions
- **📊 Advanced Visualization** - Interactive spectrograms and rotatable 3D FFT analysis using Plotly
- **🔍 Anomaly Detection** - Autoencoder-based reconstruction error + Isolation Forest for detecting unusual patterns
- **🎯 Audio Fingerprinting** - Similarity search using learned embeddings
- **🎤 Real-time Recording** - Browser-based audio recording with waveform visualization
- **📄 PDF Export** - Generate beautifully formatted analysis reports
- **🚀 Production Ready** - Dockerized, tested, and deployment-ready

### Machine Learning Pipeline

```
Audio Input → Feature Extraction → ML Inference → Visualization
    ↓              ↓                    ↓               ↓
  Upload     Mel Spectrogram    2D CNN Classifier  Spectrogram
  Record     MFCC Features      1D CNN Classifier  3D FFT Plot
  Video      FFT Analysis       Autoencoder        Waveform
                                Isolation Forest   Report
```

## 🏗 Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     AI Sound Detective                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐         ┌──────────────┐                  │
│  │   Frontend  │────────▶│   Backend    │                  │
│  │  Next.js 14 │         │   FastAPI    │                  │
│  │  TypeScript │         │   Python     │                  │
│  │  Tailwind   │         │              │                  │
│  └─────────────┘         └──────┬───────┘                  │
│                                  │                          │
│                          ┌───────▼────────┐                │
│                          │  ML Pipeline   │                │
│                          ├────────────────┤                │
│                          │ • 2D CNN       │                │
│                          │ • 1D CNN       │                │
│                          │ • Autoencoder  │                │
│                          │ • Iso. Forest  │                │
│                          └────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python** 3.10+
- **Node.js** 20+
- **FFmpeg** (for video processing)
- **Docker** (optional, for containerized deployment)

### Local Development

#### 1. Clone Repository

```bash
git clone https://github.com/yourusername/AI-Sound-Detective.git
cd AI-Sound-Detective
```

#### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at `http://localhost:8000`

#### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Frontend will be available at `http://localhost:3000`

## 🛠 Tech Stack

### Frontend
- **Next.js 14** - React framework with App Router
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **shadcn/ui** - UI components
- **Plotly.js** - 3D visualizations

### Backend
- **FastAPI** - Modern Python web framework
- **PyTorch** - Deep learning framework
- **librosa** - Audio analysis
- **scikit-learn** - ML utilities

### DevOps
- **Docker** - Containerization
- **pytest** - Testing

## 📝 License

MIT License

---

**Built with ❤️ using PyTorch, FastAPI, and Next.js**
AI sound detective 
