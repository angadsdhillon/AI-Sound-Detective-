"""Application configuration management"""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings and configuration"""
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "AI Sound Detective"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "Advanced ML-powered sound analysis platform"
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]
    
    # File Upload
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    UPLOAD_DIR: str = "uploads"
    ALLOWED_AUDIO_EXTENSIONS: List[str] = [".mp3", ".wav", ".m4a", ".flac", ".ogg"]
    ALLOWED_VIDEO_EXTENSIONS: List[str] = [".mp4", ".mov", ".avi", ".mkv"]
    
    # ML Models
    MODELS_DIR: str = "ml/models"
    CLASSIFIER_2D_PATH: str = "ml/models/classifier_2d.pth"
    CLASSIFIER_1D_PATH: str = "ml/models/classifier_1d.pth"
    AUTOENCODER_PATH: str = "ml/models/autoencoder.pth"
    FINGERPRINT_INDEX_PATH: str = "ml/models/fingerprint_index.pkl"
    
    # Audio Processing
    SAMPLE_RATE: int = 22050
    N_MELS: int = 128
    N_FFT: int = 2048
    HOP_LENGTH: int = 512
    
    # ML Inference
    BATCH_SIZE: int = 32
    NUM_CLASSES: int = 10  # Adjust based on your dataset
    ANOMALY_THRESHOLD: float = 0.15
    
    # Environment
    DEBUG: bool = True
    
    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.MODELS_DIR, exist_ok=True)
