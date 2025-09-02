"""Pydantic schemas for API request/response validation"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


class AnalysisRequest(BaseModel):
    """Request schema for audio analysis"""
    use_ensemble: bool = Field(default=True, description="Use ensemble prediction")
    detect_anomalies: bool = Field(default=True, description="Run anomaly detection")
    find_similar: bool = Field(default=True, description="Find similar sounds")


class PredictionResult(BaseModel):
    """Individual prediction result"""
    class_name: str = Field(alias="class")
    probability: float
    confidence: float
    
    class Config:
        populate_by_name = True


class ClassificationResult(BaseModel):
    """Classification results"""
    method: str
    top_class: str
    top_probability: float
    all_predictions: List[Dict[str, Any]]
    individual_predictions: Optional[Dict[str, Any]] = None


class AnomalyResult(BaseModel):
    """Anomaly detection result"""
    type: str
    severity: str
    score: float
    description: str


class AnomalyDetectionResult(BaseModel):
    """Anomaly detection results"""
    is_anomalous: bool
    anomaly_score: float
    anomalies_detected: List[AnomalyResult]
    reconstruction_error: float
    threshold: float


class SimilarSound(BaseModel):
    """Similar sound match"""
    id: str
    name: str
    similarity: float
    category: Optional[str] = None


class AudioMetadata(BaseModel):
    """Audio file metadata"""
    duration: float
    sample_rate: int
    samples: int
    rms_energy: float
    max_amplitude: float
    zero_crossing_rate: float


class SpectrogramData(BaseModel):
    """Spectrogram visualization data"""
    frequencies: List[float]
    times: List[float]
    magnitude: List[List[float]]
    magnitude_db: List[List[float]]


class FFT3DData(BaseModel):
    """3D FFT visualization data"""
    frequencies: List[float]
    times: List[float]
    magnitude: List[List[float]]
    magnitude_db: List[List[float]]


class AnalysisResponse(BaseModel):
    """Complete analysis response"""
    id: str
    filename: str
    timestamp: datetime
    metadata: AudioMetadata
    classification: ClassificationResult
    anomaly_detection: Optional[AnomalyDetectionResult] = None
    similar_sounds: Optional[List[SimilarSound]] = None
    spectrogram: Optional[Dict[str, Any]] = None
    fft_3d: Optional[Dict[str, Any]] = None
    waveform: Optional[List[float]] = None


class UploadResponse(BaseModel):
    """File upload response"""
    file_id: str
    filename: str
    size: int
    content_type: str
    message: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: datetime
    models_loaded: Dict[str, bool]


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: str
    timestamp: datetime
