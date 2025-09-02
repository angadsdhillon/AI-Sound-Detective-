"""API routes for sound analysis"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
import os
import uuid
import shutil
from datetime import datetime
from typing import Optional
import numpy as np

from ..core.config import settings
from ..ml.utils.audio_processor import AudioProcessor, convert_video_to_audio
from ..ml.inference import SoundClassifier, AnomalyDetector, FingerprintMatcher
from .schemas import (
    AnalysisResponse,
    UploadResponse,
    ClassificationResult,
    AnomalyDetectionResult,
    AudioMetadata,
    SimilarSound
)

router = APIRouter()

# Initialize ML components
audio_processor = AudioProcessor(
    sample_rate=settings.SAMPLE_RATE,
    n_mels=settings.N_MELS,
    n_fft=settings.N_FFT,
    hop_length=settings.HOP_LENGTH
)
classifier = SoundClassifier()
anomaly_detector = AnomalyDetector()
fingerprint_matcher = FingerprintMatcher()


def cleanup_file(file_path: str):
    """Background task to cleanup temporary files"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error cleaning up file {file_path}: {e}")


@router.post("/upload", response_model=UploadResponse)
async def upload_audio(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload an audio or video file for analysis
    
    - Accepts: mp3, wav, m4a, flac, ogg, mp4, mov, avi, mkv
    - Max size: 100MB
    """
    # Validate file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    allowed_extensions = (
        settings.ALLOWED_AUDIO_EXTENSIONS + 
        settings.ALLOWED_VIDEO_EXTENSIONS
    )
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Generate unique file ID
    file_id = str(uuid.uuid4())
    file_path = os.path.join(settings.UPLOAD_DIR, f"{file_id}{file_ext}")
    
    # Save uploaded file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_size = os.path.getsize(file_path)
        
        # Check file size
        if file_size > settings.MAX_UPLOAD_SIZE:
            os.remove(file_path)
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {settings.MAX_UPLOAD_SIZE // (1024*1024)}MB"
            )
        
        return UploadResponse(
            file_id=file_id,
            filename=file.filename,
            size=file_size,
            content_type=file.content_type or "unknown",
            message="File uploaded successfully"
        )
    
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/analyze/{file_id}", response_model=AnalysisResponse)
async def analyze_audio(
    file_id: str,
    use_ensemble: bool = True,
    detect_anomalies: bool = True,
    find_similar: bool = True,
    background_tasks: BackgroundTasks = None
):
    """
    Analyze uploaded audio file
    
    - Performs ML classification
    - Detects anomalies
    - Finds similar sounds
    - Generates visualizations
    """
    # Find uploaded file
    matching_files = [
        f for f in os.listdir(settings.UPLOAD_DIR)
        if f.startswith(file_id)
    ]
    
    if not matching_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_path = os.path.join(settings.UPLOAD_DIR, matching_files[0])
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        # Convert video to audio if needed
        if file_ext in settings.ALLOWED_VIDEO_EXTENSIONS:
            audio_path = os.path.join(
                settings.UPLOAD_DIR,
                f"{file_id}_extracted.wav"
            )
            convert_video_to_audio(file_path, audio_path, settings.SAMPLE_RATE)
            actual_audio_path = audio_path
        else:
            actual_audio_path = file_path
        
        # Load and preprocess audio
        audio, sr = audio_processor.load_audio(actual_audio_path)
        audio = audio_processor.apply_preprocessing(audio)
        
        # Extract metadata
        metadata_dict = audio_processor.get_audio_metadata(audio)
        metadata = AudioMetadata(**metadata_dict)
        
        # Classification
        if use_ensemble:
            classification_result = classifier.ensemble_predict(audio)
        else:
            classification_result = classifier.predict_from_spectrogram(audio)
        
        classification = ClassificationResult(**classification_result)
        
        # Anomaly detection
        anomaly_result = None
        if detect_anomalies:
            anomaly_dict = anomaly_detector.detect_anomalies(audio)
            anomaly_result = AnomalyDetectionResult(**anomaly_dict)
        
        # Similar sounds
        similar_sounds = None
        if find_similar:
            similar_list = fingerprint_matcher.find_similar_sounds(audio, top_k=5)
            similar_sounds = [SimilarSound(**s) for s in similar_list]
        
        # Generate visualizations
        # Spectrogram
        mel_spec = audio_processor.extract_mel_spectrogram(audio, to_db=True)
        times_spec = np.linspace(0, metadata.duration, mel_spec.shape[1])
        freqs_mel = np.linspace(0, sr / 2, mel_spec.shape[0])
        
        spectrogram_data = {
            "frequencies": freqs_mel.tolist(),
            "times": times_spec.tolist(),
            "magnitude": mel_spec.tolist()
        }
        
        # 3D FFT
        fft_3d_dict = audio_processor.compute_3d_fft(audio)
        fft_3d_data = {
            "frequencies": fft_3d_dict["frequencies"].tolist(),
            "times": fft_3d_dict["times"].tolist(),
            "magnitude": fft_3d_dict["magnitude"].tolist(),
            "magnitude_db": fft_3d_dict["magnitude_db"].tolist()
        }
        
        # Waveform (downsample for frontend)
        waveform_samples = 2000
        if len(audio) > waveform_samples:
            indices = np.linspace(0, len(audio) - 1, waveform_samples, dtype=int)
            waveform = audio[indices].tolist()
        else:
            waveform = audio.tolist()
        
        # Schedule cleanup
        if background_tasks:
            background_tasks.add_task(cleanup_file, file_path)
            if file_ext in settings.ALLOWED_VIDEO_EXTENSIONS:
                background_tasks.add_task(cleanup_file, actual_audio_path)
        
        return AnalysisResponse(
            id=file_id,
            filename=os.path.basename(matching_files[0]),
            timestamp=datetime.now(),
            metadata=metadata,
            classification=classification,
            anomaly_detection=anomaly_result,
            similar_sounds=similar_sounds,
            spectrogram=spectrogram_data,
            fft_3d=fft_3d_data,
            waveform=waveform
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@router.get("/report/{file_id}")
async def generate_pdf_report(file_id: str):
    """
    Generate and download PDF report for analysis
    
    Returns a beautifully formatted PDF with all analysis results
    """
    # This will be implemented with reportlab/fpdf2
    # For now, return a placeholder response
    
    raise HTTPException(
        status_code=501,
        detail="PDF generation endpoint - coming soon"
    )


@router.delete("/file/{file_id}")
async def delete_file(file_id: str):
    """Delete uploaded file"""
    matching_files = [
        f for f in os.listdir(settings.UPLOAD_DIR)
        if f.startswith(file_id)
    ]
    
    if not matching_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    for filename in matching_files:
        file_path = os.path.join(settings.UPLOAD_DIR, filename)
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    return {"message": "File(s) deleted successfully"}
