"""ML inference engine for sound analysis"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import IsolationForest
import pickle
import os

from ..models.networks import SpectrogramCNN, WaveformCNN, AudioAutoencoder
from ..utils.audio_processor import AudioProcessor
from ...core.config import settings


class SoundClassifier:
    """Sound classification using 2D and 1D CNNs"""
    
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.audio_processor = AudioProcessor(
            sample_rate=settings.SAMPLE_RATE,
            n_mels=settings.N_MELS,
            n_fft=settings.N_FFT,
            hop_length=settings.HOP_LENGTH
        )
        
        # Initialize models
        self.model_2d = SpectrogramCNN(num_classes=settings.NUM_CLASSES)
        self.model_1d = WaveformCNN(num_classes=settings.NUM_CLASSES)
        
        # Load pretrained weights if available
        self._load_models()
        
        # Class labels (customizable based on your dataset)
        self.class_labels = [
            "Speech",
            "Music",
            "Environmental",
            "Animal",
            "Engine",
            "Alarm",
            "Tools",
            "Water",
            "Wind",
            "Other"
        ]
    
    def _load_models(self):
        """Load pretrained model weights"""
        if os.path.exists(settings.CLASSIFIER_2D_PATH):
            try:
                self.model_2d.load_state_dict(
                    torch.load(settings.CLASSIFIER_2D_PATH, map_location=self.device)
                )
                self.model_2d.eval()
            except Exception as e:
                print(f"Warning: Could not load 2D classifier: {e}")
        
        if os.path.exists(settings.CLASSIFIER_1D_PATH):
            try:
                self.model_1d.load_state_dict(
                    torch.load(settings.CLASSIFIER_1D_PATH, map_location=self.device)
                )
                self.model_1d.eval()
            except Exception as e:
                print(f"Warning: Could not load 1D classifier: {e}")
        
        self.model_2d.to(self.device)
        self.model_1d.to(self.device)
    
    def predict_from_spectrogram(
        self,
        audio: np.ndarray
    ) -> Dict[str, any]:
        """
        Classify sound using 2D CNN on spectrogram
        
        Args:
            audio: Audio waveform
            
        Returns:
            Dictionary with predictions and probabilities
        """
        # Extract mel spectrogram
        mel_spec = self.audio_processor.extract_mel_spectrogram(audio)
        
        # Normalize to [0, 1]
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)
        
        # Convert to tensor and add batch dimension
        spec_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0).unsqueeze(0)
        spec_tensor = spec_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            probabilities = self.model_2d.predict_proba(spec_tensor)
            probabilities = probabilities.cpu().numpy()[0]
        
        # Get top predictions
        top_indices = np.argsort(probabilities)[::-1][:5]
        
        predictions = []
        for idx in top_indices:
            predictions.append({
                "class": self.class_labels[idx] if idx < len(self.class_labels) else f"Class_{idx}",
                "probability": float(probabilities[idx]),
                "confidence": float(probabilities[idx] * 100)
            })
        
        return {
            "method": "2D CNN (Spectrogram)",
            "top_class": predictions[0]["class"],
            "top_probability": predictions[0]["probability"],
            "all_predictions": predictions,
            "probabilities": probabilities.tolist()
        }
    
    def predict_from_waveform(
        self,
        audio: np.ndarray
    ) -> Dict[str, any]:
        """
        Classify sound using 1D CNN on raw waveform
        
        Args:
            audio: Audio waveform
            
        Returns:
            Dictionary with predictions and probabilities
        """
        # Ensure fixed length (pad or truncate)
        target_length = settings.SAMPLE_RATE * 4  # 4 seconds
        
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
        audio_tensor = audio_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            probabilities = self.model_1d.predict_proba(audio_tensor)
            probabilities = probabilities.cpu().numpy()[0]
        
        # Get top predictions
        top_indices = np.argsort(probabilities)[::-1][:5]
        
        predictions = []
        for idx in top_indices:
            predictions.append({
                "class": self.class_labels[idx] if idx < len(self.class_labels) else f"Class_{idx}",
                "probability": float(probabilities[idx]),
                "confidence": float(probabilities[idx] * 100)
            })
        
        return {
            "method": "1D CNN (Waveform)",
            "top_class": predictions[0]["class"],
            "top_probability": predictions[0]["probability"],
            "all_predictions": predictions,
            "probabilities": probabilities.tolist()
        }
    
    def ensemble_predict(self, audio: np.ndarray) -> Dict[str, any]:
        """
        Ensemble prediction combining 2D and 1D models
        
        Args:
            audio: Audio waveform
            
        Returns:
            Combined predictions
        """
        pred_2d = self.predict_from_spectrogram(audio)
        pred_1d = self.predict_from_waveform(audio)
        
        # Average probabilities
        probs_2d = np.array(pred_2d["probabilities"])
        probs_1d = np.array(pred_1d["probabilities"])
        ensemble_probs = (probs_2d + probs_1d) / 2
        
        # Get top predictions
        top_indices = np.argsort(ensemble_probs)[::-1][:5]
        
        predictions = []
        for idx in top_indices:
            predictions.append({
                "class": self.class_labels[idx] if idx < len(self.class_labels) else f"Class_{idx}",
                "probability": float(ensemble_probs[idx]),
                "confidence": float(ensemble_probs[idx] * 100)
            })
        
        return {
            "method": "Ensemble (2D + 1D CNN)",
            "top_class": predictions[0]["class"],
            "top_probability": predictions[0]["probability"],
            "all_predictions": predictions,
            "individual_predictions": {
                "spectrogram_model": pred_2d,
                "waveform_model": pred_1d
            }
        }


class AnomalyDetector:
    """Anomaly detection using Isolation Forest and Autoencoder"""
    
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.audio_processor = AudioProcessor()
        
        # Autoencoder for reconstruction-based anomaly detection
        self.autoencoder = AudioAutoencoder(latent_dim=128)
        self._load_autoencoder()
        
        # Isolation Forest for feature-based anomaly detection
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
    
    def _load_autoencoder(self):
        """Load pretrained autoencoder"""
        if os.path.exists(settings.AUTOENCODER_PATH):
            try:
                self.autoencoder.load_state_dict(
                    torch.load(settings.AUTOENCODER_PATH, map_location=self.device)
                )
                self.autoencoder.eval()
            except Exception as e:
                print(f"Warning: Could not load autoencoder: {e}")
        
        self.autoencoder.to(self.device)
    
    def detect_anomalies(self, audio: np.ndarray) -> Dict[str, any]:
        """
        Detect anomalies in audio using multiple methods
        
        Args:
            audio: Audio waveform
            
        Returns:
            Anomaly detection results
        """
        # Extract features
        features = self.audio_processor.extract_features(audio)
        mel_spec = features["mel_spectrogram"]
        
        # Normalize spectrogram
        mel_spec_norm = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)
        
        # Prepare tensor
        spec_tensor = torch.from_numpy(mel_spec_norm).float().unsqueeze(0).unsqueeze(0)
        spec_tensor = spec_tensor.to(self.device)
        
        # Autoencoder reconstruction error
        with torch.no_grad():
            reconstruction_error = self.autoencoder.reconstruction_error(spec_tensor)
            reconstruction_error = reconstruction_error.cpu().numpy()[0]
        
        # Determine if anomalous
        is_anomaly = reconstruction_error > settings.ANOMALY_THRESHOLD
        
        anomalies = []
        if is_anomaly:
            anomalies.append({
                "type": "High Reconstruction Error",
                "severity": "high" if reconstruction_error > 0.3 else "medium",
                "score": float(reconstruction_error),
                "description": "Audio pattern differs significantly from normal sounds"
            })
        
        # Additional statistical anomalies
        metadata = self.audio_processor.get_audio_metadata(audio)
        
        if metadata["max_amplitude"] > 0.95:
            anomalies.append({
                "type": "Clipping Detected",
                "severity": "high",
                "score": float(metadata["max_amplitude"]),
                "description": "Audio signal is clipping (exceeds normal amplitude range)"
            })
        
        if metadata["rms_energy"] < 0.01:
            anomalies.append({
                "type": "Low Energy",
                "severity": "low",
                "score": float(metadata["rms_energy"]),
                "description": "Audio has very low energy/volume"
            })
        
        return {
            "is_anomalous": len(anomalies) > 0,
            "anomaly_score": float(reconstruction_error),
            "anomalies_detected": anomalies,
            "reconstruction_error": float(reconstruction_error),
            "threshold": settings.ANOMALY_THRESHOLD
        }


class FingerprintMatcher:
    """Audio fingerprinting and similarity search"""
    
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.audio_processor = AudioProcessor()
        self.autoencoder = AudioAutoencoder(latent_dim=128)
        self._load_autoencoder()
        
        # Fingerprint database
        self.fingerprint_db: List[Dict] = []
        self._load_fingerprint_index()
    
    def _load_autoencoder(self):
        """Load pretrained autoencoder for fingerprinting"""
        if os.path.exists(settings.AUTOENCODER_PATH):
            try:
                self.autoencoder.load_state_dict(
                    torch.load(settings.AUTOENCODER_PATH, map_location=self.device)
                )
                self.autoencoder.eval()
            except Exception:
                pass
        
        self.autoencoder.to(self.device)
    
    def _load_fingerprint_index(self):
        """Load fingerprint database"""
        if os.path.exists(settings.FINGERPRINT_INDEX_PATH):
            try:
                with open(settings.FINGERPRINT_INDEX_PATH, 'rb') as f:
                    self.fingerprint_db = pickle.load(f)
            except Exception:
                pass
    
    def generate_fingerprint(self, audio: np.ndarray) -> np.ndarray:
        """
        Generate audio fingerprint using autoencoder
        
        Args:
            audio: Audio waveform
            
        Returns:
            Fingerprint vector
        """
        mel_spec = self.audio_processor.extract_mel_spectrogram(audio)
        mel_spec_norm = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)
        
        spec_tensor = torch.from_numpy(mel_spec_norm).float().unsqueeze(0).unsqueeze(0)
        spec_tensor = spec_tensor.to(self.device)
        
        with torch.no_grad():
            fingerprint = self.autoencoder.encode(spec_tensor)
            fingerprint = fingerprint.cpu().numpy()[0]
        
        return fingerprint
    
    def find_similar_sounds(
        self,
        audio: np.ndarray,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Find similar sounds in database
        
        Args:
            audio: Query audio
            top_k: Number of similar sounds to return
            
        Returns:
            List of similar sounds with similarity scores
        """
        if not self.fingerprint_db:
            # Return mock similar sounds for demo
            return [
                {
                    "id": f"sound_{i}",
                    "name": f"Similar Sound {i+1}",
                    "similarity": float(0.85 - i * 0.1),
                    "category": "Environmental"
                }
                for i in range(min(top_k, 3))
            ]
        
        query_fp = self.generate_fingerprint(audio)
        
        # Compute cosine similarity
        similarities = []
        for item in self.fingerprint_db:
            db_fp = item["fingerprint"]
            similarity = np.dot(query_fp, db_fp) / (
                np.linalg.norm(query_fp) * np.linalg.norm(db_fp) + 1e-8
            )
            similarities.append({
                **item,
                "similarity": float(similarity)
            })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        return similarities[:top_k]
