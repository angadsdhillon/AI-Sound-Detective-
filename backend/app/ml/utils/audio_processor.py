"""Audio processing utilities for ML pipelines"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from typing import Tuple, Dict, Optional
import io


class AudioProcessor:
    """Advanced audio processing for ML feature extraction"""
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def load_audio(
        self,
        file_path: str,
        duration: Optional[float] = None,
        offset: float = 0.0
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file and resample to target sample rate
        
        Args:
            file_path: Path to audio file
            duration: Duration to load in seconds (None for entire file)
            offset: Start time in seconds
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        y, sr = librosa.load(
            file_path,
            sr=self.sample_rate,
            duration=duration,
            offset=offset,
            mono=True
        )
        return y, sr
    
    def extract_mel_spectrogram(
        self,
        audio: np.ndarray,
        to_db: bool = True
    ) -> np.ndarray:
        """
        Extract mel spectrogram from audio
        
        Args:
            audio: Audio waveform
            to_db: Convert to decibel scale
            
        Returns:
            Mel spectrogram (n_mels, time_frames)
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        if to_db:
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec
    
    def extract_mfcc(
        self,
        audio: np.ndarray,
        n_mfcc: int = 40
    ) -> np.ndarray:
        """
        Extract MFCC features
        
        Args:
            audio: Audio waveform
            n_mfcc: Number of MFCC coefficients
            
        Returns:
            MFCC features (n_mfcc, time_frames)
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return mfcc
    
    def compute_stft(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute Short-Time Fourier Transform
        
        Args:
            audio: Audio waveform
            
        Returns:
            Complex STFT matrix
        """
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return stft
    
    def compute_3d_fft(
        self,
        audio: np.ndarray,
        segment_length: int = 2048
    ) -> Dict[str, np.ndarray]:
        """
        Compute 3D FFT analysis for visualization
        
        Args:
            audio: Audio waveform
            segment_length: Length of each segment
            
        Returns:
            Dictionary with frequency, time, and magnitude data
        """
        # Compute STFT
        stft = self.compute_stft(audio)
        magnitude = np.abs(stft)
        
        # Create time and frequency axes
        times = librosa.frames_to_time(
            np.arange(magnitude.shape[1]),
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        freqs = librosa.fft_frequencies(
            sr=self.sample_rate,
            n_fft=self.n_fft
        )
        
        return {
            "frequencies": freqs,
            "times": times,
            "magnitude": magnitude,
            "magnitude_db": librosa.amplitude_to_db(magnitude, ref=np.max)
        }
    
    def extract_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract comprehensive audio features for ML
        
        Args:
            audio: Audio waveform
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Mel Spectrogram
        features["mel_spectrogram"] = self.extract_mel_spectrogram(audio)
        
        # MFCC
        features["mfcc"] = self.extract_mfcc(audio)
        
        # Chroma features
        features["chroma"] = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Spectral features
        features["spectral_centroid"] = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        features["spectral_rolloff"] = librosa.feature.spectral_rolloff(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        features["zero_crossing_rate"] = librosa.feature.zero_crossing_rate(
            audio,
            hop_length=self.hop_length
        )
        
        # Temporal features
        features["rms"] = librosa.feature.rms(
            y=audio,
            hop_length=self.hop_length
        )
        
        return features
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range"""
        return librosa.util.normalize(audio)
    
    def apply_preprocessing(
        self,
        audio: np.ndarray,
        trim_silence: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Apply standard preprocessing pipeline
        
        Args:
            audio: Raw audio waveform
            trim_silence: Whether to trim leading/trailing silence
            normalize: Whether to normalize amplitude
            
        Returns:
            Preprocessed audio
        """
        if trim_silence:
            audio, _ = librosa.effects.trim(audio, top_db=20)
        
        if normalize:
            audio = self.normalize_audio(audio)
        
        return audio
    
    def get_audio_metadata(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract metadata from audio
        
        Args:
            audio: Audio waveform
            
        Returns:
            Dictionary of metadata
        """
        duration = librosa.get_duration(y=audio, sr=self.sample_rate)
        
        return {
            "duration": duration,
            "sample_rate": self.sample_rate,
            "samples": len(audio),
            "rms_energy": float(np.sqrt(np.mean(audio**2))),
            "max_amplitude": float(np.max(np.abs(audio))),
            "zero_crossing_rate": float(np.mean(librosa.zero_crossings(audio)))
        }


def convert_video_to_audio(
    video_path: str,
    output_path: str,
    sample_rate: int = 22050
) -> str:
    """
    Extract audio from video file using FFmpeg
    
    Args:
        video_path: Path to input video
        output_path: Path for output audio
        sample_rate: Target sample rate
        
    Returns:
        Path to extracted audio file
    """
    import subprocess
    
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",  # No video
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",  # Mono
        "-y",  # Overwrite output
        output_path
    ]
    
    try:
        subprocess.run(command, check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg conversion failed: {e.stderr.decode()}")
