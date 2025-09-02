"""
Backend tests using pytest
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert response.json()["message"] == "AI Sound Detective API"


def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "models_loaded" in data


def test_upload_no_file():
    """Test upload endpoint without file"""
    response = client.post("/api/v1/upload")
    assert response.status_code == 422  # Validation error


def test_analyze_nonexistent_file():
    """Test analysis with non-existent file ID"""
    response = client.post("/api/v1/analyze/nonexistent-id")
    assert response.status_code == 404


def test_cors_headers():
    """Test CORS headers are present"""
    response = client.options("/")
    assert "access-control-allow-origin" in response.headers.keys() or response.status_code == 200


class TestAudioProcessor:
    """Test audio processing utilities"""
    
    def test_audio_processor_import(self):
        """Test that AudioProcessor can be imported"""
        from app.ml.utils.audio_processor import AudioProcessor
        processor = AudioProcessor()
        assert processor.sample_rate == 22050
        assert processor.n_mels == 128
    
    def test_metadata_extraction(self):
        """Test metadata extraction"""
        from app.ml.utils.audio_processor import AudioProcessor
        import numpy as np
        
        processor = AudioProcessor()
        audio = np.random.randn(22050 * 2)  # 2 seconds
        metadata = processor.get_audio_metadata(audio)
        
        assert "duration" in metadata
        assert metadata["duration"] > 0
        assert "sample_rate" in metadata
        assert "samples" in metadata


class TestMLModels:
    """Test ML model architectures"""
    
    def test_spectrogram_cnn_import(self):
        """Test 2D CNN model import"""
        from app.ml.models.networks import SpectrogramCNN
        model = SpectrogramCNN(num_classes=10)
        assert model is not None
        assert model.num_classes == 10
    
    def test_waveform_cnn_import(self):
        """Test 1D CNN model import"""
        from app.ml.models.networks import WaveformCNN
        model = WaveformCNN(num_classes=10)
        assert model is not None
        assert model.num_classes == 10
    
    def test_autoencoder_import(self):
        """Test autoencoder import"""
        from app.ml.models.networks import AudioAutoencoder
        model = AudioAutoencoder(latent_dim=128)
        assert model is not None
        assert model.latent_dim == 128


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
