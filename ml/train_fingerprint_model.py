"""
Training script for audio fingerprint model
Builds a searchable index of audio fingerprints
"""

import torch
import numpy as np
import pickle
import os
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from backend.app.ml.models.networks import AudioAutoencoder
from backend.app.ml.utils.audio_processor import AudioProcessor


def build_fingerprint_database(
    data_dir: str,
    autoencoder_path: str,
    output_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Build audio fingerprint database
    
    Args:
        data_dir: Directory with audio files
        autoencoder_path: Path to trained autoencoder
        output_path: Path to save fingerprint index
        device: Device to use
    """
    print(f"Building fingerprint database on device: {device}")
    
    # Initialize audio processor
    audio_processor = AudioProcessor(
        sample_rate=22050,
        n_mels=128,
        n_fft=2048,
        hop_length=512
    )
    
    # Load autoencoder
    print("Loading autoencoder...")
    autoencoder = AudioAutoencoder(latent_dim=128)
    
    if os.path.exists(autoencoder_path):
        autoencoder.load_state_dict(
            torch.load(autoencoder_path, map_location=device)
        )
        autoencoder.eval()
        autoencoder.to(device)
        print("✓ Autoencoder loaded")
    else:
        print(f"Warning: Autoencoder not found at {autoencoder_path}")
        print("Creating fingerprint database with random autoencoder...")
    
    # Collect all audio files
    audio_files = []
    if os.path.exists(data_dir):
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(('.wav', '.mp3', '.flac')):
                    audio_files.append(os.path.join(root, file))
    
    print(f"Found {len(audio_files)} audio files")
    
    # Build database
    fingerprint_db = []
    
    for audio_path in tqdm(audio_files, desc="Generating fingerprints"):
        try:
            # Load audio
            audio, sr = audio_processor.load_audio(audio_path, duration=4.0)
            audio = audio_processor.apply_preprocessing(audio)
            
            # Extract mel spectrogram
            mel_spec = audio_processor.extract_mel_spectrogram(audio)
            
            # Resize to (128, 128)
            from scipy.ndimage import zoom
            target_shape = (128, 128)
            if mel_spec.shape != target_shape:
                zoom_factors = (
                    target_shape[0] / mel_spec.shape[0],
                    target_shape[1] / mel_spec.shape[1]
                )
                mel_spec = zoom(mel_spec, zoom_factors, order=1)
            
            # Normalize
            mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)
            mel_spec = mel_spec * 2 - 1
            
            # Convert to tensor
            spec_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0).unsqueeze(0)
            spec_tensor = spec_tensor.to(device)
            
            # Generate fingerprint
            with torch.no_grad():
                fingerprint = autoencoder.encode(spec_tensor)
                fingerprint = fingerprint.cpu().numpy()[0]
            
            # Get metadata
            filename = os.path.basename(audio_path)
            category = os.path.basename(os.path.dirname(audio_path))
            
            # Add to database
            fingerprint_db.append({
                'id': f"sound_{len(fingerprint_db)}",
                'name': filename,
                'category': category,
                'path': audio_path,
                'fingerprint': fingerprint
            })
        
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
    
    print(f"\n✓ Generated {len(fingerprint_db)} fingerprints")
    
    # Save database
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(fingerprint_db, f)
    
    print(f"✓ Fingerprint database saved to {output_path}")
    
    return fingerprint_db


def test_fingerprint_search(fingerprint_db: list, num_samples: int = 5):
    """Test fingerprint similarity search"""
    print(f"\nTesting fingerprint search with {num_samples} samples...")
    
    for i in range(min(num_samples, len(fingerprint_db))):
        query = fingerprint_db[i]
        query_fp = query['fingerprint']
        
        # Find similar sounds
        similarities = []
        for item in fingerprint_db:
            if item['id'] != query['id']:
                db_fp = item['fingerprint']
                similarity = np.dot(query_fp, db_fp) / (
                    np.linalg.norm(query_fp) * np.linalg.norm(db_fp) + 1e-8
                )
                similarities.append((item['name'], similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nQuery: {query['name']} ({query['category']})")
        print("Top 3 similar sounds:")
        for name, sim in similarities[:3]:
            print(f"  - {name}: {sim:.4f}")


if __name__ == "__main__":
    # Configuration
    DATA_DIR = "ml/datasets/train"
    AUTOENCODER_PATH = "ml/models/autoencoder.pth"
    OUTPUT_PATH = "ml/models/fingerprint_index.pkl"
    
    # Build database
    fingerprint_db = build_fingerprint_database(
        data_dir=DATA_DIR,
        autoencoder_path=AUTOENCODER_PATH,
        output_path=OUTPUT_PATH
    )
    
    # Test search
    if fingerprint_db:
        test_fingerprint_search(fingerprint_db)
    
    print("\n✓ Fingerprint database creation complete!")
