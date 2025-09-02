"""
Training script for audio autoencoder
Used for audio fingerprinting and anomaly detection
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from backend.app.ml.models.networks import AudioAutoencoder, initialize_weights
from backend.app.ml.utils.audio_processor import AudioProcessor


class AudioSpectrogramDataset(Dataset):
    """Dataset for loading audio spectrograms for autoencoder training"""
    
    def __init__(
        self,
        data_dir: str,
        audio_processor: AudioProcessor,
        target_shape: tuple = (128, 128)
    ):
        self.data_dir = data_dir
        self.audio_processor = audio_processor
        self.target_shape = target_shape
        self.audio_files = []
        
        # Collect all audio files
        if os.path.exists(data_dir):
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if file.endswith(('.wav', '.mp3', '.flac')):
                        self.audio_files.append(os.path.join(root, file))
        
        print(f"Loaded {len(self.audio_files)} audio files")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        
        try:
            # Load audio
            audio, sr = self.audio_processor.load_audio(audio_path, duration=4.0)
            audio = self.audio_processor.apply_preprocessing(audio)
            
            # Extract mel spectrogram
            mel_spec = self.audio_processor.extract_mel_spectrogram(audio)
            
            # Resize to target shape
            from scipy.ndimage import zoom
            if mel_spec.shape != self.target_shape:
                zoom_factors = (
                    self.target_shape[0] / mel_spec.shape[0],
                    self.target_shape[1] / mel_spec.shape[1]
                )
                mel_spec = zoom(mel_spec, zoom_factors, order=1)
            
            # Normalize to [-1, 1]
            mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)
            mel_spec = mel_spec * 2 - 1
            
            # Convert to tensor
            spec_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0)
            
            return spec_tensor
        
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return torch.zeros((1, *self.target_shape))


def train_autoencoder(
    train_dir: str,
    val_dir: str,
    output_path: str,
    latent_dim: int = 128,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train the audio autoencoder
    
    Args:
        train_dir: Directory with training data
        val_dir: Directory with validation data
        output_path: Path to save trained model
        latent_dim: Latent dimension size
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
    """
    print(f"Training autoencoder on device: {device}")
    
    # Initialize audio processor
    audio_processor = AudioProcessor(
        sample_rate=22050,
        n_mels=128,
        n_fft=2048,
        hop_length=512
    )
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = AudioSpectrogramDataset(train_dir, audio_processor)
    val_dataset = AudioSpectrogramDataset(val_dir, audio_processor)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device == "cuda" else False
    )
    
    # Initialize model
    print("Initializing autoencoder...")
    model = AudioAutoencoder(latent_dim=latent_dim)
    initialize_weights(model)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    best_val_loss = float('inf')
    
    # Training loop
    print("Starting training...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for spectrograms in pbar:
            spectrograms = spectrograms.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            reconstructions, _ = model(spectrograms)
            loss = criterion(reconstructions, spectrograms)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for spectrograms in val_loader:
                spectrograms = spectrograms.to(device)
                reconstructions, _ = model(spectrograms)
                loss = criterion(reconstructions, spectrograms)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_path)
            print(f"✓ Model saved with validation loss: {val_loss:.6f}")
        
        # Save sample reconstructions every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_reconstruction_samples(model, val_loader, device, epoch + 1)
    
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.6f}")
    
    # Plot training history
    plot_autoencoder_history(history)
    
    return model, history


def save_reconstruction_samples(model, dataloader, device, epoch):
    """Save sample reconstructions for visualization"""
    model.eval()
    
    # Get one batch
    spectrograms = next(iter(dataloader))
    spectrograms = spectrograms.to(device)
    
    with torch.no_grad():
        reconstructions, _ = model(spectrograms)
    
    # Plot first 4 samples
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i in range(4):
        original = spectrograms[i].cpu().numpy()[0]
        reconstructed = reconstructions[i].cpu().numpy()[0]
        
        axes[0, i].imshow(original, aspect='auto', origin='lower', cmap='viridis')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(reconstructed, aspect='auto', origin='lower', cmap='viridis')
        axes[1, i].set_title(f'Reconstructed {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'ml/models/reconstruction_epoch_{epoch}.png')
    plt.close()


def plot_autoencoder_history(history: dict):
    """Plot training metrics"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Autoencoder Training History')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ml/models/autoencoder_history.png')
    print("Training history plot saved to ml/models/autoencoder_history.png")


if __name__ == "__main__":
    # Configuration
    TRAIN_DIR = "ml/datasets/train"
    VAL_DIR = "ml/datasets/val"
    OUTPUT_PATH = "ml/models/autoencoder.pth"
    LATENT_DIM = 128
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    
    # Create output directory
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Train model
    model, history = train_autoencoder(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        output_path=OUTPUT_PATH,
        latent_dim=LATENT_DIM,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
    
    print("\n✓ Autoencoder training complete!")
