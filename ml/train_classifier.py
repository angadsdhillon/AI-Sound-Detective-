"""
Training script for 2D CNN sound classifier
Trains on mel spectrogram representations
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

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.app.ml.models.networks import SpectrogramCNN, initialize_weights
from backend.app.ml.utils.audio_processor import AudioProcessor


class SpectrogramDataset(Dataset):
    """Dataset for loading audio spectrograms"""
    
    def __init__(
        self,
        data_dir: str,
        audio_processor: AudioProcessor,
        max_samples: int = None
    ):
        self.data_dir = data_dir
        self.audio_processor = audio_processor
        self.samples = []
        
        # Load dataset structure
        # Expected structure: data_dir/class_name/audio_file.wav
        if os.path.exists(data_dir):
            for class_idx, class_name in enumerate(sorted(os.listdir(data_dir))):
                class_path = os.path.join(data_dir, class_name)
                if os.path.isdir(class_path):
                    for audio_file in os.listdir(class_path):
                        if audio_file.endswith(('.wav', '.mp3', '.flac')):
                            self.samples.append({
                                'path': os.path.join(class_path, audio_file),
                                'label': class_idx,
                                'class_name': class_name
                            })
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        print(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Load audio
            audio, sr = self.audio_processor.load_audio(sample['path'])
            audio = self.audio_processor.apply_preprocessing(audio)
            
            # Extract mel spectrogram
            mel_spec = self.audio_processor.extract_mel_spectrogram(audio)
            
            # Normalize
            mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)
            
            # Convert to tensor
            spec_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0)
            
            return spec_tensor, sample['label']
        
        except Exception as e:
            print(f"Error loading {sample['path']}: {e}")
            # Return a zero tensor on error
            return torch.zeros((1, 128, 128)), 0


def train_classifier(
    train_dir: str,
    val_dir: str,
    output_path: str,
    num_classes: int = 10,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train the 2D CNN classifier
    
    Args:
        train_dir: Directory with training data
        val_dir: Directory with validation data
        output_path: Path to save trained model
        num_classes: Number of sound classes
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
    """
    print(f"Training on device: {device}")
    
    # Initialize audio processor
    audio_processor = AudioProcessor(
        sample_rate=22050,
        n_mels=128,
        n_fft=2048,
        hop_length=512
    )
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = SpectrogramDataset(train_dir, audio_processor)
    val_dataset = SpectrogramDataset(val_dir, audio_processor)
    
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
    print("Initializing model...")
    model = SpectrogramCNN(num_classes=num_classes)
    initialize_weights(model)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    # Training loop
    print("Starting training...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * train_correct / train_total:.2f}%'
            })
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path)
            print(f"✓ Model saved with validation accuracy: {val_acc:.2f}%")
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")
    
    # Plot training history
    plot_training_history(history)
    
    return model, history


def plot_training_history(history: dict):
    """Plot training metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('ml/models/training_history_2d.png')
    print("Training history plot saved to ml/models/training_history_2d.png")


if __name__ == "__main__":
    # Configuration
    TRAIN_DIR = "ml/datasets/train"
    VAL_DIR = "ml/datasets/val"
    OUTPUT_PATH = "ml/models/classifier_2d.pth"
    NUM_CLASSES = 10
    EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    
    # Create output directory
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Train model
    model, history = train_classifier(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        output_path=OUTPUT_PATH,
        num_classes=NUM_CLASSES,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
    
    print("\n✓ Training complete!")
