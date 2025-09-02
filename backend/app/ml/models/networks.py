"""Neural network models for sound classification and analysis"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SpectrogramCNN(nn.Module):
    """
    2D CNN for spectrogram-based sound classification
    Industry-standard architecture with residual connections
    """
    
    def __init__(self, num_classes: int = 10, input_channels: int = 1):
        super(SpectrogramCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Convolutional blocks with batch normalization
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, channels, height, width)
            
        Returns:
            Class logits (batch, num_classes)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions"""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


class WaveformCNN(nn.Module):
    """
    1D CNN for raw waveform classification
    Processes audio signals directly without spectrograms
    """
    
    def __init__(self, num_classes: int = 10, sample_rate: int = 22050):
        super(WaveformCNN, self).__init__()
        
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        
        # 1D Convolutional blocks
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=80, stride=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input waveform (batch, 1, samples)
            
        Returns:
            Class logits (batch, num_classes)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions"""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


class AudioAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for audio fingerprinting and anomaly detection
    Learns compressed representations of audio spectrograms
    """
    
    def __init__(self, latent_dim: int = 128):
        super(AudioAutoencoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: (batch, 1, 128, time_frames)
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Adaptive pooling to fixed size
        self.encoder_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Bottleneck
        self.fc_encode = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation
        
        Args:
            x: Input spectrogram (batch, 1, height, width)
            
        Returns:
            Latent vector (batch, latent_dim)
        """
        x = self.encoder(x)
        x = self.encoder_pool(x)
        x = x.view(x.size(0), -1)
        z = self.fc_encode(x)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to spectrogram
        
        Args:
            z: Latent vector (batch, latent_dim)
            
        Returns:
            Reconstructed spectrogram (batch, 1, height, width)
        """
        x = self.fc_decode(z)
        x = x.view(x.size(0), 256, 4, 4)
        x = self.decoder(x)
        return x
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder
        
        Args:
            x: Input spectrogram
            
        Returns:
            Tuple of (reconstruction, latent_vector)
        """
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, z
    
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction error for anomaly detection
        
        Args:
            x: Input spectrogram
            
        Returns:
            Reconstruction error per sample
        """
        reconstruction, _ = self.forward(x)
        error = F.mse_loss(reconstruction, x, reduction='none')
        error = error.view(error.size(0), -1).mean(dim=1)
        return error


def initialize_weights(model: nn.Module):
    """Initialize model weights using best practices"""
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
