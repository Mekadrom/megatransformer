import torch
import torch.nn as nn
import torch.nn.functional as F

from model.audio.shared_window_buffer import SharedWindowBuffer


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt-style block - efficient and effective for audio.
    Depthwise conv -> pointwise expand -> pointwise contract
    """
    
    def __init__(self, dim, kernel_size=7, expansion=4):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size, 
            padding=kernel_size // 2, 
            groups=dim  # Depthwise
        )
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, dim * expansion)  # Pointwise expand
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dim * expansion, dim)  # Pointwise contract
    
    def forward(self, x):
        # x: [B, C, T]
        residual = x
        
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # [B, T, C] for LayerNorm
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.transpose(1, 2)  # [B, C, T]
        
        return residual + x


class FrequencyDomainVocoder(nn.Module):
    """
    Predicts complex STFT (magnitude + phase) from mel spectrogram,
    then uses iSTFT for lossless waveform reconstruction.
    
    No upsampling needed - mel frames and STFT frames have same time resolution.
    """
    
    def __init__(
        self,
        shared_window_buffer: SharedWindowBuffer,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 8,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.freq_bins = n_fft // 2 + 1  # 513 for n_fft=1024
        
        # Input projection: mel bins -> hidden
        self.input_proj = nn.Conv1d(n_mels, hidden_dim, kernel_size=7, padding=3)
        
        # Backbone - no upsampling, just channel transformations
        self.backbone = nn.ModuleList([
            ConvNeXtBlock(hidden_dim) for _ in range(num_layers)
        ])
        
        # Magnitude head - must output positive values
        self.mag_head = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden_dim, self.freq_bins, kernel_size=7, padding=3),
            nn.Softplus(),  # Ensures positive magnitude
        )
        
        # Phase head - outputs (real, imag) of unit phasor
        # Predicting unit phasor avoids phase wrapping issues
        self.phase_head = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden_dim, self.freq_bins * 2, kernel_size=7, padding=3),
        )
        
        # iSTFT window
        self.register_buffer('window', shared_window_buffer.get_window(n_fft))
        
        self._init_weights()
    
    def _init_weights(self):
        # Input projection
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)
        
        # Magnitude head - small init so initial output is small positive
        for m in self.mag_head:
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Phase head - small init for stable start
        for m in self.phase_head:
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, mel):
        """
        Args:
            mel: [B, n_mels, T] log mel spectrogram
            
        Returns:
            waveform: [B, T * hop_length]
            stft: [B, freq_bins, T] complex - for loss computation
        """
        # Backbone
        x = self.input_proj(mel)
        for block in self.backbone:
            x = x + block(x)  # Residual
        
        # Predict magnitude: [B, 513, T]
        mag = self.mag_head(x)
        
        # Predict phase as unit phasor (real, imag): [B, 1026, T]
        phase_ri = self.phase_head(x)
        phase_real, phase_imag = phase_ri.chunk(2, dim=1)  # [B, 513, T] each
        
        # Normalize to unit circle - ensures valid phase representation
        norm = torch.sqrt(phase_real**2 + phase_imag**2).clamp(min=1e-8)
        phase_real = phase_real / norm
        phase_imag = phase_imag / norm
        
        # Construct complex STFT: magnitude * e^(i*phase)
        stft_real = mag * phase_real
        stft_imag = mag * phase_imag
        stft = torch.complex(stft_real, stft_imag)
        
        # iSTFT to waveform - this is pure math, lossless
        waveform = torch.istft(
            stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(stft.device),
            return_complex=False,
        )
        
        return waveform
    
    def get_phase(self, stft):
        """Extract phase angle from predicted STFT for loss computation."""
        return torch.angle(stft)
    
    def get_magnitude(self, stft):
        """Extract magnitude from predicted STFT for loss computation."""
        return stft.abs()
