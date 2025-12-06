import torch
import torch.nn as nn
import torch.nn.functional as F
from model.activations import Snake


class AntiAliasedUpsampleVocoderUpsampleBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        upsample_factor, 
        kernel_size=7,
        lowpass_learnable=False,
        num_taps=12
    ):
        super().__init__()
        self.upsample_factor = upsample_factor
        self.in_channels = in_channels
        
        # Lowpass filter
        lowpass = self._make_lowpass(upsample_factor, num_taps)
        if lowpass_learnable:
            # Initialize with sinc, but allow learning
            self.lowpass = nn.Parameter(lowpass)
        else:
            self.register_buffer('lowpass', lowpass)
        
        self.lowpass_padding = num_taps // 2
        

        # Channel projection
        self.conv =  nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)

        # init before weight norm
        self._init_weights()

        self.conv = nn.utils.weight_norm(self.conv)

        self.act = Snake(out_channels)

    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def _make_lowpass(self, factor, num_taps=12):
        """Windowed sinc lowpass filter."""
        cutoff = 1.0 / factor
        t = torch.arange(-num_taps // 2, num_taps // 2 + 1, dtype=torch.float32)
        
        # Handle t=0 case for sinc
        sinc = torch.where(
            t == 0,
            torch.tensor(2.0 * cutoff),
            torch.sin(2 * torch.pi * cutoff * t) / (torch.pi * t)
        )
        
        # Hann window
        window = torch.hann_window(len(t))
        kernel = sinc * window
        kernel = kernel / kernel.sum()
        
        return kernel.view(1, 1, -1)
    
    def forward(self, x):
        B, C, T = x.shape
        
        # Zero-insert upsample
        x = x.unsqueeze(-1)
        x = F.pad(x, (0, self.upsample_factor - 1))
        x = x.reshape(B, C, T * self.upsample_factor)
        
        # Lowpass filter (depthwise)
        # Expand kernel for all channels
        kernel = self.lowpass.expand(C, 1, -1)
        if not self.lowpass.requires_grad:
            kernel = kernel.to(x.device)
        
        x = F.conv1d(x, kernel, padding=self.lowpass_padding, groups=C)
        
        # Gain correction (upsampling reduces energy)
        x = x * self.upsample_factor
        
        # Channel projection
        x = self.conv(x)
        
        x = self.act(x)

        return x