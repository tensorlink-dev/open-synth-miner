"""Advanced blocks for recurrent, convolutional, frequency, and U-Net structures."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.registry import registry


# ---------------------------------------------------------------------------
# RNN / GRU Variants
# ---------------------------------------------------------------------------


@registry.register_block("rnnblock")
class RNNBlock(nn.Module):
    """Simple Elman RNN block."""

    def __init__(self, d_model: int, num_layers: int = 1, dropout: float = 0.0, nonlinearity: str = "tanh") -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            nonlinearity=nonlinearity,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return out


@registry.register_block("grublock")
class GRUBlock(nn.Module):
    """Gated Recurrent Unit (GRU) block."""

    def __init__(self, d_model: int, num_layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return out


# ---------------------------------------------------------------------------
# Convolutional & BiTCN Blocks
# ---------------------------------------------------------------------------


@registry.register_block("resconvblock")
class ResConvBlock(nn.Module):
    """1D residual convolutional block."""

    def __init__(self, d_model: int, kernel_size: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, padding=padding)
        self.norm1 = nn.BatchNorm1d(d_model)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, padding=padding)
        self.norm2 = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x.transpose(1, 2)

        y = self.activation(self.norm1(self.conv1(x_in)))
        y = self.dropout(y)
        y = self.norm2(self.conv2(y))

        return (x_in + y).transpose(1, 2)


@registry.register_block("bitcnblock")
class BiTCNBlock(nn.Module):
    """Bidirectional temporal convolutional block with dilation."""

    def __init__(self, d_model: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, padding=padding, dilation=dilation)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        out = self.conv(x_t)
        out = out.transpose(1, 2)

        return self.norm(x + self.dropout(self.activation(out)))


# ---------------------------------------------------------------------------
# Patching & Unet
# ---------------------------------------------------------------------------


@registry.register_block("patchembedding")
class PatchEmbedding(nn.Module):
    """Projects raw input sequence into ``d_model`` using strided convolutions."""

    def __init__(self, d_model: int, patch_size: int = 4, in_channels: int | None = None) -> None:
        super().__init__()
        dim = in_channels if in_channels is not None else d_model
        self.proj = nn.Conv1d(dim, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.proj(x)
        return x.transpose(1, 2)


@registry.register_block("unet1dblock")
class Unet1DBlock(nn.Module):
    """U-Net style block that preserves sequence length."""

    def __init__(self, d_model: int, reduction: int = 2) -> None:
        super().__init__()
        hidden = d_model * reduction
        self.down_conv = nn.Conv1d(d_model, hidden, kernel_size=3, stride=2, padding=1)
        self.bottleneck = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
        )
        self.up_conv = nn.ConvTranspose1d(hidden, d_model, kernel_size=4, stride=2, padding=1)
        self.norm = nn.LayerNorm(d_model)
        self.fusion = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x.transpose(1, 2)

        down = F.gelu(self.down_conv(x_in))
        neck = self.bottleneck(down)
        up = self.up_conv(neck)

        if up.shape[2] != x_in.shape[2]:
            up = F.interpolate(up, size=x_in.shape[2], mode="linear", align_corners=False)

        up = up.transpose(1, 2)
        combined = torch.cat([x, up], dim=-1)
        out = self.fusion(combined)
        return self.norm(out + x)


# ---------------------------------------------------------------------------
# Encoders / Decoders (Transformer Adapters)
# ---------------------------------------------------------------------------


@registry.register_block("transformerencoder")
class TransformerEncoderAdapter(nn.Module):
    """Wraps ``torch.nn.TransformerEncoder`` as a block."""

    def __init__(self, d_model: int, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 128) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


@registry.register_block("transformerdecoder")
class TransformerDecoderAdapter(nn.Module):
    """Wraps ``torch.nn.TransformerDecoder`` with self-conditioning memory."""

    def __init__(self, d_model: int, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 128) -> None:
        super().__init__()
        layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x, x)


# ---------------------------------------------------------------------------
# Frequency Domain (FedFormer-inspired)
# ---------------------------------------------------------------------------


@registry.register_block("fourierblock")
class FourierBlock(nn.Module):
    """Frequency enhanced block similar to FedFormer."""

    def __init__(self, d_model: int, modes: int = 32) -> None:
        super().__init__()
        self.modes = modes
        self.scale = 0.02
        self.complex_weight = nn.Parameter(torch.randn(d_model, modes, 2, dtype=torch.float32) * self.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        x_in = x.permute(0, 2, 1)

        x_fft = torch.fft.rfft(x_in, dim=-1, norm="ortho")

        eff_modes = min(self.modes, x_fft.shape[-1])
        weights = torch.view_as_complex(self.complex_weight[:, :eff_modes, :])

        res_fft = torch.zeros_like(x_fft)
        res_fft[:, :, :eff_modes] = x_fft[:, :, :eff_modes] * weights.unsqueeze(0)

        x_out = torch.fft.irfft(res_fft, n=seq_len, dim=-1, norm="ortho")

        return x_out.permute(0, 2, 1) + x


@registry.register_block("laststepadapter")
class LastStepAdapter(nn.Module):
    """Lightweight pooling adapter to align backbone outputs with heads."""

    def __init__(self, d_model: int, mode: str = "last") -> None:
        super().__init__()
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "mean":
            return x.mean(dim=1, keepdim=True)
        if self.mode == "max":
            return x.max(dim=1, keepdim=True).values
        return x
