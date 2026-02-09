"""Advanced blocks for recurrent, convolutional, frequency, and U-Net structures."""
from __future__ import annotations

from typing import List, Tuple

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


# ---------------------------------------------------------------------------
# Flexible Patching Architectures & Methods
# ---------------------------------------------------------------------------


@registry.register_block("revin")
class RevIN(nn.Module):
    """Reversible instance normalization for non-stationary sequences."""

    def __init__(self, d_model: int, affine: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        self.d_model = d_model
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.weight = nn.Parameter(torch.ones(d_model))
            self.bias = nn.Parameter(torch.zeros(d_model))

        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("stdev", torch.ones(1))

    def forward(self, x: torch.Tensor, mode: str = "norm") -> torch.Tensor:
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x_norm = (x - self.mean) / self.stdev
            if self.affine:
                x_norm = x_norm * self.weight + self.bias
            return x_norm
        if mode == "denorm":
            x_denorm = x
            if self.affine:
                x_denorm = (x_denorm - self.bias) / (self.weight + self.eps)
            return x_denorm * self.stdev + self.mean
        raise ValueError(f"Unknown RevIN mode: {mode}")


@registry.register_block("flexiblepatchembed")
class FlexiblePatchEmbed(nn.Module):
    """Patch embedding supporting channel-independence and masking."""

    def __init__(
        self,
        d_model: int,
        patch_len: int = 16,
        stride: int = 8,
        in_channels: int = 1,
        channel_independence: bool = True,
        mask_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.channel_independence = channel_independence
        self.mask_ratio = mask_ratio

        self.input_dim = 1 if channel_independence else in_channels
        self.proj = nn.Conv1d(
            in_channels=self.input_dim,
            out_channels=d_model,
            kernel_size=patch_len,
            stride=stride,
            padding=0,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.shape

        if self.channel_independence:
            x = x.permute(0, 2, 1).reshape(batch * channels, seq_len, 1)
        else:
            if channels != self.input_dim and self.input_dim != 1:
                # fall back to passthrough if misconfigured
                return x

        pad_len = 0
        if self.stride > 0:
            pad_len = max(0, self.patch_len - seq_len)
            remainder = (seq_len + pad_len - self.patch_len) % self.stride
            if remainder != 0:
                pad_len += self.stride - remainder
            x = x.permute(0, 2, 1)
            x = F.pad(x, (0, pad_len), mode="replicate")
        else:
            x = x.permute(0, 2, 1)

        x_patched = self.proj(x).transpose(1, 2)

        if self.training and self.mask_ratio > 0:
            x_patched, _ = self._apply_mask(x_patched)

        return self.norm(x_patched)

    def _apply_mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, length, d_model = x.shape
        len_keep = int(length * (1 - self.mask_ratio))

        noise = torch.rand(batch, length, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = x.clone()
        mask = torch.ones([batch, length], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        mask_expanded = mask.unsqueeze(-1).bool()
        x_masked[mask_expanded] = 0.0

        return x_masked, mask


@registry.register_block("channelrejoin")
class ChannelRejoin(nn.Module):
    """Reshape channel-independent batches back to original structure."""

    def __init__(self, num_channels: int, mode: str = "flatten") -> None:
        super().__init__()
        self.num_channels = num_channels
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_channels, seq_len, d_model = x.shape
        if batch_channels % self.num_channels != 0:
            return x

        batch = batch_channels // self.num_channels
        x = x.view(batch, self.num_channels, seq_len, d_model)

        if self.mode == "flatten":
            return x.permute(0, 2, 1, 3).reshape(batch, seq_len, self.num_channels * d_model)
        if self.mode == "stack":
            return x
        return x


@registry.register_block("multiscalepatcher")
class MultiScalePatcher(nn.Module):
    """Apply multiple patch sizes in parallel and fuse them."""

    def __init__(
        self,
        d_model: int,
        scales: List[int] | None = None,
        channel_independence: bool = True,
    ) -> None:
        super().__init__()
        self.scales = scales or [4, 16]
        self.branches = nn.ModuleList(
            [
                FlexiblePatchEmbed(
                    d_model=d_model,
                    patch_len=scale,
                    stride=scale,
                    channel_independence=channel_independence,
                )
                for scale in self.scales
            ]
        )
        self.fusion = nn.Linear(d_model * len(self.scales), d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [branch(x) for branch in self.branches]
        max_len = max(out.shape[1] for out in outs)

        aligned = []
        for out in outs:
            if out.shape[1] != max_len:
                out_t = out.transpose(1, 2)
                out_up = F.interpolate(out_t, size=max_len, mode="linear", align_corners=False)
                aligned.append(out_up.transpose(1, 2))
            else:
                aligned.append(out)

        concatenated = torch.cat(aligned, dim=-1)
        return self.fusion(concatenated)


# ---------------------------------------------------------------------------
# DLinear (trend-seasonal decomposition + linear)
# ---------------------------------------------------------------------------


@registry.register_block("dlinearblock")
class DLinearBlock(nn.Module):
    """DLinear block: moving-average decomposition into trend + seasonal, each with a linear layer.

    Based on "Are Transformers Effective for Time Series Forecasting?" (Zeng et al., 2023).
    Preserves the ``(batch, seq, d_model)`` shape contract.
    """

    def __init__(self, d_model: int, kernel_size: int = 25) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=padding)
        self.linear_trend = nn.Linear(d_model, d_model)
        self.linear_seasonal = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        # Decompose along the feature dim per time step
        x_t = x.transpose(1, 2)  # (batch, d_model, seq_len)
        trend = self.avg_pool(x_t).transpose(1, 2)  # (batch, seq_len, d_model)
        # Handle edge case where avg_pool changes length
        if trend.shape[1] != x.shape[1]:
            trend = F.interpolate(
                trend.transpose(1, 2), size=x.shape[1], mode="linear", align_corners=False
            ).transpose(1, 2)
        seasonal = x - trend

        trend_out = self.linear_trend(trend)
        seasonal_out = self.linear_seasonal(seasonal)
        return trend_out + seasonal_out


# ---------------------------------------------------------------------------
# TimesNet (temporal 2D-variation modeling)
# ---------------------------------------------------------------------------


class _InceptionBlock2D(nn.Module):
    """Simplified Inception-style 2D convolution used inside TimesNetBlock."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
        )
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        return self.norm(b1 + b3 + b5)


@registry.register_block("timesnetblock")
class TimesNetBlock(nn.Module):
    """TimesNet block: discovers dominant periods via FFT, reshapes into 2D, and applies Inception convolutions.

    Based on "TimesNet: Temporal 2D-Variation Modeling" (Wu et al., 2023).
    Preserves the ``(batch, seq, d_model)`` shape contract.
    """

    def __init__(self, d_model: int, top_k: int = 3, d_ff: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.top_k = top_k
        d_ff = d_ff or d_model
        self.conv2d = _InceptionBlock2D(d_model, d_ff)
        self.proj_back = nn.Linear(d_ff, d_model) if d_ff != d_model else nn.Identity()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _period_discovery(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Use FFT to find the top-k dominant periods."""
        # x: (batch, seq_len, d_model)
        xf = torch.fft.rfft(x.mean(dim=-1), dim=-1)  # (batch, freq)
        amplitude = xf.abs()
        # Exclude DC component (index 0)
        amplitude[:, 0] = 0
        eff_k = min(self.top_k, amplitude.shape[-1] - 1)
        top_values, top_indices = torch.topk(amplitude, eff_k, dim=-1)
        # Convert frequency indices to period lengths
        periods = (x.shape[1] / (top_indices.float() + 1e-6)).clamp(min=2).long()
        weights = F.softmax(top_values, dim=-1)
        return periods, weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        periods, weights = self._period_discovery(x)

        accumulated = torch.zeros_like(x)
        for k in range(periods.shape[-1]):
            # Use the median period across the batch for stable reshaping
            period = int(periods[:, k].float().median().item())
            period = max(2, min(period, seq_len))

            # Pad sequence so it's evenly divisible by the period
            n_segments = (seq_len + period - 1) // period
            pad_len = n_segments * period - seq_len
            x_pad = F.pad(x.transpose(1, 2), (0, pad_len), mode="replicate").transpose(1, 2)

            # Reshape to 2D: (batch, d_model, n_segments, period)
            x_2d = x_pad.transpose(1, 2).reshape(batch, d_model, n_segments, period)

            # 2D convolution
            out_2d = F.gelu(self.conv2d(x_2d))
            # Reshape back: (batch, d_ff, n_segments * period) -> trim -> (batch, seq_len, d_ff)
            out_1d = out_2d.reshape(batch, -1, n_segments * period)[:, :, :seq_len].transpose(1, 2)
            out_1d = self.proj_back(out_1d)

            # Weight by the period's importance
            w = weights[:, k].unsqueeze(-1).unsqueeze(-1)  # (batch, 1, 1)
            accumulated = accumulated + out_1d * w

        return self.norm(x + self.dropout(accumulated))


@registry.register_block("patchmixerblock")
class PatchMixerBlock(nn.Module):
    """MLP-Mixer style block for patched time series."""

    def __init__(self, d_model: int, seq_len: int, patch_len: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_patches = seq_len // patch_len

        self.norm_time = nn.LayerNorm(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.num_patches, self.num_patches),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.num_patches, self.num_patches),
        )

        self.norm_feat = nn.LayerNorm(d_model)
        self.feat_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm_time(x)
        x_time = x_norm.transpose(1, 2)
        if x_time.shape[2] == self.time_mlp[0].in_features:
            x_time = self.time_mlp(x_time)
        x = x_time.transpose(1, 2) + residual

        residual = x
        x = self.norm_feat(x)
        x = self.feat_mlp(x)
        return x + residual
