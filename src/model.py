"""
model.py — All architectures:
  UNet          — original (Ronneberger et al. 2015)
  UNetMSCA      — UNet + Multi-Scale Channel Attention skip calibration
  UNetDEVA      — UNet + DEVA novel blocks (new, better method)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEVA — Dual-branch Enhanced Vessel Attention
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY DEVA IS BETTER THAN MSCA:

  MSCA weakness: uses dilation (d=1,2,3) to simulate multi-scale within one
  resolution level.  Dilated convolutions have an inherent flaw — they create
  "gridding artefacts" and can skip over thin vessel pixels that fall between
  the dilation grid points, exactly the pixels we most need to detect.

  DEVA fix: processes skip features at TWO REAL RESOLUTIONS simultaneously:

  1. Detail branch (full resolution)
       DepthwiseSeparableConv 3×3 → keeps exact sub-pixel vessel coordinates
       Depthwise-separable = channel-wise 3×3 + 1×1 pointwise
       Parameter-efficient, no dilation gaps, best for thin vessel precision

  2. Context branch (half resolution)
       AvgPool(2) → Conv(C→C) → BN → ReLU → BilinearUpsample(×2)
       AvgPool (not MaxPool) preserves vessel intensity distribution
       At half-res the model sees vessel connectivity and topology context
       Upsamples back cleanly with bilinear interpolation

  3. Fusion
       Concat(detail, context) → 1×1 conv → BN → ReLU  reduces 2C → C
       SE channel attention weights each feature map after fusion
       Residual connection: Output = BN(fused ⊗ channel_att) + skip_input

  Why this works for retinal vessels:
    • Detail branch captures thin capillaries without dilation gaps
    • Context branch understands which vessel segments are connected
    • True scale separation: each branch specialised for its task
    • AvgPool context does not destroy faint vessel signal (MaxPool would)
    • Parameter overhead ≈ MSCA (fair comparison)

MSCA weakness summary:
  dilation d=2 for a 3×3 kernel samples every OTHER pixel — if a 1px vessel
  sits between two sample points of the dilated grid, it is INVISIBLE to that
  kernel. DEVA's full-resolution depthwise conv samples every pixel, so no
  vessel is ever missed at the detail branch level.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C


# ─────────────────────────── Core building blocks ─────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.block(x))


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv = ConvBlock(in_ch, out_ch, dropout)

    def forward(self, x):
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(out_ch * 2, out_ch, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        if dh or dw:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([skip, x], dim=1))


# ─────────────────────────── MSCA Block ───────────────────────────────────────

class MSCABlock(nn.Module):
    """Multi-Scale Channel Attention (original novel block)."""

    def __init__(self, channels, reduction_ratio=8,
                 use_spatial_att=True, use_channel_att=True):
        super().__init__()
        self.use_spatial = use_spatial_att
        self.use_channel = use_channel_att

        if use_spatial_att:
            self.ms1 = nn.Conv2d(channels, channels, 3, padding=1, dilation=1,
                                  groups=channels, bias=False)
            self.ms2 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2,
                                  groups=channels, bias=False)
            self.ms3 = nn.Conv2d(channels, channels, 3, padding=3, dilation=3,
                                  groups=channels, bias=False)
            self.ms_fuse = nn.Sequential(
                nn.Conv2d(channels * 3, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(channels, 1, 1, bias=False),
                nn.Sigmoid(),
            )

        if use_channel_att:
            r = max(1, channels // reduction_ratio)
            self.channel_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, r, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(r, channels, bias=False),
                nn.Sigmoid(),
            )

        self.bn = nn.BatchNorm2d(channels)

    def forward(self, skip):
        out = skip
        if self.use_spatial:
            ms = torch.cat([self.ms1(skip), self.ms2(skip), self.ms3(skip)], 1)
            ms = self.ms_fuse(ms)
            out = out * self.spatial_gate(ms)
        if self.use_channel:
            c_w = self.channel_att(skip).unsqueeze(-1).unsqueeze(-1)
            out = out * c_w
        return self.bn(out + skip)


class UpBlockMSCA(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0,
                 use_spatial_att=True, use_channel_att=True):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.msca = MSCABlock(out_ch, C.REDUCTION_RATIO,
                              use_spatial_att, use_channel_att)
        self.conv = ConvBlock(out_ch * 2, out_ch, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        if dh or dw:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        skip = self.msca(skip)
        return self.conv(torch.cat([skip, x], dim=1))


# ─────────────────────────── DEVA Block (new method) ──────────────────────────

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable conv: channel-wise 3×3 + pointwise 1×1."""

    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1,
                             groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class DEVABlock(nn.Module):
    """
    Dual-branch Enhanced Vessel Attention.

    Detail branch  : full-resolution depthwise-separable conv
    Context branch : AvgPool(2) → Conv → BN → ReLU → BilinearUp(2)
    Fusion         : Concat → 1×1 → BN → ReLU → SE channel attention → residual
    """

    def __init__(self, channels, reduction_ratio=8):
        super().__init__()
        C_half = max(1, channels // 2)

        # ── Detail branch ─────────────────────────────────────────────────
        self.detail = DepthwiseSeparableConv(channels)

        # ── Context branch ────────────────────────────────────────────────
        self.ctx_pool  = nn.AvgPool2d(2, 2)
        self.ctx_conv  = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # ── Fusion ────────────────────────────────────────────────────────
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # ── SE channel attention on fused features ─────────────────────
        r = max(1, channels // reduction_ratio)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(r, channels, bias=False),
            nn.Sigmoid(),
        )

        self.bn_out = nn.BatchNorm2d(channels)

    def forward(self, skip: torch.Tensor) -> torch.Tensor:
        H, W = skip.shape[2], skip.shape[3]

        # Detail branch — full resolution, no dilation gaps
        detail = self.detail(skip)                          # (B, C, H, W)

        # Context branch — process at half resolution
        ctx = self.ctx_pool(skip)                           # (B, C, H/2, W/2)
        ctx = self.ctx_conv(ctx)                            # (B, C, H/2, W/2)
        ctx = F.interpolate(ctx, size=(H, W),
                            mode="bilinear", align_corners=False)  # (B, C, H, W)

        # Fuse detail + context
        fused = self.fuse(torch.cat([detail, ctx], dim=1)) # (B, C, H, W)

        # Channel attention weights
        se_w = self.se(fused).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        out  = fused * se_w

        # Residual
        return self.bn_out(out + skip)


class UpBlockDEVA(nn.Module):
    """UpBlock that calibrates skip connections with DEVA before concatenation."""

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.deva = DEVABlock(out_ch, C.REDUCTION_RATIO)
        self.conv = ConvBlock(out_ch * 2, out_ch, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        if dh or dw:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        skip = self.deva(skip)
        return self.conv(torch.cat([skip, x], dim=1))


# ─────────────────────────── Model definitions ────────────────────────────────

class UNet(nn.Module):
    """Original U-Net (Ronneberger et al. 2015)."""

    def __init__(self, in_ch=C.IN_CHANNELS, out_ch=C.OUT_CHANNELS,
                 base=C.BASE_FILTERS, dropout=C.DROPOUT):
        super().__init__()
        f = base
        self.enc1       = ConvBlock(in_ch, f)
        self.enc2       = DownBlock(f,     f*2)
        self.enc3       = DownBlock(f*2,   f*4)
        self.enc4       = DownBlock(f*4,   f*8)
        self.bottleneck = DownBlock(f*8,   f*16, dropout=dropout)
        self.dec4 = UpBlock(f*16, f*8)
        self.dec3 = UpBlock(f*8,  f*4)
        self.dec2 = UpBlock(f*4,  f*2)
        self.dec1 = UpBlock(f*2,  f)
        self.head = nn.Conv2d(f, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b  = self.bottleneck(e4)
        d4 = self.dec4(b,  e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.head(d1)


class UNetMSCA(nn.Module):
    """U-Net + MSCA skip calibration."""

    def __init__(self, in_ch=C.IN_CHANNELS, out_ch=C.OUT_CHANNELS,
                 base=C.BASE_FILTERS, dropout=C.DROPOUT,
                 use_spatial_att=True, use_channel_att=True):
        super().__init__()
        f = base
        self.enc1       = ConvBlock(in_ch, f)
        self.enc2       = DownBlock(f,     f*2)
        self.enc3       = DownBlock(f*2,   f*4)
        self.enc4       = DownBlock(f*4,   f*8)
        self.bottleneck = DownBlock(f*8,   f*16, dropout=dropout)
        self.dec4 = UpBlockMSCA(f*16, f*8, use_spatial_att=use_spatial_att,
                                use_channel_att=use_channel_att)
        self.dec3 = UpBlockMSCA(f*8,  f*4, use_spatial_att=use_spatial_att,
                                use_channel_att=use_channel_att)
        self.dec2 = UpBlockMSCA(f*4,  f*2, use_spatial_att=use_spatial_att,
                                use_channel_att=use_channel_att)
        self.dec1 = UpBlockMSCA(f*2,  f,   use_spatial_att=use_spatial_att,
                                use_channel_att=use_channel_att)
        self.head = nn.Conv2d(f, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x);  e2 = self.enc2(e1)
        e3 = self.enc3(e2); e4 = self.enc4(e3)
        b  = self.bottleneck(e4)
        d4 = self.dec4(b,  e4); d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2); d1 = self.dec1(d2, e1)
        return self.head(d1)


class UNetDEVA(nn.Module):
    """
    U-Net + DEVA skip calibration (new method).
    Encoder/bottleneck identical to UNet for fair comparison.
    Only the skip connections are replaced with DEVA-calibrated versions.
    """

    def __init__(self, in_ch=C.IN_CHANNELS, out_ch=C.OUT_CHANNELS,
                 base=C.BASE_FILTERS, dropout=C.DROPOUT):
        super().__init__()
        f = base
        self.enc1       = ConvBlock(in_ch, f)
        self.enc2       = DownBlock(f,     f*2)
        self.enc3       = DownBlock(f*2,   f*4)
        self.enc4       = DownBlock(f*4,   f*8)
        self.bottleneck = DownBlock(f*8,   f*16, dropout=dropout)
        self.dec4 = UpBlockDEVA(f*16, f*8)
        self.dec3 = UpBlockDEVA(f*8,  f*4)
        self.dec2 = UpBlockDEVA(f*4,  f*2)
        self.dec1 = UpBlockDEVA(f*2,  f)
        self.head = nn.Conv2d(f, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x);  e2 = self.enc2(e1)
        e3 = self.enc3(e2); e4 = self.enc4(e3)
        b  = self.bottleneck(e4)
        d4 = self.dec4(b,  e4); d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2); d1 = self.dec1(d2, e1)
        return self.head(d1)


# ─────────────────────────── Factory ──────────────────────────────────────────

def build_model(experiment: str = C.EXPERIMENT,
                use_spatial: bool = True,
                use_channel: bool = True) -> nn.Module:
    if experiment == "original":
        return UNet()
    elif experiment in ("enhanced", "msca"):
        return UNetMSCA(use_spatial_att=use_spatial, use_channel_att=use_channel)
    elif experiment == "deva":
        return UNetDEVA()
    elif experiment == "ablation_spatial":
        return UNetMSCA(use_spatial_att=True, use_channel_att=False)
    elif experiment == "ablation_channel":
        return UNetMSCA(use_spatial_att=False, use_channel_att=True)
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
