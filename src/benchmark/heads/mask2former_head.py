"""Compact Mask2Former-style fusion head for multi-scale features (pathocellbench-only)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class Mask2FormerHead(nn.Module):
    """Lightweight Mask2Former-style multi-scale fusion head.

    This is intentionally compact for pathocellbench integration and keeps the
    output API aligned with existing heads: logits (B, num_classes, H, W).
    """

    def __init__(self, in_channels: int = 256, hidden_dim: int = 256, num_classes: int = 15):
        super().__init__()
        self.proj4 = _ConvBlock(in_channels, hidden_dim)
        self.proj3 = _ConvBlock(in_channels, hidden_dim)
        self.proj2 = _ConvBlock(in_channels, hidden_dim)
        self.proj1 = _ConvBlock(in_channels, hidden_dim)
        self.fuse3 = _ConvBlock(hidden_dim * 2, hidden_dim)
        self.fuse2 = _ConvBlock(hidden_dim * 2, hidden_dim)
        self.fuse1 = _ConvBlock(hidden_dim * 2, hidden_dim)
        self.cls = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, feats: list[torch.Tensor], out_size: tuple[int, int]) -> torch.Tensor:
        # feats are expected in [1/4, 1/8, 1/16, 1/32] order.
        f1, f2, f3, f4 = feats
        p4 = self.proj4(f4)
        p3 = self.proj3(f3)
        p2 = self.proj2(f2)
        p1 = self.proj1(f1)

        u3 = F.interpolate(p4, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        h3 = self.fuse3(torch.cat([u3, p3], dim=1))
        u2 = F.interpolate(h3, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        h2 = self.fuse2(torch.cat([u2, p2], dim=1))
        u1 = F.interpolate(h2, size=p1.shape[-2:], mode="bilinear", align_corners=False)
        h1 = self.fuse1(torch.cat([u1, p1], dim=1))
        logits = self.cls(h1)
        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)

