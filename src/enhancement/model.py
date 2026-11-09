"""Lightweight reference network extracted from the unified training source."""
from __future__ import annotations
from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F

class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.adaptive_avg_pool2d(x, 1)
        s = self.act(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, act: nn.Module | None = None) -> None:
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1,
                            groups=in_ch, bias=True)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True)
        self.act = act if act is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.pw(self.dw(x)))


class EnhanceNetLight(nn.Module):
    """
    Lightweight enhancement network consistent with the manuscript description.

    Forward returns:
      - enhanced_image_1: after curve_steps//2 iterations
      - enhanced_image: after curve_steps iterations
      - A: 3-channel parameter map (tanh bounded)
    """

    def __init__(self, number_f: int = 16, curve_steps: int = 8, se_reduction: int = 4) -> None:
        super().__init__()
        assert curve_steps >= 2, "curve_steps should be >=2"
        self.number_f = number_f
        self.curve_steps = curve_steps
        self.act = nn.SiLU(inplace=True)

        # Stem: expand to 16 channels (paper: "first processed through a convolutional layer")
        self.stem = nn.Conv2d(3, number_f, kernel_size=3, stride=1, padding=1, bias=True)

        # Encoder Groups 1-4
        self.g1 = DepthwiseSeparableConv(number_f, number_f, act=self.act)
        self.g2 = DepthwiseSeparableConv(number_f, number_f, act=self.act)
        self.g3 = DepthwiseSeparableConv(number_f, number_f, act=self.act)
        self.g4 = DepthwiseSeparableConv(number_f, number_f, act=self.act)
        self.se4 = SqueezeExcite(number_f, reduction=se_reduction)

        # Decoder Groups 5-7 with skip concatenations
        self.g5 = DepthwiseSeparableConv(2 * number_f, number_f, act=self.act)
        self.se5 = SqueezeExcite(number_f, reduction=se_reduction)

        self.g6 = DepthwiseSeparableConv(2 * number_f, number_f, act=self.act)
        self.se6 = SqueezeExcite(number_f, reduction=se_reduction)

        self.g7 = DepthwiseSeparableConv(2 * number_f, number_f, act=self.act)

        # Terminal 1×1 conv produces 3-channel A
        self.out = nn.Conv2d(number_f, 3, kernel_size=1, stride=1, padding=0, bias=True)

    @staticmethod
    def _apply_le_curve(x: torch.Tensor, A: torch.Tensor, steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Iterative curve: x <- x + A*(x^2 - x)
        out = x
        out_mid = None
        mid_step = steps // 2
        for i in range(steps):
            out = out + A * (out * out - out)
            out = torch.clamp(out, 0.0, 1.0)
            if i + 1 == mid_step:
                out_mid = out
        if out_mid is None:
            out_mid = out
        return out_mid, out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encoder
        x0 = self.act(self.stem(x))
        x1 = self.g1(x0)
        x2 = self.g2(x1)
        x3 = self.g3(x2)
        x4 = self.se4(self.g4(x3))

        # Decoder with skip-concat
        x5 = self.se5(self.g5(torch.cat([x3, x4], dim=1)))
        x6 = self.se6(self.g6(torch.cat([x2, x5], dim=1)))
        x7 = self.g7(torch.cat([x1, x6], dim=1))

        A = torch.tanh(self.out(x7))  # [-1,1]
        enhance_1, enhance = self._apply_le_curve(x, A, steps=self.curve_steps)
        return enhance_1, enhance, A
