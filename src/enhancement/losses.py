"""Six zero-reference losses; formulas preserved from the unified source."""
import torch
from torch import nn
from torch.nn import functional as F

def rgb_to_hsv(rgb: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Stable RGB->HSV for tensors in [0,1], shape [B,3,H,W].
    Hue in [0,1], Saturation in [0,1], Value in [0,1].
    """
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]  # [B,H,W]
    maxc, _ = rgb.max(dim=1)
    minc, _ = rgb.min(dim=1)
    delta = maxc - minc

    v = maxc
    s = delta / (maxc + eps)

    # Hue
    h = torch.zeros_like(maxc)
    # Avoid division by zero via eps
    delta_safe = delta + eps

    mask = delta > eps
    # When max == r
    idx = (maxc == r) & mask
    h[idx] = ((g[idx] - b[idx]) / delta_safe[idx]) % 6.0
    # When max == g
    idx = (maxc == g) & mask
    h[idx] = ((b[idx] - r[idx]) / delta_safe[idx]) + 2.0
    # When max == b
    idx = (maxc == b) & mask
    h[idx] = ((r[idx] - g[idx]) / delta_safe[idx]) + 4.0

    h = (h / 6.0) % 1.0  # normalize to [0,1)
    return torch.stack([h, s, v], dim=1)


class L_color(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Color constancy (mean channel distance)
        mean_rgb = torch.mean(x, dim=(2, 3), keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        drg = (mr - mg).pow(2)
        drb = (mr - mb).pow(2)
        dgb = (mb - mg).pow(2)
        return torch.sqrt(drg.pow(2) + drb.pow(2) + dgb.pow(2) + 1e-12)


class L_spa(nn.Module):
    """
    Spatial consistency loss:
    Penalize differences between local gradients of pooled grayscale maps.
    """
    def __init__(self) -> None:
        super().__init__()
        # 3x3 directional kernels (registered as buffers; device follows the module)
        k_left = torch.tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        k_right = torch.tensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        k_up = torch.tensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        k_down = torch.tensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer("k_left", k_left)
        self.register_buffer("k_right", k_right)
        self.register_buffer("k_up", k_up)
        self.register_buffer("k_down", k_down)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org: torch.Tensor, enhance: torch.Tensor) -> torch.Tensor:
        org_mean = torch.mean(org, dim=1, keepdim=True)
        enhance_mean = torch.mean(enhance, dim=1, keepdim=True)

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        d_org_left = F.conv2d(org_pool, self.k_left, padding=1)
        d_org_right = F.conv2d(org_pool, self.k_right, padding=1)
        d_org_up = F.conv2d(org_pool, self.k_up, padding=1)
        d_org_down = F.conv2d(org_pool, self.k_down, padding=1)

        d_en_left = F.conv2d(enhance_pool, self.k_left, padding=1)
        d_en_right = F.conv2d(enhance_pool, self.k_right, padding=1)
        d_en_up = F.conv2d(enhance_pool, self.k_up, padding=1)
        d_en_down = F.conv2d(enhance_pool, self.k_down, padding=1)

        e = (d_org_left - d_en_left).pow(2) + (d_org_right - d_en_right).pow(2) + \
            (d_org_up - d_en_up).pow(2) + (d_org_down - d_en_down).pow(2)
        return e


class L_exp(nn.Module):
    def __init__(self, patch_size: int = 16, mean_val: float = 0.6) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_gray = torch.mean(x, dim=1, keepdim=True)
        mean = self.pool(x_gray)
        return torch.mean((mean - self.mean_val) ** 2)


class L_TV(nn.Module):
    def __init__(self, TVLoss_weight: float = 1.0) -> None:
        super().__init__()
        self.w = TVLoss_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        h_x = x.size(2)
        w_x = x.size(3)
        count_h = (h_x - 1) * w_x
        count_w = h_x * (w_x - 1)
        h_tv = (x[:, :, 1:, :] - x[:, :, :h_x - 1, :]).pow(2).sum()
        w_tv = (x[:, :, :, 1:] - x[:, :, :, :w_x - 1]).pow(2).sum()
        return self.w * 2.0 * (h_tv / count_h + w_tv / count_w) / b


class L_chroma(nn.Module):
    def forward(self, org: torch.Tensor, enhance: torch.Tensor) -> torch.Tensor:
        org_hsv = rgb_to_hsv(org)
        en_hsv = rgb_to_hsv(enhance)

        # Hue wrap-around distance in [0,1]
        dh = torch.abs(org_hsv[:, 0] - en_hsv[:, 0])
        dh = torch.minimum(dh, 1.0 - dh)
        hue_loss = dh.mean()

        sat_loss = torch.abs(org_hsv[:, 1] - en_hsv[:, 1]).mean()
        return 0.5 * hue_loss + 0.5 * sat_loss


class L_illumination(nn.Module):
    """
    Illumination consistency loss:
    variance of gradients of the grayscale enhanced image (paper Eq. 12 style).
    """
    def __init__(self) -> None:
        super().__init__()
        kernels = torch.tensor([
            [[-1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[-1, 0], [0, 1]],
            [[0, -1], [1, 0]],
            [[0, -1], [0, 1]],
            [[0, 0], [-1, 1]],
        ], dtype=torch.float32).view(6, 1, 2, 2)
        self.register_buffer("grad_kernel", kernels)

    def forward(self, enhance: torch.Tensor) -> torch.Tensor:
        mean_enhance = enhance.mean(dim=1, keepdim=True)
        grad = F.conv2d(mean_enhance, self.grad_kernel, padding=1)
        return torch.var(grad)
