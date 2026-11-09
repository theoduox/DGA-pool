"""Zero-reference training adapted from the unified source."""
from __future__ import annotations
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from .model import EnhanceNetLight
from .losses import L_color, L_spa, L_exp, L_TV, L_chroma, L_illumination
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def weights_init(m: nn.Module) -> None:
    """Mimics the original init in lowlight_train[4].py, but safely handles conv/bn layers."""
    classname = m.__class__.__name__
    if "Conv" in classname:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif "BatchNorm" in classname:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.zeros_(m.bias)


class LowlightFolderDataset(Dataset):
    """
    A simple folder dataset. It replaces the external 'dataloader.lowlight_loader(...)' dependency.
    Assumption: lowlight_images_path contains images (optionally nested in subfolders).
    """

    def __init__(self, root: str | Path, image_size: int = 224) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"lowlight_images_path not found: {self.root}")

        self.paths: List[Path] = [
            p for p in sorted(self.root.rglob("*")) if p.is_file() and p.suffix.lower() in _IMG_EXTS
        ]
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found under: {self.root}")

        self.tf = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),  # -> [0,1], shape [C,H,W]
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        p = self.paths[idx]
        with Image.open(p) as im:
            im = im.convert("RGB")
            x = self.tf(im)
        return x


@dataclass
class TrainConfig:
    lowlight_images_path: str = "data/train_data/"
    snapshots_folder: str = "snapshots/"
    image_size: int = 224

    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 0.1
    num_epochs: int = 200
    train_batch_size: int = 8
    num_workers: int = 4
    display_iter: int = 10
    snapshot_iter: int = 0  # default: save per-epoch by setting large value

    load_pretrain: bool = False
    pretrain_path: str = ""

    curve_steps: int = 8
    seed: int = 42
    cpu: bool = False


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    device = torch.device("cpu" if cfg.cpu or not torch.cuda.is_available() else "cuda")

    os.makedirs(cfg.snapshots_folder, exist_ok=True)

    # Data
    dataset = LowlightFolderDataset(cfg.lowlight_images_path, image_size=cfg.image_size)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # Model
    net = EnhanceNetLight(number_f=16, curve_steps=cfg.curve_steps).to(device)
    net.apply(weights_init)

    if cfg.load_pretrain:
        if not cfg.pretrain_path:
            raise ValueError("load_pretrain=True but pretrain_path is empty.")
        state = torch.load(cfg.pretrain_path, map_location=device, weights_only=True)
        net.load_state_dict(state, strict=True)

    # Losses
    l_color = L_color().to(device)
    l_spa = L_spa().to(device)
    l_exp = L_exp(16, 0.6).to(device)
    l_tv = L_TV().to(device)
    l_chroma = L_chroma().to(device)
    l_illum = L_illumination().to(device)

    # Optimizer & scheduler
    opt = torch.optim.Adam(
        net.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.999),
        weight_decay=cfg.weight_decay,
    )
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=cfg.num_epochs,
        eta_min=1e-6,
    )

    net.train()

    global_step = 0
    for epoch in range(cfg.num_epochs):
        print(f"Epoch {epoch + 1}/{cfg.num_epochs} | LR={sch.get_last_lr()[0]:.6g}")

        for it, img_low in enumerate(loader, start=1):
            global_step += 1
            img_low = img_low.to(device, non_blocking=True)

            _, img_enh, A = net(img_low)

            # Weights (match the manuscript Table 1)
            loss_tv = 200.0 * l_tv(A)
            loss_spa = 1.0 * torch.mean(l_spa(img_enh, img_low))
            loss_col = 5.0 * torch.mean(l_color(img_enh))
            loss_exp = 10.0 * l_exp(img_enh)  # already scalar
            loss_chroma = 5.0 * l_chroma(img_low, img_enh)
            loss_illum = 2.0 * l_illum(img_enh)

            loss = loss_tv + loss_spa + loss_col + loss_exp + loss_chroma + loss_illum

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip_norm)
            opt.step()

            if (it % cfg.display_iter) == 0:
                print(
                    f"  iter {it:5d} | loss={loss.item():.6f} "
                    f"(tv={loss_tv.item():.3f}, spa={loss_spa.item():.3f}, col={loss_col.item():.3f}, "
                    f"exp={loss_exp.item():.3f}, chroma={loss_chroma.item():.3f}, illum={loss_illum.item():.3f})"
                )

            if (cfg.snapshot_iter > 0) and (global_step % cfg.snapshot_iter == 0):
                save_path = Path(cfg.snapshots_folder) / f"step{global_step:07d}_epoch{epoch+1:03d}.pth"
                torch.save(net.state_dict(), save_path.as_posix())
                print(f"  saved: {save_path}")

        # save per epoch
        save_path = Path(cfg.snapshots_folder) / f"epoch{epoch+1:03d}.pth"
        torch.save(net.state_dict(), save_path.as_posix())
        sch.step()
