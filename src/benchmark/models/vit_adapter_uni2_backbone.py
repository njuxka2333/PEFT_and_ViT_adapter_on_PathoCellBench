"""UNI2 ViT backbone with ViT-Adapter-style injectors (pathocellbench; PFMB has no equivalent module)."""
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from huggingface_hub import login
from omegaconf import OmegaConf
from timm import create_model
from timm.layers import SwiGLUPacked


dotenv_path = Path(__file__).parents[3] / ".env"
load_dotenv(dotenv_path=dotenv_path)
HF_TOKEN = os.getenv("HF_TOKEN")


class _InjectorBlock(nn.Module):
    """Fuse UNI2 token feature map with local convolutional context."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.token_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        self.local_proj = nn.Conv2d(out_dim, out_dim, kernel_size=1)
        self.mix = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, token_map: torch.Tensor, local_feat: torch.Tensor) -> torch.Tensor:
        x = self.token_proj(token_map)
        if local_feat.shape[-2:] != x.shape[-2:]:
            local_feat = F.interpolate(local_feat, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = x + self.local_proj(local_feat)
        return self.mix(x)


class ViTAdapterUNI2Backbone(nn.Module):
    """UNI2-backed ViT-Adapter-like backbone with injector alignment."""

    def __init__(
        self,
        model_name: str = "uni2",
        out_channels: int = 256,
        interaction_indexes: list[list[int]] | None = None,
        strict_mapping: bool = False,
        load_pretrained: bool = True,
    ):
        super().__init__()
        if model_name.lower() != "uni2":
            raise ValueError(f"ViTAdapterUNI2Backbone expects model_name='uni2', got: {model_name}")
        if HF_TOKEN:
            login(token=HF_TOKEN)

        model_cfg = OmegaConf.load("configs/models/uni2.yaml")
        self.patch_size = int(model_cfg.patch_size)
        self.embed_dim = int(model_cfg.embed_dim)
        self.reg_tokens = int(model_cfg.reg_tokens)
        self.interaction_indexes = interaction_indexes or [[0, 5], [6, 11], [12, 17], [18, 23]]
        self.out_channels = out_channels

        self.vit = create_model(
            f"hf_hub:{model_cfg.url}",
            pretrained=False,
            img_size=model_cfg.img_size,
            patch_size=model_cfg.patch_size,
            depth=model_cfg.depth,
            num_heads=model_cfg.num_heads,
            init_values=model_cfg.init_values,
            embed_dim=model_cfg.embed_dim,
            mlp_ratio=model_cfg.mlp_ratio,
            num_classes=model_cfg.num_classes,
            no_embed_class=model_cfg.no_embed_class,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            reg_tokens=model_cfg.reg_tokens,
            dynamic_img_size=model_cfg.dynamic_img_size,
        )
        if load_pretrained:
            self._load_uni2_pretrained_with_mapping(strict_mapping=strict_mapping)
        else:
            print("UNI2 mapping skipped (load_pretrained=False).")

        # Local context pyramid (1/4, 1/8, 1/16, 1/32)
        self.local_stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.local_down1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.local_down2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.local_down3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        self.injectors = nn.ModuleList([_InjectorBlock(self.embed_dim, out_channels) for _ in range(4)])

    def freeze_pretrained_vit(self):
        """Freeze timm UNI2 ViT weights only; keep local pyramid + injectors trainable."""
        for p in self.vit.parameters():
            p.requires_grad = False
        for m in (self.local_stem, self.local_down1, self.local_down2, self.local_down3, self.injectors):
            for p in m.parameters():
                p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def _build_mapping(self, source_state: dict[str, torch.Tensor], target_state: dict[str, torch.Tensor]) -> dict[str, str]:
        mapping = {}
        for t_key in target_state.keys():
            candidates = [
                t_key,
                f"model.{t_key}",
                t_key.replace("vit.", ""),
            ]
            for c in candidates:
                if c in source_state and source_state[c].shape == target_state[t_key].shape:
                    mapping[c] = t_key
                    break
        return mapping

    def _load_uni2_pretrained_with_mapping(self, strict_mapping: bool = False):
        model_cfg = OmegaConf.load("configs/models/uni2.yaml")
        src_model = create_model(
            f"hf_hub:{model_cfg.url}",
            pretrained=True,
            img_size=model_cfg.img_size,
            patch_size=model_cfg.patch_size,
            depth=model_cfg.depth,
            num_heads=model_cfg.num_heads,
            init_values=model_cfg.init_values,
            embed_dim=model_cfg.embed_dim,
            mlp_ratio=model_cfg.mlp_ratio,
            num_classes=model_cfg.num_classes,
            no_embed_class=model_cfg.no_embed_class,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            reg_tokens=model_cfg.reg_tokens,
            dynamic_img_size=model_cfg.dynamic_img_size,
        )
        src_state = src_model.state_dict()
        tgt_state = self.vit.state_dict()

        mapping = self._build_mapping(src_state, tgt_state)
        remapped = {t_key: src_state[s_key] for s_key, t_key in mapping.items()}
        incompatible = self.vit.load_state_dict(remapped, strict=False)

        print(
            "UNI2 mapping report | mapped:",
            len(remapped),
            "missing:",
            len(incompatible.missing_keys),
            "unexpected:",
            len(incompatible.unexpected_keys),
        )
        if strict_mapping and (incompatible.missing_keys or incompatible.unexpected_keys):
            raise RuntimeError(
                "UNI2 mapping strict mode failed. "
                f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
            )
        del src_model

    def _tokens_to_map(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # Remove CLS + register tokens, then reshape to patch grid.
        prefix = 1 + self.reg_tokens
        tokens = tokens[:, prefix:, :]
        gh = h // self.patch_size
        gw = w // self.patch_size
        tokens = tokens[:, : gh * gw, :]
        return tokens.reshape(tokens.shape[0], gh, gw, tokens.shape[-1]).permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        b, _, h, w = x.shape
        x_local1 = self.local_stem(x)
        x_local2 = self.local_down1(x_local1)
        x_local3 = self.local_down2(x_local2)
        x_local4 = self.local_down3(x_local3)
        locals_ = [x_local1, x_local2, x_local3, x_local4]

        # Run UNI2 transformer and collect stage outputs for injector alignment.
        t = self.vit.patch_embed(x)
        t = self.vit._pos_embed(t)
        t = self.vit.patch_drop(t)
        t = self.vit.norm_pre(t)

        captures = []
        end_indices = [int(r[1]) for r in self.interaction_indexes]
        for i, blk in enumerate(self.vit.blocks):
            t = blk(t)
            if i in end_indices:
                captures.append(t)
        if len(captures) != 4:
            # Fallback to last 4 block outputs if custom indexes mismatch.
            captures = []
            t2 = self.vit.patch_embed(x)
            t2 = self.vit._pos_embed(t2)
            t2 = self.vit.patch_drop(t2)
            t2 = self.vit.norm_pre(t2)
            for i, blk in enumerate(self.vit.blocks):
                t2 = blk(t2)
                if i >= len(self.vit.blocks) - 4:
                    captures.append(t2)

        outs = []
        for stage_idx, token_stage in enumerate(captures):
            token_map = self._tokens_to_map(token_stage, h, w)
            aligned = self.injectors[stage_idx](token_map, locals_[stage_idx])
            outs.append(aligned)
        return outs

