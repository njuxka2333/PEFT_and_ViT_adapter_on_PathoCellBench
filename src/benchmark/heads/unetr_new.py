"""Alternate UNETR wrapper (same decoder as ``benchmark.heads.unetr.UnetR``); optional ``backbone_out_indices``."""

from collections import OrderedDict

import einops
import torch
import torch.nn as nn

from benchmark.models.simple_segmentation_model import clean_str, load_model_and_transform


class Conv2DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class Deconv2DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


def forward_musk_embed(
    self,
    visual_tokens=None,
    attn_mask=None,
    vision_masked_position=None,
    incremental_state=None,
    positions=None,
):
    x = self.vision_embed(visual_tokens, vision_masked_position)
    encoder_out = self.encoder(
        src_tokens=None,
        encoder_padding_mask=None,
        attn_mask=attn_mask,
        token_embeddings=x,
        multiway_split_position=-1,
        incremental_state=incremental_state,
        positions=positions,
        return_all_hiddens=True,
    )
    return encoder_out


class UnetRNew(nn.Module):
    """Same architecture as ``UnetR`` in ``unetr.py``; edit here without touching the original."""

    def __init__(
        self,
        model_name,
        num_classes,
        drop_rate: float = 0.0,
        do_ms_aug: bool = False,
        lora_r: int | None = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        use_dora: bool = False,
        kradapter_enable: bool = False,
        kradapter_rank_order: int = 2,
        kradapter_alpha: float = 1.0,
        kradapter_dropout: float = 0.0,
        kradapter_attn_only: bool = False,
        kradapter_mlp_only: bool = False,
        kradapter_bottleneck_dim: int | None = None,
        backbone_out_indices: tuple[int, ...] | list[int] | None = None,
        randlora_enable: bool = False,
        randlora_rank: int = 4,
        randlora_num_basis: int = 16,
        randlora_seed: int = 42,
        randlora_param_type: str = "randlora",
        randlora_sparse: bool = False,
        randlora_dist_type: str = "uniform",
        randlora_sparcity: int = 6,
        randlora_attn_only: bool = False,
        randlora_mlp_only: bool = False,
    ):
        super().__init__()
        self.model, self.transform, model_dim, self.patch_size, self.image_size = load_model_and_transform(
            model_name,
            features_only=True,
            do_ms_aug=do_ms_aug,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_dora=use_dora,
            kradapter_enable=kradapter_enable,
            kradapter_rank_order=kradapter_rank_order,
            kradapter_alpha=kradapter_alpha,
            kradapter_dropout=kradapter_dropout,
            kradapter_attn_only=kradapter_attn_only,
            kradapter_mlp_only=kradapter_mlp_only,
            kradapter_bottleneck_dim=kradapter_bottleneck_dim,
            backbone_out_indices=backbone_out_indices,
            randlora_enable=randlora_enable,
            randlora_rank=randlora_rank,
            randlora_num_basis=randlora_num_basis,
            randlora_seed=randlora_seed,
            randlora_param_type=randlora_param_type,
            randlora_sparse=randlora_sparse,
            randlora_dist_type=randlora_dist_type,
            randlora_sparcity=randlora_sparcity,
            randlora_attn_only=randlora_attn_only,
            randlora_mlp_only=randlora_mlp_only,
        )
        self.model_name = clean_str(model_name)
        self.embed_dim = model_dim[-1] if not isinstance(model_dim, int) else model_dim
        self.drop_rate = drop_rate

        if self.embed_dim < 512:
            self.skip_dim_11 = 256
            self.skip_dim_12 = 128
            self.bottleneck_dim = 312
        else:
            self.skip_dim_11 = 512
            self.skip_dim_12 = 256
            self.bottleneck_dim = 512

        self.decoder0 = nn.Sequential(
            Conv2DBlock(3, 32, 3, dropout=self.drop_rate),
            Conv2DBlock(32, 64, 3, dropout=self.drop_rate),
        )
        self.decoder1 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, self.skip_dim_12, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_12, 128, dropout=self.drop_rate),
        )
        self.decoder2 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, 256, dropout=self.drop_rate),
        )
        self.decoder3 = nn.Sequential(Deconv2DBlock(self.embed_dim, self.bottleneck_dim, dropout=self.drop_rate))
        self.nuclei_type_maps_decoder = self.create_upsampling_branch(num_classes)

    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def freeze_pretrained_backbone(self):
        """Keep UNI2 (timm) backbone frozen; only adapter factors (kr_a/kr_b or RandLoRA ``loramat``) train."""
        for name, param in self.model.named_parameters():
            param.requires_grad = ("kr_a" in name) or ("kr_b" in name) or ("loramat" in name)

    def unfreeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        if self.model_name == "phikonv2":
            x = self.transform(x, return_tensors="pt")["pixel_values"]
            x = x.to(next(self.model.parameters()).device)
        else:
            x = self.transform(x)

        if self.model_name == "musk":
            outputs = forward_musk_embed(self.model.beit3, visual_tokens=x)
            patch_embeddings = outputs["encoder_states"][-4:]
        else:
            patch_embeddings = self.model(x)

        if self.model_name == "phikonv2":
            patch_embeddings = patch_embeddings["hidden_states"][-4:]

        if self.model_name in ["phikonv2", "titan", "musk"]:
            reshaped_embed = []
            for layer in patch_embeddings:
                reshaped_embed.append(
                    einops.rearrange(
                        layer[:, 1:, :],
                        "b (p1 p2) d -> b d p1 p2",
                        p1=self.image_size // self.patch_size,
                        p2=self.image_size // self.patch_size,
                    )
                )
            patch_embeddings = reshaped_embed

        z0, z1, z2, z3, z4 = x, *patch_embeddings
        return self._forward_upsample(z0, z1, z2, z3, z4, self.nuclei_type_maps_decoder)

    def _forward_upsample(self, z0, z1, z2, z3, z4, branch_decoder):
        b4 = branch_decoder.bottleneck_upsampler(z4)
        b3 = self.decoder3(z3)
        b3 = branch_decoder.decoder3_upsampler(torch.cat([b3, b4], dim=1))
        b2 = self.decoder2(z2)
        b2 = branch_decoder.decoder2_upsampler(torch.cat([b2, b3], dim=1))
        b1 = self.decoder1(z1)
        b1 = branch_decoder.decoder1_upsampler(torch.cat([b1, b2], dim=1))
        b0 = self.decoder0(z0)
        if b0.shape != b1.shape:
            _, _, h, w = b0.shape
            b1 = nn.functional.interpolate(b1, size=(h, w), mode="bilinear", align_corners=False)
        return branch_decoder.decoder0_header(torch.cat([b0, b1], dim=1))

    def create_upsampling_branch(self, num_classes: int):
        bottleneck_upsampler = nn.ConvTranspose2d(
            in_channels=self.embed_dim,
            out_channels=self.bottleneck_dim,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        )
        decoder3_upsampler = nn.Sequential(
            Conv2DBlock(self.bottleneck_dim * 2, self.bottleneck_dim, dropout=self.drop_rate),
            Conv2DBlock(self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate),
            Conv2DBlock(self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate),
            nn.ConvTranspose2d(self.bottleneck_dim, 256, kernel_size=2, stride=2),
        )
        decoder2_upsampler = nn.Sequential(
            Conv2DBlock(256 * 2, 256, dropout=self.drop_rate),
            Conv2DBlock(256, 256, dropout=self.drop_rate),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
        )
        decoder1_upsampler = nn.Sequential(
            Conv2DBlock(128 * 2, 128, dropout=self.drop_rate),
            Conv2DBlock(128, 128, dropout=self.drop_rate),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        )
        decoder0_header = nn.Sequential(
            Conv2DBlock(64 * 2, 64, dropout=self.drop_rate),
            Conv2DBlock(64, 64, dropout=self.drop_rate),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )
        return nn.Sequential(
            OrderedDict(
                [
                    ("bottleneck_upsampler", bottleneck_upsampler),
                    ("decoder3_upsampler", decoder3_upsampler),
                    ("decoder2_upsampler", decoder2_upsampler),
                    ("decoder1_upsampler", decoder1_upsampler),
                    ("decoder0_header", decoder0_header),
                ]
            )
        )
