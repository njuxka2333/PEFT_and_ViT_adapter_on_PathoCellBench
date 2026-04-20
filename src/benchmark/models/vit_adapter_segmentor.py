"""ViT-Adapter + Mask2Former segmentor for UNI2 (pathocellbench extension vs Pathology-Foundation-Model-Benchmark)."""
import torch
import torch.nn as nn

from benchmark.heads.mask2former_head import Mask2FormerHead
from benchmark.models.vit_adapter_uni2_backbone import ViTAdapterUNI2Backbone


class ViTAdapterUNI2(nn.Module):
    """Pathocellbench-native ViT-Adapter segmentor with UNI2 backbone."""

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        do_ms_aug: bool = False,
        interaction_indexes: list[list[int]] | None = None,
        adapter_out_channels: int = 256,
        decoder_hidden_dim: int = 256,
        strict_mapping: bool = False,
        load_pretrained: bool = True,
        **_,
    ):
        super().__init__()
        self.do_ms_aug = do_ms_aug

        self.model = ViTAdapterUNI2Backbone(
            model_name=model_name,
            out_channels=adapter_out_channels,
            interaction_indexes=interaction_indexes,
            strict_mapping=strict_mapping,
            load_pretrained=load_pretrained,
        )
        self.decoder = Mask2FormerHead(
            in_channels=adapter_out_channels,
            hidden_dim=decoder_hidden_dim,
            num_classes=num_classes,
        )

    def freeze_model(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def unfreeze_model(self):
        self.model.unfreeze_all()

    def freeze_pretrained_backbone(self):
        """Match Run B: frozen UNI2 trunk, train adapter injectors + local stem + decoder."""
        self.model.freeze_pretrained_vit()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.model(x)
        return self.decoder(feats, out_size=(x.shape[-2], x.shape[-1]))

