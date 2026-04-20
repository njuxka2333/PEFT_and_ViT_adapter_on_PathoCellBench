# pathocellbench

`src/benchmark` is kept **aligned with** the upstream [Pathology-Foundation-Model-Benchmark](https://github.com/Kainmueller-Lab/Pathology-Foundation-Model-Benchmark) tree (`../Pathology-Foundation-Model-Benchmark/src/benchmark`). Refresh from upstream with:

```bash
rsync -a --exclude '__pycache__' ../Pathology-Foundation-Model-Benchmark/src/benchmark/ ./src/benchmark/
```

Then re-apply the **minimal local patches** below (only these files should diverge from upstream).

## Local patches (downstream-only)

| Area | Files |
|------|--------|
| Optional model wrappers in `train.py` | Add imports: `UnetRNew`, `ViTAdapterUNI2` so `eval(cfg.model.model_wrapper)` can resolve them. |
| `augmentations/augmentations.py` | Coerce YAML list `mean` / `std` for Kornia `Normalize` to `torch.Tensor` (same pattern as `RandomElasticTransform`). |
| `models/simple_segmentation_model.py` | Optional `backbone_out_indices`; PEFT LoRA; **KRAdapter** (`KRLinear`); **RandLoRA** (optional `rand_lora` on `PYTHONPATH`). |
| `heads/unetr.py` | Extra kwargs passed into `load_model_and_transform`; `freeze_pretrained_backbone()` for adapter-only runs. |
| `run/train.py` | Text log `{experiment}.txt` under the run `log_dir` (train steps + validation + test), rank-0 only under DDP. |
| Extra modules (not in upstream) | `models/kr_adapter.py`, `models/vit_adapter_uni2_backbone.py`, `models/vit_adapter_segmentor.py`, `heads/mask2former_head.py`, `heads/unetr_new.py`. |

Installation, data, and benchmark tables follow the upstream README.
