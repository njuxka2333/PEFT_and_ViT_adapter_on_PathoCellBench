import argparse
import csv
import os
from datetime import datetime
from time import time

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf

from benchmark.heads.dpt import DPT
from benchmark.heads.unetr import UnetR
from benchmark.heads.unetr_new import UnetRNew
from benchmark.models.hovernext import HoverNext
from benchmark.models.simple_segmentation_model import MockModel, SimpleSegmentationModel
from benchmark.models.vit_adapter_segmentor import ViTAdapterUNI2
from benchmark.run.eval import Eval
from benchmark.run.train import create_dataloaders, initialize_dist, make_log_dirs
from benchmark.utils.utils import prep_datasets

os.environ["OMP_NUM_THREADS"] = "1"
torch.backends.cudnn.benchmark = True
MAX_EARLY_STOPPING = 5000
MAX_UMAP_POINTS = 50000
RARE_CLASS_ALIASES = {
    "adipocyte": {"adipocyte", "adipocytes"},
    "dendritic_cell": {"dendritic cell", "dendritic cells"},
    "nerves": {"nerve", "nerves", "nvevers"},
    "nk_cell": {"nk cell", "nk cells"},
}

try:
    import matplotlib.pyplot as plt
    import umap

    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

try:
    from sklearn.metrics import davies_bouldin_score, silhouette_score

    HAS_SKLEARN_METRICS = True
except Exception:
    HAS_SKLEARN_METRICS = False


def print_ds_stats(train_dset, val_dset, test_dset, label_dict):
    print("-------- printing dataset stats -------")
    print(f"Number of classes: {len(label_dict.keys())}")
    print(f"Unique samples: {len(label_dict.keys())}")

    def print_ds(ds):
        print(f"Number of samples: {len(ds)}")

    print("Train")
    print_ds(train_dset)
    print("Val")
    print_ds(val_dset)
    print("Test")
    print_ds(test_dset)


def _resolve_model_wrapper(name):
    wrappers = {
        "SimpleSegmentationModel": SimpleSegmentationModel,
        "MockModel": MockModel,
        "DPT": DPT,
        "HoverNext": HoverNext,
        "UnetR": UnetR,
        "UnetRNew": UnetRNew,
        "ViTAdapterUNI2": ViTAdapterUNI2,
    }
    if name not in wrappers:
        raise ValueError(f"Unknown model wrapper: {name}")
    return wrappers[name]


def _sample_indices(indices, n):
    if n <= 0 or indices.numel() == 0:
        return indices.new_empty((0,), dtype=torch.long)
    if indices.numel() <= n:
        return indices
    return indices[torch.randperm(indices.numel(), device=indices.device)[:n]]


def _make_local_region_mask(rare_mask_2d, radius):
    if radius <= 0:
        return rare_mask_2d
    x = rare_mask_2d.float().unsqueeze(0).unsqueeze(0)
    k = 2 * int(radius) + 1
    local = F.max_pool2d(x, kernel_size=k, stride=1, padding=radius)
    return local.squeeze(0).squeeze(0) > 0


def _pick_feature_tensor(output):
    if isinstance(output, (list, tuple)) and len(output) > 0:
        return output[-1]
    if isinstance(output, dict):
        if "x" in output and torch.is_tensor(output["x"]):
            return output["x"]
        if "out" in output and torch.is_tensor(output["out"]):
            return output["out"]
    return output if torch.is_tensor(output) else None


def _normalize_feature_map(feat):
    if feat is None:
        return None
    if feat.dim() == 3:
        b, n, d = feat.shape
        side = int(round((n) ** 0.5))
        if side * side == n:
            feat = feat.transpose(1, 2).reshape(b, d, side, side)
        else:
            return None
    if feat.dim() != 4:
        return None
    return feat


def _find_ids(label_dict):
    id2name = {int(k): str(v).strip().lower() for k, v in label_dict.items()}
    rare_ids = []
    for cid, cname in id2name.items():
        for aliases in RARE_CLASS_ALIASES.values():
            if cname in aliases:
                rare_ids.append(cid)
                break
    tumor_id = None
    for cid, cname in id2name.items():
        if "tumor" in cname:
            tumor_id = cid
            break
    return sorted(set(rare_ids)), tumor_id


def _compute_high_dim_cluster_metrics(x, y):
    out = {
        "silhouette_score": float("nan"),
        "davies_bouldin_index": float("nan"),
        "separability_metric": float("nan"),
    }
    labels = y.cpu().numpy()
    if x.shape[0] < 10 or len(np.unique(labels)) < 2:
        return out
    x_np = x.cpu().numpy()
    if HAS_SKLEARN_METRICS:
        try:
            out["silhouette_score"] = float(silhouette_score(x_np, labels))
        except Exception:
            pass
        try:
            out["davies_bouldin_index"] = float(davies_bouldin_score(x_np, labels))
        except Exception:
            pass
    centers = []
    intra = []
    for c in np.unique(labels):
        xc = x_np[labels == c]
        if xc.shape[0] < 2:
            continue
        mu = xc.mean(axis=0)
        centers.append(mu)
        intra.append(float(np.linalg.norm(xc - mu, axis=1).mean()))
    if len(centers) >= 2 and len(intra) > 0:
        centers = np.stack(centers, axis=0)
        dmat = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
        inter = dmat[np.triu_indices_from(dmat, k=1)].mean()
        out["separability_metric"] = float(inter / (np.mean(intra) + 1e-12))
    return out


def run_umap_targeted_cooccurrence(
    model,
    test_dataloader,
    device,
    label_dict,
    output_dir,
    radius=4,
    include_background=False,
):
    os.makedirs(output_dir, exist_ok=True)
    metrics_csv = os.path.join(output_dir, "umap_targeted_cooccurrence_metrics.csv")
    rare_ids, tumor_id = _find_ids(label_dict)
    if len(rare_ids) == 0:
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["status"])
            w.writeheader()
            w.writerow({"status": "skipped_no_rare_ids"})
        return

    model_ref = getattr(model, "module", model)
    hook_feats = []

    def _hook(_, __, out):
        hook_feats.append(out)

    if not hasattr(model_ref, "model"):
        return
    handle = model_ref.model.register_forward_hook(_hook)
    xs, ys = [], []
    try:
        model.eval()
        with torch.no_grad():
            for batch in test_dataloader:
                img = batch["image"].float().to(device)
                sem = batch["semantic_mask"].to(device)
                _ = model(img)
                if len(hook_feats) == 0:
                    continue
                feat = _normalize_feature_map(_pick_feature_tensor(hook_feats[-1]))
                hook_feats.clear()
                if feat is None:
                    continue
                b, c, h4, w4 = feat.shape
                sem4 = F.interpolate(sem.float().unsqueeze(1), size=(h4, w4), mode="nearest").squeeze(1).long()
                z = F.normalize(feat, dim=1).permute(0, 2, 3, 1).reshape(b, h4 * w4, c)
                semf = sem4.reshape(b, h4 * w4)
                for bi in range(b):
                    yb = semf[bi]
                    zb = z[bi]
                    rare_mask = torch.zeros_like(yb, dtype=torch.bool)
                    for rid in rare_ids:
                        rare_mask |= (yb == rid)
                    rare_idx = torch.where(rare_mask)[0]
                    if rare_idx.numel() == 0:
                        continue
                    rare_2d = rare_mask.reshape(h4, w4)
                    local = _make_local_region_mask(rare_2d, radius).reshape(-1)
                    ctx_mask = local & (~rare_mask)
                    if tumor_id is not None:
                        ctx_mask = ctx_mask & ((yb == tumor_id) | (yb == 0))
                    if not include_background:
                        ctx_mask = ctx_mask & (yb != 0)
                    ctx_idx = torch.where(ctx_mask)[0]
                    n = min(int(rare_idx.numel()), int(ctx_idx.numel()))
                    if n <= 0:
                        continue
                    rare_take = _sample_indices(rare_idx, n)
                    ctx_take = _sample_indices(ctx_idx, n)
                    take = torch.cat([rare_take, ctx_take], dim=0)
                    y_take = torch.cat(
                        [
                            torch.ones(n, device=yb.device, dtype=torch.long),
                            torch.zeros(n, device=yb.device, dtype=torch.long),
                        ],
                        dim=0,
                    )
                    xs.append(zb[take].cpu())
                    ys.append(y_take.cpu())
                    if sum(xi.shape[0] for xi in xs) >= MAX_UMAP_POINTS:
                        break
                if sum(xi.shape[0] for xi in xs) >= MAX_UMAP_POINTS:
                    break
    finally:
        handle.remove()

    if len(xs) == 0:
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["status"])
            w.writeheader()
            w.writerow({"status": "skipped_no_samples"})
        return

    x = torch.cat(xs, dim=0)[:MAX_UMAP_POINTS]
    y = torch.cat(ys, dim=0)[:MAX_UMAP_POINTS]
    metrics = _compute_high_dim_cluster_metrics(x, y)
    metrics.update({"n_points": int(x.shape[0]), "n_classes": int(y.unique().numel())})

    if HAS_UMAP:
        emb2 = umap.UMAP(n_components=2, random_state=42).fit_transform(x.numpy())
        fig = plt.figure(figsize=(7, 6))
        cls0 = y.numpy() == 0
        plt.scatter(emb2[cls0, 0], emb2[cls0, 1], s=2, alpha=0.35, label="context")
        plt.scatter(emb2[~cls0, 0], emb2[~cls0, 1], s=2, alpha=0.45, label="rare")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "umap_targeted_cooccurrence.png"), dpi=220)
        plt.close(fig)

    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        w.writeheader()
        w.writerow(metrics)


def train(cfg):
    log_dir, _, snap_dir = make_log_dirs(cfg, split="test")
    best_model_path = cfg.best_model_path
    train_dset, val_dset, test_dset, label_dict = prep_datasets(cfg)
    print_ds_stats(train_dset, val_dset, test_dset, label_dict)

    model_wrapper = _resolve_model_wrapper(cfg.model.model_wrapper)
    model_kwargs = dict(
        model_name=cfg.model.backbone,
        num_classes=len(label_dict),
        do_ms_aug=getattr(cfg.model, "do_ms_aug", False),
    )
    if cfg.model.model_wrapper in ("UnetR", "UnetRNew", "SimpleSegmentationModel"):
        model_kwargs["lora_r"] = getattr(cfg.model, "lora_r", None)
        model_kwargs["lora_alpha"] = int(getattr(cfg.model, "lora_alpha", 32))
        model_kwargs["lora_dropout"] = float(getattr(cfg.model, "lora_dropout", 0.1))
        model_kwargs["use_dora"] = bool(getattr(cfg.model, "use_dora", False))
        if cfg.model.model_wrapper in ("UnetR", "UnetRNew"):
            model_kwargs["kradapter_enable"] = bool(getattr(cfg.model, "kradapter_enable", False))
            model_kwargs["kradapter_rank_order"] = int(getattr(cfg.model, "kradapter_rank_order", 2))
            model_kwargs["kradapter_alpha"] = float(getattr(cfg.model, "kradapter_alpha", 1.0))
            model_kwargs["kradapter_dropout"] = float(getattr(cfg.model, "kradapter_dropout", 0.0))
            model_kwargs["kradapter_attn_only"] = bool(getattr(cfg.model, "kradapter_attn_only", False))
            model_kwargs["kradapter_mlp_only"] = bool(getattr(cfg.model, "kradapter_mlp_only", False))
            model_kwargs["kradapter_bottleneck_dim"] = getattr(cfg.model, "kradapter_bottleneck_dim", None)
            model_kwargs["randlora_enable"] = bool(getattr(cfg.model, "randlora_enable", False))
            model_kwargs["randlora_rank"] = int(getattr(cfg.model, "randlora_rank", 4))
            model_kwargs["randlora_num_basis"] = int(getattr(cfg.model, "randlora_num_basis", 16))
            model_kwargs["randlora_seed"] = int(getattr(cfg.model, "randlora_seed", 42))
            model_kwargs["randlora_param_type"] = str(getattr(cfg.model, "randlora_param_type", "randlora"))
            model_kwargs["randlora_sparse"] = bool(getattr(cfg.model, "randlora_sparse", False))
            model_kwargs["randlora_dist_type"] = str(getattr(cfg.model, "randlora_dist_type", "uniform"))
            model_kwargs["randlora_sparcity"] = int(getattr(cfg.model, "randlora_sparcity", 6))
            model_kwargs["randlora_attn_only"] = bool(getattr(cfg.model, "randlora_attn_only", False))
            model_kwargs["randlora_mlp_only"] = bool(getattr(cfg.model, "randlora_mlp_only", False))
    model = model_wrapper(**model_kwargs)

    evaluater = Eval(
        label_dict,
        instance_level=True,
        pixel_level=False,
        save_dir=os.path.join(log_dir, "validation_results"),
        fname="validation_metrics.csv",
    )
    model, device, WORLD_SIZE, LOCAL_RANK = initialize_dist(model)

    if "RANK" in os.environ and LOCAL_RANK == 0 or "mock" not in cfg.experiment.lower():
        logging = True
        wandb.init(
            name=cfg.experiment,
            project=cfg.project,
            entity="kainmueller-lab",
            config=OmegaConf.to_container(cfg),
            dir=cfg.writer_dir,
            mode=cfg.get("wandb_mode", "online"),
        )
    else:
        logging = False

    _, _, test_dataloader = create_dataloaders(cfg, train_dset, val_dset, test_dset, train_sampler=None)
    np.random.seed(cfg.seed if hasattr(cfg, "seed") else time())

    if hasattr(cfg, "primary_metric"):
        model.load_state_dict(torch.load(best_model_path))
        evaluater.save_dir = os.path.join(log_dir, "test_results")
        evaluater.fname = "test_metrics_best_model.csv"
        logging_dict, classwise_dict = evaluater.compute_metrics(model, test_dataloader, device, save_preds=True, snap_dir=snap_dir)
        run_umap_targeted_cooccurrence(
            model=model,
            test_dataloader=test_dataloader,
            device=device,
            label_dict={int(k): v for k, v in label_dict.items()},
            output_dir=evaluater.save_dir,
            radius=int(getattr(getattr(cfg, "umap", {}), "neighbor_radius", 4)) if hasattr(cfg, "umap") else 4,
            include_background=bool(getattr(getattr(cfg, "umap", {}), "include_background", False))
            if hasattr(cfg, "umap")
            else False,
        )

    if logging:
        logging_dict = {f"test/{k}": v for k, v in logging_dict.items()}
        classwise_dict = {k + "_test": v for k, v in classwise_dict.items()}
        wandb.log(logging_dict)
        wandb.log(classwise_dict)

    wandb.finish()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--job_id", type=int, default=0)
    parser.add_argument("--model_name", type=str, default="uni")

    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.model.backbone = args.model_name
    cfg.experiment = f"Test_{cfg.experiment}_{cfg.model.backbone}_{datetime.now().strftime('%d%m_%H%M')}_{cfg.dataset.name}_{args.job_id}"
    print(f"Experiment name: {cfg.experiment}")
    if not hasattr(cfg, "early_stopping"):
        cfg.early_stopping = MAX_EARLY_STOPPING
    if not hasattr(cfg, "save_snapshots"):
        cfg.save_snapshots = False
    if not hasattr(cfg, "save_all_ckpts"):
        cfg.save_all_ckpts = False
    if not hasattr(cfg.model, "do_ms_aug"):
        cfg.model.do_ms_aug = False
    train(cfg)
import argparse
import os
from datetime import datetime
from time import time

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf

from benchmark.heads.dpt import DPT
from benchmark.heads.unetr import UnetR
from benchmark.models.hovernext import HoverNext
from benchmark.models.simple_segmentation_model import (
    MockModel,
    SimpleSegmentationModel,
)
from benchmark.run.eval import Eval
from benchmark.run.train import create_dataloaders, initialize_dist, make_log_dirs
from benchmark.utils.utils import prep_datasets

os.environ["OMP_NUM_THREADS"] = "1"
torch.backends.cudnn.benchmark = True
MAX_EARLY_STOPPING = 5000


def print_ds_stats(train_dset, val_dset, test_dset, label_dict):
    print("-------- printing dataset stats -------")
    print(f"Number of classes: {len(label_dict.keys())}")
    print(f"Unique samples: {len(label_dict.keys())}")

    def print_ds(ds):
        print(f"Number of samples: {len(ds)}")

    print("Train")
    print_ds(train_dset)
    print("Val")
    print_ds(val_dset)
    print("Test")
    print_ds(test_dset)


def train(cfg):
    log_dir, _, snap_dir = make_log_dirs(cfg, split="test")
    best_model_path = cfg.best_model_path
    train_dset, val_dset, test_dset, label_dict = prep_datasets(cfg)
    print_ds_stats(train_dset, val_dset, test_dset, label_dict)
    model_wrapper = eval(cfg.model.model_wrapper)
    model = model_wrapper(
        model_name=cfg.model.backbone,
        num_classes=len(label_dict),
        do_ms_aug=getattr(cfg.model, "do_ms_aug", False),  # Default to False if not specified
    )

    evaluater = Eval(
        label_dict,
        instance_level=True,
        pixel_level=False,
        save_dir=os.path.join(log_dir, "validation_results"),
        fname="validation_metrics.csv",
    )
    model, device, WORLD_SIZE, LOCAL_RANK = initialize_dist(model)

    # log only on rank 0
    if "RANK" in os.environ and LOCAL_RANK == 0 or "mock" not in cfg.experiment.lower():
        logging = True
        wandb.init(
            name=cfg.experiment,
            project=cfg.project,
            entity="kainmueller-lab",
            config=OmegaConf.to_container(cfg),
            dir=cfg.writer_dir,
            mode=cfg.get("wandb_mode", "online"),
        )
    else:
        logging = False

    _, _, test_dataloader = create_dataloaders(cfg, train_dset, val_dset, test_dset, train_sampler=None)
    np.random.seed(cfg.seed if hasattr(cfg, "seed") else time())

    # Testing
    if hasattr(cfg, "primary_metric"):
        model.load_state_dict(torch.load(best_model_path))
        evaluater.save_dir = os.path.join(log_dir, "test_results")
        evaluater.fname = "test_metrics_best_model.csv"
        logging_dict, classwise_dict = evaluater.compute_metrics(
            model, test_dataloader, device, save_preds=True, snap_dir=snap_dir
        )

    if logging:
        logging_dict = {f"test/{k}": v for k, v in logging_dict.items()}
        classwise_dict = {k + "_test": v for k, v in classwise_dict.items()}
        wandb.log(logging_dict)
        wandb.log(classwise_dict)

    wandb.finish()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--job_id", type=int, default=0)
    parser.add_argument("--model_name", type=str, default="uni")

    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.model.backbone = args.model_name
    cfg.experiment = f"Test_{cfg.experiment}_{cfg.model.backbone}_{datetime.now().strftime('%d%m_%H%M')}_{cfg.dataset.name}_{args.job_id}"
    print(f"Experiment name: {cfg.experiment}")
    if not hasattr(cfg, "early_stopping"):
        cfg.early_stopping = MAX_EARLY_STOPPING
    if not hasattr(cfg, "save_snapshots"):
        cfg.save_snapshots = False
    if not hasattr(cfg, "save_all_ckpts"):
        cfg.save_all_ckpts = False
    if not hasattr(cfg.model, "do_ms_aug"):
        cfg.model.do_ms_aug = False
    train(cfg)
