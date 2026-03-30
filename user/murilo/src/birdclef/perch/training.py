"""Training loops for ProtoSSM and ResidualSSM.

Cross-ref: ref_yuriy_protossm_v5.py lines 1683-1998 (ProtoSSM),
           2444-2620 (ResidualSSM).
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

from birdclef.perch.ensemble import macro_auc_skip_empty, temporal_shift_tta
from birdclef.perch.losses import focal_bce_with_logits, mixup_files
from birdclef.perch.models import ProtoSSMv2, ResidualSSM

N_WINDOWS = 12
N_CLASSES = 234


def train_proto_ssm_single(
    model: ProtoSSMv2,
    emb_train: np.ndarray,
    logits_train: np.ndarray,
    labels_train: np.ndarray,
    site_ids_train: np.ndarray | None = None,
    hours_train: np.ndarray | None = None,
    emb_val: np.ndarray | None = None,
    logits_val: np.ndarray | None = None,
    labels_val: np.ndarray | None = None,
    site_ids_val: np.ndarray | None = None,
    hours_val: np.ndarray | None = None,
    file_families_train: np.ndarray | None = None,
    file_families_val: np.ndarray | None = None,
    cfg: dict | None = None,
    verbose: bool = True,
) -> tuple[ProtoSSMv2, dict]:
    """Train a single ProtoSSM model with focal loss, mixup, and SWA.

    Args:
        model: ProtoSSMv2 instance (already initialized)
        emb_train: (n_files, 12, 1536)
        logits_train: (n_files, 12, 234)
        labels_train: (n_files, 12, 234)
        cfg: training config dict

    Returns:
        model: trained model (with SWA or best state)
        history: dict with train_loss, val_loss, val_auc lists
    """
    if cfg is None:
        from birdclef.perch.config import get_protossm_train_config

        cfg = get_protossm_train_config()

    label_smoothing = cfg.get("label_smoothing", 0.0)
    mixup_alpha = cfg.get("mixup_alpha", 0.0)
    focal_gamma = cfg.get("focal_gamma", 0.0)
    swa_start_frac = cfg.get("swa_start_frac", 1.0)
    n_epochs = cfg["n_epochs"]
    swa_start_epoch = int(n_epochs * swa_start_frac)

    # Apply label smoothing to base labels
    labels_np = labels_train.copy()
    if label_smoothing > 0:
        labels_np = labels_np * (1.0 - label_smoothing) + label_smoothing / 2.0

    has_val = emb_val is not None
    if has_val:
        emb_v = torch.tensor(emb_val, dtype=torch.float32)
        logits_v = torch.tensor(logits_val, dtype=torch.float32)
        labels_v = torch.tensor(labels_val, dtype=torch.float32)
        site_v = torch.tensor(site_ids_val, dtype=torch.long) if site_ids_val is not None else None
        hour_v = torch.tensor(hours_val, dtype=torch.long) if hours_val is not None else None

    # Note: fam_v not used in val forward pass (family loss only computed on train)

    # Class weights for imbalanced data
    labels_tr_t = torch.tensor(labels_np, dtype=torch.float32)
    pos_counts = labels_tr_t.sum(dim=(0, 1))
    total = labels_tr_t.shape[0] * labels_tr_t.shape[1]
    pos_weight = ((total - pos_counts) / (pos_counts + 1)).clamp(max=cfg["pos_weight_cap"])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg["lr"],
        epochs=n_epochs,
        steps_per_epoch=1,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    best_val_loss = float("inf")
    best_state = None
    wait = 0
    history: dict[str, list] = {"train_loss": [], "val_loss": [], "val_auc": []}

    # SWA state accumulator
    swa_state = None
    swa_count = 0

    for epoch in range(n_epochs):
        # === Mixup augmentation (skip first 5 epochs for warmup) ===
        if mixup_alpha > 0 and epoch > 5:
            emb_mix, logits_mix, labels_mix, _, _, fam_mix = mixup_files(
                emb_train,
                logits_train,
                labels_np,
                site_ids_train,
                hours_train,
                file_families_train,
                alpha=mixup_alpha,
            )
        else:
            emb_mix, logits_mix, labels_mix = emb_train, logits_train, labels_np
            fam_mix = file_families_train

        emb_tr = torch.tensor(emb_mix, dtype=torch.float32)
        logits_tr = torch.tensor(logits_mix, dtype=torch.float32)
        labels_tr = torch.tensor(labels_mix, dtype=torch.float32)
        site_tr = (
            torch.tensor(site_ids_train, dtype=torch.long) if site_ids_train is not None else None
        )
        hour_tr = torch.tensor(hours_train, dtype=torch.long) if hours_train is not None else None
        fam_tr = torch.tensor(fam_mix, dtype=torch.float32) if fam_mix is not None else None

        # === Train ===
        model.train()
        species_out, family_out, _ = model(emb_tr, logits_tr, site_ids=site_tr, hours=hour_tr)

        # Primary loss: focal BCE or weighted BCE
        if focal_gamma > 0:
            loss_main = focal_bce_with_logits(
                species_out,
                labels_tr,
                gamma=focal_gamma,
                pos_weight=pos_weight[None, None, :],
            )
        else:
            loss_main = F.binary_cross_entropy_with_logits(
                species_out, labels_tr, pos_weight=pos_weight[None, None, :]
            )

        # Knowledge distillation loss
        loss_distill = F.mse_loss(species_out, logits_tr)

        # Total loss
        loss = loss_main + cfg["distill_weight"] * loss_distill

        # Taxonomic auxiliary loss
        if family_out is not None and fam_tr is not None:
            loss_family = F.binary_cross_entropy_with_logits(family_out, fam_tr)
            loss = loss + 0.1 * loss_family

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # === SWA accumulation ===
        if epoch >= swa_start_epoch:
            if swa_state is None:
                swa_state = {k: v.clone() for k, v in model.state_dict().items()}
                swa_count = 1
            else:
                for k in swa_state:
                    swa_state[k] += model.state_dict()[k]
                swa_count += 1

        # === Validate ===
        model.eval()
        with torch.no_grad():
            if has_val:
                val_out, val_fam, _ = model(emb_v, logits_v, site_ids=site_v, hours=hour_v)
                val_loss = F.binary_cross_entropy_with_logits(
                    val_out, labels_v, pos_weight=pos_weight[None, None, :]
                )

                val_pred = val_out.reshape(-1, val_out.shape[-1]).numpy()
                val_true = labels_v.reshape(-1, labels_v.shape[-1]).numpy()
                try:
                    val_auc = macro_auc_skip_empty(val_true, val_pred)
                except Exception:
                    val_auc = 0.0
            else:
                val_loss = loss
                val_auc = 0.0

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss.item())
        history["val_auc"].append(val_auc)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if verbose and (epoch + 1) % 20 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            swa_info = f" swa={swa_count}" if swa_count > 0 else ""
            print(
                f"  Epoch {epoch + 1:3d}: train={loss.item():.4f} val={val_loss.item():.4f} "
                f"auc={val_auc:.4f} lr={lr_now:.6f} wait={wait}{swa_info}"
            )

        if wait >= cfg["patience"]:
            if verbose:
                print(f"  Early stopping at epoch {epoch + 1} (best val_loss={best_val_loss:.4f})")
            break

    # Apply SWA if we accumulated enough checkpoints (>=3)
    if swa_state is not None and swa_count >= 3:
        if verbose:
            print(f"  Applying SWA (averaged {swa_count} checkpoints)")
        avg_state = {k: v / swa_count for k, v in swa_state.items()}
        model.load_state_dict(avg_state)
    elif best_state is not None:
        model.load_state_dict(best_state)

    if verbose:
        print(f"  Training complete. Best val_loss={best_val_loss:.4f}")
        with torch.no_grad():
            alphas = torch.sigmoid(model.fusion_alpha).numpy()
            print(
                f"  Fusion alpha: mean={alphas.mean():.3f} min={alphas.min():.3f} "
                f"max={alphas.max():.3f}"
            )
            print(f"  Proto temperature: {F.softplus(model.proto_temp).item():.3f}")

    return model, history


def run_proto_ssm_oof(
    emb_files: np.ndarray,
    logits_files: np.ndarray,
    labels_files: np.ndarray,
    site_ids_all: np.ndarray,
    hours_all: np.ndarray,
    file_families: np.ndarray,
    file_groups: np.ndarray,
    n_families: int,
    class_to_family: list[int],
    ssm_cfg: dict | None = None,
    train_cfg: dict | None = None,
    tta_shifts: list[int] | None = None,
    n_classes: int = N_CLASSES,
    n_windows: int = N_WINDOWS,
    verbose: bool = True,
) -> tuple[np.ndarray, list[dict], list[np.ndarray]]:
    """Run GroupKFold OOF cross-validation for ProtoSSM.

    Returns:
        oof_preds: (n_files, 12, 234) OOF predictions
        fold_histories: list of training history dicts
        fold_alphas: list of per-fold fusion alpha arrays
    """
    if ssm_cfg is None:
        from birdclef.perch.config import get_protossm_config

        ssm_cfg = get_protossm_config()
    if train_cfg is None:
        from birdclef.perch.config import get_protossm_train_config

        train_cfg = get_protossm_train_config()
    if tta_shifts is None:
        tta_shifts = [0, 1, -1]

    n_splits = train_cfg.get("oof_n_splits", 3)
    n_files = len(emb_files)

    oof_preds = np.zeros((n_files, n_windows, n_classes), dtype=np.float32)
    fold_histories = []
    fold_alphas = []

    n_unique_groups = len(set(file_groups))
    if n_unique_groups < n_splits:
        if verbose:
            print(
                f"  WARNING: Only {n_unique_groups} groups, reducing n_splits "
                f"from {n_splits} to {n_unique_groups}"
            )
        n_splits = n_unique_groups

    gkf = GroupKFold(n_splits=n_splits)
    dummy_y = np.zeros(n_files)

    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(dummy_y, dummy_y, file_groups)):
        if verbose:
            print(
                f"\n--- Fold {fold_i + 1}/{n_splits} "
                f"(train={len(train_idx)}, val={len(val_idx)}) ---"
            )

        fold_model = ProtoSSMv2(
            d_input=emb_files.shape[2],
            d_model=ssm_cfg["d_model"],
            d_state=ssm_cfg["d_state"],
            n_ssm_layers=ssm_cfg["n_ssm_layers"],
            n_classes=n_classes,
            n_windows=n_windows,
            dropout=ssm_cfg["dropout"],
            n_sites=ssm_cfg["n_sites"],
            meta_dim=ssm_cfg["meta_dim"],
            use_cross_attn=ssm_cfg.get("use_cross_attn", True),
            cross_attn_heads=ssm_cfg.get("cross_attn_heads", 4),
        )

        # Initialize prototypes from fold training data
        emb_flat_fold = emb_files[train_idx].reshape(-1, emb_files.shape[2])
        labels_flat_fold = labels_files[train_idx].reshape(-1, n_classes)
        fold_model.init_prototypes_from_data(
            torch.tensor(emb_flat_fold, dtype=torch.float32),
            torch.tensor(labels_flat_fold, dtype=torch.float32),
        )
        fold_model.init_family_head(n_families, class_to_family)

        # Train on fold
        fold_model, fold_hist = train_proto_ssm_single(
            fold_model,
            emb_files[train_idx],
            logits_files[train_idx],
            labels_files[train_idx].astype(np.float32),
            site_ids_train=site_ids_all[train_idx],
            hours_train=hours_all[train_idx],
            emb_val=emb_files[val_idx],
            logits_val=logits_files[val_idx],
            labels_val=labels_files[val_idx].astype(np.float32),
            site_ids_val=site_ids_all[val_idx],
            hours_val=hours_all[val_idx],
            file_families_train=file_families[train_idx],
            file_families_val=file_families[val_idx],
            cfg=train_cfg,
            verbose=verbose,
        )

        # OOF predictions with TTA
        fold_model.eval()
        if len(tta_shifts) > 1:
            oof_preds[val_idx] = temporal_shift_tta(
                emb_files[val_idx],
                logits_files[val_idx],
                fold_model,
                site_ids_all[val_idx],
                hours_all[val_idx],
                shifts=tta_shifts,
            )
        else:
            with torch.no_grad():
                val_emb = torch.tensor(emb_files[val_idx], dtype=torch.float32)
                val_logits = torch.tensor(logits_files[val_idx], dtype=torch.float32)
                val_sites = torch.tensor(site_ids_all[val_idx], dtype=torch.long)
                val_hours = torch.tensor(hours_all[val_idx], dtype=torch.long)
                val_out, _, _ = fold_model(val_emb, val_logits, site_ids=val_sites, hours=val_hours)
                oof_preds[val_idx] = val_out.numpy()

        fold_alphas.append(torch.sigmoid(fold_model.fusion_alpha).detach().numpy().copy())
        fold_histories.append(fold_hist)

    return oof_preds, fold_histories, fold_alphas


def train_residual_ssm(
    emb_files: np.ndarray,
    first_pass_files: np.ndarray,
    labels_files: np.ndarray,
    site_ids_all: np.ndarray,
    hours_all: np.ndarray,
    cfg: dict | None = None,
    n_sites: int = 20,
    wall_time_limit: float | None = None,
    start_time: float | None = None,
    verbose: bool = True,
) -> tuple[ResidualSSM | None, float]:
    """Train ResidualSSM on first-pass errors.

    Args:
        emb_files: (n_files, 12, 1536) embeddings
        first_pass_files: (n_files, 12, 234) first-pass ensemble logits
        labels_files: (n_files, 12, 234) ground truth
        site_ids_all: (n_files,) site indices
        hours_all: (n_files,) hours
        cfg: residual SSM config
        wall_time_limit: max wall time in minutes (skip if exceeded)
        start_time: wall clock start time

    Returns:
        res_model: trained ResidualSSM or None if skipped
        correction_weight: weight for correction (0.0 if skipped)
    """
    if cfg is None:
        from birdclef.perch.config import get_residual_ssm_config

        cfg = get_residual_ssm_config()

    # Wall-time safety gate
    if wall_time_limit is not None and start_time is not None:
        elapsed_min = (time.time() - start_time) / 60.0
        if elapsed_min > wall_time_limit:
            if verbose:
                print(
                    f"  SKIPPED ResidualSSM (wall time {elapsed_min:.1f}min > {wall_time_limit}min)"
                )
            return None, 0.0

    n_classes = first_pass_files.shape[2]

    # Compute residuals: target - sigmoid(first_pass)
    first_pass_probs = 1.0 / (1.0 + np.exp(-np.clip(first_pass_files, -30, 30)))
    residuals = labels_files.astype(np.float32) - first_pass_probs

    if verbose:
        print(
            f"  Residuals: mean={residuals.mean():.4f}, std={residuals.std():.4f}, "
            f"abs_mean={np.abs(residuals).mean():.4f}"
        )

    # Create model
    res_model = ResidualSSM(
        d_input=emb_files.shape[2],
        d_scores=n_classes,
        d_model=cfg["d_model"],
        d_state=cfg["d_state"],
        n_classes=n_classes,
        n_windows=emb_files.shape[1],
        dropout=cfg["dropout"],
        n_sites=n_sites,
        meta_dim=8,
    )

    if verbose:
        print(f"  ResidualSSM parameters: {res_model.count_parameters():,}")

    # Train/val split (15% val, seed 123)
    n_files = len(emb_files)
    n_val = max(1, int(n_files * 0.15))
    perm = torch.randperm(n_files, generator=torch.Generator().manual_seed(123))
    val_i = perm[:n_val].numpy()
    train_i = perm[n_val:].numpy()

    emb_tr = torch.tensor(emb_files[train_i], dtype=torch.float32)
    fp_tr = torch.tensor(first_pass_files[train_i], dtype=torch.float32)
    res_tr = torch.tensor(residuals[train_i], dtype=torch.float32)
    site_tr = torch.tensor(site_ids_all[train_i], dtype=torch.long)
    hour_tr = torch.tensor(hours_all[train_i], dtype=torch.long)

    emb_va = torch.tensor(emb_files[val_i], dtype=torch.float32)
    fp_va = torch.tensor(first_pass_files[val_i], dtype=torch.float32)
    res_va = torch.tensor(residuals[val_i], dtype=torch.float32)
    site_va = torch.tensor(site_ids_all[val_i], dtype=torch.long)
    hour_va = torch.tensor(hours_all[val_i], dtype=torch.long)

    optimizer = torch.optim.AdamW(res_model.parameters(), lr=cfg["lr"], weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg["lr"],
        epochs=cfg["n_epochs"],
        steps_per_epoch=1,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    t0 = time.time()
    for epoch in range(cfg["n_epochs"]):
        res_model.train()
        correction = res_model(emb_tr, fp_tr, site_ids=site_tr, hours=hour_tr)
        loss = F.mse_loss(correction, res_tr)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(res_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        res_model.eval()
        with torch.no_grad():
            val_corr = res_model(emb_va, fp_va, site_ids=site_va, hours=hour_va)
            val_loss = F.mse_loss(val_corr, res_va)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = {k: v.clone() for k, v in res_model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if verbose and (epoch + 1) % 20 == 0:
            print(
                f"  ResidualSSM epoch {epoch + 1}: "
                f"train={loss.item():.6f} val={val_loss.item():.6f} wait={wait}"
            )

        if wait >= cfg["patience"]:
            if verbose:
                print(f"  ResidualSSM early stop at epoch {epoch + 1}")
            break

    if best_state is not None:
        res_model.load_state_dict(best_state)

    res_time = time.time() - t0
    if verbose:
        print(f"  ResidualSSM training: {res_time:.1f}s, best val MSE: {best_val_loss:.6f}")

    return res_model, cfg["correction_weight"]
