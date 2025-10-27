# ==================== Standard library ====================
import os
import json
import time
import inspect

# ==================== General scientific stack ====================
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==================== PyTorch ====================
import torch
from torch.optim import Adam
from torch.nn import MSELoss
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm

# ==================== PyTorch Geometric ====================
from torch_geometric.data import Data as PygData, Batch as PygBatch
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.utils import (
    remove_self_loops, coalesce, degree, add_self_loops,
    subgraph, to_undirected
)


def _as_pyg_batch(x):
    """Data / list[Data] / Batch / dict(Data-like) → Batch 로 통일"""
    if x is None:
        return None
    if isinstance(x, PygBatch):
        return x
    if isinstance(x, PygData):
        return PygBatch.from_data_list([x])
    if isinstance(x, (list, tuple)):
        return PygBatch.from_data_list(list(x))
    if isinstance(x, dict) and ("x" in x and "edge_index" in x):
        return PygBatch.from_data_list([PygData(**x)])
    raise TypeError(f"Unsupported graph type for _as_pyg_batch: {type(x)}")
# ───────────────────────── 공용 유틸 (당신 코드 유지/활용) ─────────────────────────
def _move_to_device(x, device):
    if hasattr(x, "to"):
        try:
            return x.to(device)
        except TypeError:
            return x
    if isinstance(x, (list, tuple)):
        return type(x)(_move_to_device(v, device) for v in x)
    if isinstance(x, dict):
        return {k: _move_to_device(v, device) for k, v in x.items()}
    return x

def _first_present(d: dict, keys: list[str]):
    """dict에서 None이 아닌 첫 값을 반환 (Tensor에도 안전)"""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

def _unpack_batch(batch):
    """
    다양한 포맷의 배치를 받아
    (sequences, graphs, protein_graphs, labels, tree_batch)로 반환.
    """
    sequences = graphs = protein_graphs = labels = tree_batch = None

    if isinstance(batch, dict):
        sequences       = _first_present(batch, ["protein_sequences", "sequences", "sequence"])
        graphs          = _first_present(batch, ["ligand_batch", "graphs", "graph"])
        protein_graphs  = _first_present(batch, ["protein_graphs", "protein_graph"])
        labels          = _first_present(batch, ["y", "labels", "label"])
        tree_batch      = _first_present(batch, ["tree_batch", "tree"])

    elif isinstance(batch, (list, tuple)):
        # 구형 튜플 포맷 대응
        if len(batch) == 5:
            sequences, graphs, protein_graphs, labels, tree_batch = batch
        elif len(batch) == 4:
            a, b, c, d = batch
            # c가 그래프처럼 보이면 protein_graphs로 간주
            is_graph_like = hasattr(c, "edge_index") or (hasattr(c, "to") and not torch.is_tensor(c))
            if is_graph_like:
                sequences, graphs, protein_graphs, labels = a, b, c, d
            else:
                sequences, graphs, labels, tree_batch = a, b, c, d
        elif len(batch) == 3:
            sequences, graphs, labels = batch
        else:
            raise ValueError(f"Unexpected batch length: {len(batch)}")
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    # 타입 정리
    if isinstance(sequences, str):
        sequences = [sequences]
    if labels is not None and hasattr(labels, "shape"):
        labels = labels.view(-1)

    # 결측 진단
    missing = []
    if sequences is None:      missing.append("protein_sequences/sequences/sequence")
    if graphs is None:         missing.append("ligand_batch/graphs/graph")
    if labels is None:         missing.append("y/labels/label")
    if missing:
        keys = list(batch.keys()) if isinstance(batch, dict) else f"tuple_len={len(batch)}"
        raise KeyError(f"[_unpack_batch] Missing keys: {missing}. Batch keys: {keys}")

    return sequences, graphs, protein_graphs, labels, tree_batch

def _model_accepts_arg(model, arg_name: str) -> bool:
    try:
        sig = inspect.signature(model.forward)
        return arg_name in sig.parameters
    except (ValueError, AttributeError):
        return False

# ───────────────────────── 평가 지표 ─────────────────────────
def _regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)   # <-- add
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)   # <-- add
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    denom = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / denom) if denom > 0 else 0.0
    return dict(mse=float(mse), rmse=rmse, mae=mae, r2=r2)

def _get_curr_lr(optimizer):
    # 첫 param_group 기준
    return optimizer.param_groups[0].get("lr", None)

# ───────────────────────── 에폭 단위 학습/검증 ─────────────────────────
def train_epoch(
    model, loader, optimizer, criterion, device,
    use_sepool: bool = True,
    cluster_level_for_readout: int | None = 1,
    return_attn_weights: bool = False,
    return_cluster_levels: bool = False,
    scheduler=None, scheduler_step_on: str = "batch",  # "batch" or "epoch" or None
    pool_reg_lambda: float = 1e-3,
    grad_clip: float | None = None,
    amp: bool = True,
    log_interval: int = 0,
):
    model.train()
    scaler = GradScaler(enabled=amp)
    accepts_protein = _model_accepts_arg(model, "protein_graphs")
    accepts_tree    = _model_accepts_arg(model, "tree_batch")

    running_loss  = 0.0
    running_items = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for step, batch in enumerate(pbar, 1):
        # ⚠️ 현재 _unpack_batch 반환 순서: (sequences, graphs, protein_graphs, labels, tree_batch)
        sequences, graphs, protein_graphs, labels, tree_batch = _unpack_batch(batch)

        # --- 정규화: 그래프를 항상 PyG Batch로 ---
        graphs         = _as_pyg_batch(graphs)
        protein_graphs = _as_pyg_batch(protein_graphs) if protein_graphs is not None else None

        # --- 디바이스 이동 ---
        labels = _move_to_device(labels, device)
        graphs = _move_to_device(graphs, device)
        if protein_graphs is not None:
            protein_graphs = _move_to_device(protein_graphs, device)
        if (use_sepool is True) and (tree_batch is not None):
            tree_batch = _move_to_device(tree_batch, device)

        # --- forward kwargs 구성 ---
        kwargs = dict(
            cluster_level_for_readout=cluster_level_for_readout,
            return_attn_weights=return_attn_weights,
            return_cluster_levels=return_cluster_levels,
        )
        if use_sepool and (tree_batch is not None) and accepts_tree:
            kwargs["tree_batch"] = tree_batch
        if (protein_graphs is not None) and accepts_protein:
            kwargs["protein_graphs"] = protein_graphs

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            outputs = model(sequences, graphs, **kwargs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            base_loss = criterion(outputs.squeeze(-1), labels)

            # --- 정규화 항: 그래프 유지 & dtype/device 정렬 ---
            if pool_reg_lambda and hasattr(model, "pool_regularizer"):
                reg = model.pool_regularizer()  # 텐서여야 함 (그래프 유지!)
                # 손실 dtype/device와 맞추기
                reg = reg.to(dtype=base_loss.dtype, device=base_loss.device)
            else:
                reg = torch.zeros((), dtype=base_loss.dtype, device=base_loss.device)

            loss = base_loss + pool_reg_lambda * reg

        # --- backward & step ---
        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None and scheduler_step_on == "batch":
            scheduler.step()

        # --- 통계 ---
        bs = labels.size(0) if torch.is_tensor(labels) else len(labels)
        running_loss  += float(loss.item()) * bs
        running_items += bs

        if log_interval and (step % log_interval == 0):
            lr = _get_curr_lr(optimizer)
            pbar.set_description(
                f"train | loss {running_loss / running_items:.4f} "
                f"(base {base_loss.detach().item():.4f}, reg {reg.detach().item():.4f}) | lr {lr:.2e}"
            )

        # (옵션) 커스텀 훅
        if hasattr(model, "temperature_clamp") and callable(getattr(model, "temperature_clamp")):
            model.temperature_clamp()

    if scheduler is not None and scheduler_step_on == "epoch":
        scheduler.step()

    epoch_loss = running_loss / max(1, running_items)
    return dict(loss=epoch_loss)



@torch.no_grad()
def validate_epoch(
    model, loader, criterion, device,
    use_sepool: bool = True,
    cluster_level_for_readout: int | None = 1,
    return_attn_weights: bool = False,
    return_cluster_levels: bool = False,
    amp: bool = True,
):
    model.eval()
    accepts_protein = _model_accepts_arg(model, "protein_graphs")
    accepts_tree = _model_accepts_arg(model, "tree_batch")

    total_loss = 0.0
    total_items = 0
    ys = []
    preds = []

    for batch in loader:
        sequences, graphs, protein_graphs, labels, tree_batch = _unpack_batch(batch)

        labels = _move_to_device(labels, device)
        graphs = _move_to_device(graphs, device)
        if protein_graphs is not None:
            protein_graphs = _move_to_device(protein_graphs, device)
        if tree_batch is not None:
            tree_batch = _move_to_device(tree_batch, device)

        kwargs = dict(
            cluster_level_for_readout=cluster_level_for_readout,
            return_attn_weights=return_attn_weights,
            return_cluster_levels=return_cluster_levels,
        )
        if use_sepool and tree_batch is not None and accepts_tree:
            kwargs["tree_batch"] = tree_batch
        if protein_graphs is not None and accepts_protein:
            kwargs["protein_graphs"] = protein_graphs

        with autocast(enabled=amp):
            outputs = model(sequences, graphs, **kwargs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs.squeeze(-1), labels)

        bs = labels.size(0) if torch.is_tensor(labels) else len(labels)
        total_loss += float(loss.item()) * bs
        total_items += bs

        ys.append(labels.detach().float().cpu().numpy().reshape(-1))
        preds.append(outputs.detach().float().cpu().numpy().reshape(-1))

    epoch_loss = total_loss / max(1, total_items)
    y_true = np.concatenate(ys, axis=0) if ys else np.array([])
    y_pred = np.concatenate(preds, axis=0) if preds else np.array([])
    metrics = _regression_metrics(y_true, y_pred) if len(y_true) else dict(mse=np.nan, rmse=np.nan, mae=np.nan, r2=np.nan)
    metrics["loss"] = epoch_loss
    return metrics

# ───────────────────────── 전체 학습 루프 (로깅/체크포인트) ─────────────────────────
def fit(
    model,
    train_loader,
    val_loader=None,
    test_loader=None,
    optimizer=None,
    criterion=None,
    device="cuda",
    *,
    epochs: int = 10,
    scheduler=None,
    scheduler_step_on: str = "batch",   # "batch" | "epoch" | None
    grad_clip: float | None = None,
    amp: bool = True,
    log_interval: int = 0,
    evaluate_every: int = 1,
    evaluate_metric: str = "rmse",      # "loss" | "rmse" | "mse" | "mae" | "r2"
    checkpoint_dir: str | None = None,
    run_id: str | int = 0,
    early_stopping_patience: int | None = None,
    pool_reg_lambda: float = 1e-3,
    use_sepool: bool = True,
    cluster_level_for_readout: int | None = 1,
    return_attn_weights: bool = False,
    return_cluster_levels: bool = False,
):
    """
    반환: history(dict), best_ckpt_path(str|None)
    """
    assert optimizer is not None and criterion is not None, "optimizer와 criterion을 전달하세요."

    os.makedirs(checkpoint_dir or ".", exist_ok=True)
    log_path = os.path.join(checkpoint_dir or ".", f"train_log_{run_id}.jsonl")
    best_path = os.path.join(checkpoint_dir or ".", f"model_best_{run_id}.pt")
    last_path = os.path.join(checkpoint_dir or ".", f"model_last_{run_id}.pt")

    def _is_better(curr, best):
        if np.isnan(curr):  # NaN 보호
            return False
        if evaluate_metric in ("loss", "rmse", "mse", "mae"):
            return curr < best
        else:  # r2 등 큰 게 좋음
            return curr > best

    best_val = np.inf if evaluate_metric in ("loss", "rmse", "mse", "mae") else -np.inf
    best_epoch = -1
    best_ckpt_path = None
    no_improve = 0

    history = []

    for epoch in range(1, epochs + 1):
        # 학습
        train_stats = train_epoch(
            model, train_loader, optimizer, criterion, device,
            use_sepool=use_sepool,
            cluster_level_for_readout=cluster_level_for_readout,
            return_attn_weights=return_attn_weights,
            return_cluster_levels=return_cluster_levels,
            scheduler=scheduler, scheduler_step_on=scheduler_step_on,
            grad_clip=grad_clip, amp=amp, log_interval=log_interval,
            pool_reg_lambda=pool_reg_lambda,
        )

        # 검증/테스트
        val_stats = {}
        test_stats = {}
        if (val_loader is not None) and (epoch % evaluate_every == 0):
            val_stats = validate_epoch(
                model, val_loader, criterion, device,
                use_sepool=use_sepool,
                cluster_level_for_readout=cluster_level_for_readout,
                return_attn_weights=return_attn_weights,
                return_cluster_levels=return_cluster_levels,
                amp=amp,
            )
            # 베스트 모델 갱신
            target = val_stats.get(evaluate_metric, np.inf)
            if _is_better(target, best_val):
                best_val = target
                best_epoch = epoch
                torch.save(model.state_dict(), best_path)
                best_ckpt_path = best_path
                no_improve = 0
            else:
                no_improve += 1

            # 검증과 동시에 테스트도 보고 싶다면(옵션)
            if test_loader is not None:
                test_stats = validate_epoch(
                    model, test_loader, criterion, device,
                    use_sepool=use_sepool,
                    cluster_level_for_readout=cluster_level_for_readout,
                    return_attn_weights=return_attn_weights,
                    return_cluster_levels=return_cluster_levels,
                    amp=amp,
                )

        # 마지막 체크포인트 항상 저장
        torch.save(model.state_dict(), last_path)

        # 콘솔 로그
        curr_lr = _get_curr_lr(optimizer)
        head = f"[epoch {epoch:03d}] lr={curr_lr:.2e}  train_loss={train_stats['loss']:.4f}"
        if 'base_loss' in train_stats and 'reg' in train_stats:
            head += f" (base {train_stats['base_loss']:.4f}, reg {train_stats['reg']:.4f} * λ={pool_reg_lambda:g})"
        tail_v = ""
        tail_t = ""
        if val_stats:
            tail_v = " | val " + ", ".join(f"{k}={val_stats[k]:.4f}" for k in ["loss","rmse","mse","mae","r2"] if k in val_stats)
        if test_stats:
            tail_t = " | test " + ", ".join(f"{k}={test_stats[k]:.4f}" for k in ["loss","rmse","mse","mae","r2"] if k in test_stats)
        print(head + tail_v + tail_t)

        # 파일 로그(JSONL)
        rec = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": epoch,
            "lr": curr_lr,
            "train": train_stats,
            "val": val_stats,
            "test": test_stats,
            "best_epoch": best_epoch,
            "best_val": best_val,
            "evaluate_metric": evaluate_metric,
            "pool_reg_lambda": pool_reg_lambda,  # ★ 선택: 실험 재현 위해 기록
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        # 조기 종료
        if (early_stopping_patience is not None) and (val_loader is not None):
            if no_improve >= early_stopping_patience:
                print(f"Early stopping triggered. No improvement in {early_stopping_patience} eval steps.")
                break

    return dict(history=history, best_epoch=best_epoch, best_val=best_val), best_ckpt_path
