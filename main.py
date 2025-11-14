# -*- coding: utf-8 -*-
# ==================== Standard library ====================
import os
import sys
import json
import argparse
from pathlib import Path
import torch.multiprocessing as mp

# ==================== PyTorch ====================
import torch
from torch.optim import Adam
from torch.nn import MSELoss

# ==================== Local modules ====================
sys.path.append(os.getcwd())  # ensure local 'modules' is importable
from modules.dta_models import MultimodalBindingAffinityModel
from modules.dta_dataset import PrecomputedBindingDataset
from utils import collate_binding, set_seed, positive_int, bool_flag, register_safe_globals
from sampler import ShardAwareBatchSampler
from trainer import fit

# ==================== Hugging Face ====================
from huggingface_hub import login

# ------------------ small dataset util ------------------
import torch
from torch.utils.data import Dataset
from modules.dualgraph.dataset import DGData
torch.serialization.add_safe_globals([DGData])

# main5.py 상단 import들 아래 어딘가 (torch import 이후) -----------------
import importlib, inspect
import torch

########################################################################################################################
# 1) 프로젝트 커스텀 클래스 (있을 때만)
try:
    from modules.dualgraph.dataset import DGData
except Exception:
    DGData = None

# 2) torch_geometric 클래스들 (버전별 경로 호환)
try:
    from torch_geometric.data.data import DataEdgeAttr
except Exception:
    DataEdgeAttr = None

try:
    from torch_geometric.data import Data
except Exception:
    Data = None

safe_list = []
for cls in (DGData, DataEdgeAttr, Data):
    if cls is not None:
        safe_list.append(cls)

if safe_list:
    torch.serialization.add_safe_globals(safe_list)
#########################################################################################################################

register_safe_globals()

# ------------------ arg parsing ------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="Train MultimodalBindingAffinityModel on precomputed DTI shards"
    )
    # Where precomputed lives
    p.add_argument("--precomputed_root", type=str, default="./precomputed",
                   help="Base directory that contains <data_type>_s<seed>/train|val|test")
    p.add_argument("--seed", type=int, default=42,
                   help="Used for directory naming and for model seeding (not for splitting)")

    # Run / IO
    p.add_argument("--checkpoint_dir", type=str, default="./ckpt/runs/exp1")
    p.add_argument("--run_id", type=str, default="0")
    p.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"), help="HuggingFace token (or env HF_TOKEN)")

    # Device
    p.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"])

    # DataLoader / batching
    p.add_argument("--batch_size", type=positive_int, default=32)
    p.add_argument("--num_workers", type=int,
                   default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--pin_memory", type=bool_flag, default=True)
    p.add_argument("--persistent_workers", type=bool_flag, default=None)
    p.add_argument("--prefetch_factor", type=int, default=None)

    # Training
    p.add_argument("--epochs", type=positive_int, default=100)
    p.add_argument("--evaluate_every", type=positive_int, default=1)
    p.add_argument("--early_stopping_patience", type=int, default=10)
    p.add_argument("--evaluate_metric", type=str, default="rmse",
                   choices=["loss","rmse","mse","mae","r2"])
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Optimizer LRs
    p.add_argument("--lr_esm", type=float, default=4e-5)
    p.add_argument("--lr_gnn", type=float, default=2e-4)
    p.add_argument("--lr_other", type=float, default=4e-4)

    # Scheduler
    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine","none"])
    p.add_argument("--scheduler_step_on", type=str, default="batch", choices=["batch","epoch"])

    # Model (shared)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--fusion_dim", type=int, default=256)
    p.add_argument("--esm_model_name", type=str, default="esmc_300m")
    p.add_argument("--dropout_fc", type=float, default=0.1)
    p.add_argument("--freeze_esm", type=bool_flag, default=True)
    p.add_argument("--num_unfrozen_layers", type=int, default=2)
    p.add_argument("--max_len_cap", type=int, default=1024)

    # DualGraph GNN
    p.add_argument("--mlp_hidden_size", type=int, default=512)
    p.add_argument("--mlp_layers", type=int, default=2)
    p.add_argument("--use_layer_norm", type=bool_flag, default=False)
    p.add_argument("--use_face", type=bool_flag, default=True)
    p.add_argument("--dropedge_rate", type=float, default=0.1)
    p.add_argument("--dropnode_rate", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--dropnet", type=float, default=0.1)
    p.add_argument("--global_reducer", type=str, default="sum")
    p.add_argument("--node_reducer", type=str, default="sum")
    p.add_argument("--face_reducer", type=str, default="sum")
    p.add_argument("--graph_pooling", type=str, default="sum")
    p.add_argument("--node_attn", type=bool_flag, default=True)
    p.add_argument("--face_attn", type=bool_flag, default=True)

    # Ligand encoder
    p.add_argument("--ligand_gnn_type", type=str, default="dualgraph")
    p.add_argument("--ligand_hidden_dim", type=int, default=512)
    p.add_argument("--ligand_num_layers", type=int, default=3)

    # Protein structure + pooling
    p.add_argument("--use_protein_graph_structure", type=bool_flag, default=True)
    p.add_argument("--protein_mp_layers", type=int, default=2)
    p.add_argument("--protein_pool_type", type=str, default="sag",
                   choices=["sag","mincut","diffpool","none"])
    p.add_argument("--pool_reg_lambda", type=float, default=1e-3)
    p.add_argument("--protein_pool_ratio", type=float, default=0.6)
    p.add_argument("--protein_pool_min_nodes", type=int, default=8)

    p.add_argument("--cluster_level_for_readout", type=int, default=1)

    # Debug / NaN
    p.add_argument("--debug_nan", type=bool_flag, default=True)
    p.add_argument("--nan_policy", type=str, default="raise", choices=["raise","warn","silent"])

    return p


# ------------------ main ------------------
def main():
    args = build_parser().parse_args()
    set_seed(args.seed)

    # ENV
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        mp.set_start_method("spawn", force=True)  # safer w/ CUDA + dataloader
    except RuntimeError:
        pass

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # DataLoader runtime flags (auto)
    auto_pin_memory = (device.type == "cuda")
    pin_memory = auto_pin_memory if args.pin_memory is None else bool(args.pin_memory)

    auto_persistent = (args.num_workers > 0)
    persistent_workers = auto_persistent if args.persistent_workers is None else bool(args.persistent_workers)

    prefetch_kwargs = {}
    if args.num_workers > 0 and args.prefetch_factor is not None:
        prefetch_kwargs["prefetch_factor"] = int(args.prefetch_factor)

    # HF Login (optional)
    if args.hf_token:
        try:
            login(args.hf_token)
        except Exception as e:
            print(f"[WARN] HuggingFace login failed: {e}")
    else:
        print("[INFO] HF token not provided. Skipping login.")

    # ------------------ Precomputed datasets ------------------
    root = Path(args.precomputed_root)
    train_dir = root / "train"
    val_dir   = root / "val"
    test_dir  = root / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Train split not found at {train_dir}. Did you run preprocessing?")
    if not val_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Val/Test split missing under {root}.")

    train_dataset = PrecomputedBindingDataset(str(train_dir))
    val_dataset   = PrecomputedBindingDataset(str(val_dir))
    test_dataset  = PrecomputedBindingDataset(str(test_dir))

    common_loader_kwargs = dict(
        batch_size=args.batch_size,
        collate_fn=collate_binding,   # 기존 collate 재사용 (배치 후 coalesce 방어막 권장)
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if args.num_workers > 0 else False,
        **prefetch_kwargs,
    )

    train_batch_sampler = ShardAwareBatchSampler(train_dataset, args.batch_size, drop_last=False, shuffle=True,
                                                 seed=args.seed)
    val_batch_sampler = ShardAwareBatchSampler(val_dataset, args.batch_size, drop_last=False, shuffle=False,
                                               seed=args.seed)
    test_batch_sampler = ShardAwareBatchSampler(test_dataset, args.batch_size, drop_last=False, shuffle=False,
                                                seed=args.seed)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        collate_fn=collate_binding,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if args.num_workers > 0 else False,
        **prefetch_kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_sampler=val_batch_sampler,
        collate_fn=collate_binding,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if args.num_workers > 0 else False,
        **prefetch_kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_sampler=test_batch_sampler,
        collate_fn=collate_binding,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if args.num_workers > 0 else False,
        **prefetch_kwargs,
    )

    # ------------------ Model ------------------
    model = MultimodalBindingAffinityModel(
        # common
        in_dim=None,
        hidden_dim=args.hidden_dim,
        fusion_dim=args.fusion_dim,
        ddi=None,
        esm_model_name=args.esm_model_name,
        dropout_fc=args.dropout_fc,
        device=args.device,
        freeze_esm=args.freeze_esm,
        num_unfrozen_layers=args.num_unfrozen_layers,
        max_len_cap=args.max_len_cap,

        # DualGraph
        mlp_hidden_size=args.mlp_hidden_size,
        mlp_layers=args.mlp_layers,
        use_layer_norm=args.use_layer_norm,
        use_face=args.use_face,
        dropedge_rate=args.dropedge_rate,
        dropnode_rate=args.dropnode_rate,
        dropout=args.dropout,
        dropnet=args.dropnet,
        global_reducer=args.global_reducer,
        node_reducer=args.node_reducer,
        face_reducer=args.face_reducer,
        graph_pooling=args.graph_pooling,
        node_attn=args.node_attn,
        face_attn=args.face_attn,

        # Ligand encoder
        ligand_gnn_type=args.ligand_gnn_type,
        ligand_in_dim=None,
        ligand_hidden_dim=args.ligand_hidden_dim,
        ligand_num_layers=args.ligand_num_layers,

        # Protein structure/pooling
        use_protein_graph_structure=args.use_protein_graph_structure,
        protein_mp_layers=args.protein_mp_layers,
        protein_pool_type=args.protein_pool_type,
        protein_pool_ratio=args.protein_pool_ratio,
        protein_pool_min_nodes=args.protein_pool_min_nodes,

        # Debug/NaN
        debug_nan=args.debug_nan,
        nan_policy=args.nan_policy,
    ).to(device)

    # ------------------ Optimizer ------------------
    param_groups = [
        {"params": [p for p in model.esm.parameters() if p.requires_grad], "lr": args.lr_esm},
        {"params": model.gnn.parameters(),          "lr": args.lr_gnn},
        {"params": model.protein_proj.parameters(), "lr": args.lr_other},
        {"params": model.ligand_proj.parameters(),  "lr": args.lr_other},
        {"params": model.cross_attn.parameters(),   "lr": args.lr_other},
        {"params": model.regressor.parameters(),    "lr": args.lr_other},
    ]
    optimizer = Adam(param_groups)
    criterion = MSELoss()

    # ------------------ Scheduler ------------------
    if args.scheduler == "cosine":
        total_steps = len(train_loader) * args.epochs if args.scheduler_step_on == "batch" else args.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    else:
        scheduler = None

    # ------------------ Train ------------------
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_out = ckpt_dir / f"config_run_{args.run_id}.json"
    with open(config_out, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    history, best_path = fit(
        model,
        train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=args.epochs,
        scheduler=scheduler,
        scheduler_step_on=args.scheduler_step_on,
        grad_clip=args.grad_clip,
        log_interval=args.log_interval,
        evaluate_every=args.evaluate_every,
        evaluate_metric=args.evaluate_metric,
        checkpoint_dir=str(ckpt_dir),
        run_id=args.run_id,
        early_stopping_patience=args.early_stopping_patience,
        cluster_level_for_readout=args.cluster_level_for_readout,
        pool_reg_lambda=args.pool_reg_lambda,
    )

    print("best checkpoint:", best_path)


if __name__ == "__main__":
    # matmul / conv TF32 (Ampere+)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # PyTorch 2.x
    try:
        torch.set_float32_matmul_precision('high')  # 'high' or 'medium'
    except Exception:
        pass
    main()
