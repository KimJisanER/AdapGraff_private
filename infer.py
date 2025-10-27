# infer.py
# -*- coding: utf-8 -*-
# ==================== Standard library ====================
import os
import sys
import json
import argparse
from pathlib import Path

# ==================== Scientific stack ====================
import numpy as np
from sklearn.model_selection import train_test_split

# ==================== PyTorch ====================
import torch
from torch.optim import Adam
from torch.nn import MSELoss

# ==================== Hugging Face ====================
from huggingface_hub import login

# ==================== TDC ====================
from tdc.multi_pred import DTI

# ==================== Local modules ====================
sys.path.append(os.getcwd())  # ensure local 'modules' is importable
from modules.dta_models_SEP9 import MultimodalBindingAffinityModel
from modules.dta_dataset_SEP2 import BindingDataset
from utils import collate_binding
from trainer import fit
from typing import Optional
import re
import torch

def _remap_checkpoint_keys(sd: dict) -> dict:
    new_sd = {}
    for k, v in sd.items():
        nk = k

        # 1) dualgraph edge_model 네이밍 변경 보정
        #    submodule_list -> submodule.module_list
        if ".submodule_list." in nk:
            nk = nk.replace(".submodule_list.", ".submodule.module_list.")

        # 2) 혹시 반대 방향으로 저장된 ckpt가 있을 때도 커버
        nk = nk.replace(".submodule.module_list.", ".submodule.module_list.")

        # 3) DataParallel prefix 제거 (예방적)
        if nk.startswith("module."):
            nk = nk[len("module."):]

        new_sd[nk] = v
    return new_sd

def load_ckpt_with_remap(model, ckpt_path: str, verbose=True):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # 다양한 포맷에서 state_dict 추출
    sd = None
    for k in ("state_dict", "model", "model_state_dict", "net", "module"):
        if isinstance(ckpt, dict) and k in ckpt and isinstance(ckpt[k], dict):
            sd = ckpt[k]; break
    if sd is None and isinstance(ckpt, dict):
        sd = ckpt

    if not isinstance(sd, dict):
            raise RuntimeError("Checkpoint does not contain a dict-like state_dict")

    # 키 리매핑
    sd_remap = _remap_checkpoint_keys(sd)

    # 로드
    missing, unexpected = model.load_state_dict(sd_remap, strict=False)
    if verbose:
        if missing:
            print(f"[INFO] Missing keys({len(missing)}): {missing[:10]}{' ...' if len(missing)>10 else ''}")
        if unexpected:
            print(f"[INFO] Unexpected keys({len(unexpected)}): {unexpected[:10]}{' ...' if len(unexpected)>10 else ''}")
# ------------------ utilities ------------------

# ====== [ADD] 엄격 파서: 먼저 정확 튜플 시도, 실패 시 기존 느슨 파서로 ======
def extract_all_strict(outputs):
    """
    기대 튜플:
      (y_pred, (attn_pl, attn_lp), level_feats, protein_pool_info)
    를 우선 시도. 실패 시 _extract_everything 로 폴백.
    """
    y_pred = attn_tuple = level_feats = protein_pool_info = None
    if isinstance(outputs, tuple):
        if len(outputs) == 4 and isinstance(outputs[1], (tuple, list)) and len(outputs[1]) == 2:
            y_pred, attn_pair, level_feats, protein_pool_info = outputs
            attn_tuple = (attn_pair[0], attn_pair[1])
            return y_pred, attn_tuple, level_feats, protein_pool_info
        # 가끔 (y, (attn_pl, attn_lp), ppi, level) 순서 뒤집힌 케이스 방어
        if len(outputs) == 4 and isinstance(outputs[1], (tuple, list)) and len(outputs[1]) == 2:
            # 위와 동일하지만 혹시 tail 순서가 바뀐 경우를 대비해 한번 더 검사
            pass
    # dict / list 등은 기존 파서에 맡김
    return _extract_everything(outputs)


# ====== [ADD] 출력 구조 덤프 (INFER_DEBUG=1일 때만 자세히) ======
def dump_outputs_shape(outputs, prefix="[OUT]"):
    def _t(x):
        if torch.is_tensor(x):
            return f"Tensor{tuple(x.shape)} {x.dtype} {x.device}"
        return type(x).__name__
    print(f"{prefix} type={type(outputs).__name__}")
    if isinstance(outputs, tuple):
        print(f"{prefix} tuple-len={len(outputs)}")
        for i, o in enumerate(outputs):
            if isinstance(o, tuple):
                print(f"{prefix}  [{i}] tuple-len={len(o)}")
                for j, oo in enumerate(o):
                    print(f"{prefix}    [{i}][{j}] {_t(oo)}")
            else:
                print(f"{prefix}  [{i}] {_t(o)}")
    elif isinstance(outputs, dict):
        print(f"{prefix} dict-keys={list(outputs.keys())[:10]}")
    else:
        print(f"{prefix} {_t(outputs)}")


# ====== [ADD] 모델/체크포인트 어텐션 진단 ======
def debug_attn_capabilities(model, ckpt_path: Optional[str]):
    try:
        import modules.dta_models_SEP9 as m
        print(f"[DEBUG] Loaded model module file: {getattr(m, '__file__', '<?>')}")
    except Exception as e:
        print(f"[DEBUG] Could not inspect model module path: {e}")

    print(f"[DEBUG] Has cross_attn_pl? {hasattr(model, 'cross_attn_pl')}")
    print(f"[DEBUG] Has cross_attn_lp? {hasattr(model, 'cross_attn_lp')}")

    if ckpt_path is not None and Path(ckpt_path).exists():
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            sd = None
            for k in ("state_dict", "model", "model_state_dict", "net", "module"):
                if isinstance(ckpt, dict) and k in ckpt and isinstance(ckpt[k], dict):
                    sd = ckpt[k]; break
            if sd is None and isinstance(ckpt, dict):
                sd = ckpt
            if isinstance(sd, dict):
                has_pl = any(str(k).startswith("cross_attn_pl.") for k in sd.keys())
                has_lp = any(str(k).startswith("cross_attn_lp.") for k in sd.keys())
                print(f"[DEBUG] CKPT has cross_attn_pl.*? {has_pl}")
                print(f"[DEBUG] CKPT has cross_attn_lp.*? {has_lp}")
            else:
                print("[DEBUG] CKPT has no state_dict-like content.")
        except Exception as e:
            print(f"[DEBUG] CKPT inspect failed: {e}")
    else:
        print("[DEBUG] No ckpt to inspect or path missing.")


def move_to_device(obj, device):
    """Tensor, PyG Batch/Data, DGLGraph, dict/list/tuple 모두 안전 이관"""
    if obj is None:
        return None
    if torch.is_tensor(obj):
        return obj.to(device)
    # PyG / DGL / custom: 대부분 .to(device) 지원
    if hasattr(obj, "to") and callable(getattr(obj, "to")):
        try:
            return obj.to(device)
        except TypeError:
            # 일부 객체는 .to(dtype=...) 시그니처만 있는 경우가 있어 방어
            pass
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        elem = [move_to_device(v, device) for v in obj]
        return type(obj)(elem) if not isinstance(obj, list) else elem
    return obj  # 이동 불필요/불명


def set_seed(seed: int):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def positive_int(v):
    iv = int(v)
    if iv <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return iv

def bool_flag(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes","y","true","t","1"):  return True
    if v in ("no","n","false","f","0"):  return False
    raise argparse.ArgumentTypeError("boolean value expected")

# ------------------ arg parsing ------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="Train MultimodalBindingAffinityModel on BindingDB_Kd with CLI hyperparameters"
    )
    # Run / IO
    p.add_argument("--checkpoint_dir", type=str,
                   default="./ckpt/runs/exp1")
    p.add_argument("--run_id", type=str, default="0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"])
    p.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"), help="HuggingFace token (or env HF_TOKEN)")
    # Data
    p.add_argument("--dataset_name", type=str, default="BindingDB_Kd")
    p.add_argument("--max_target_len", type=int, default=1000)
    p.add_argument("--batch_size", type=positive_int, default=32)
    p.add_argument("--protein_weight_mode", type=str, default="linear",
                   choices=["linear","inverse","gaussian","binary","raw"])
    # Training
    p.add_argument("--epochs", type=positive_int, default=100)
    p.add_argument("--evaluate_every", type=positive_int, default=1)
    p.add_argument("--early_stopping_patience", type=int, default=10)
    p.add_argument("--evaluate_metric", type=str, default="rmse",
                   choices=["loss","rmse","mse","mae","r2"])
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", type=bool_flag, default=False)
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
    p.add_argument("--som_mode", type=bool_flag, default=False)
    # Ligand encoder
    p.add_argument("--ligand_gnn_type", type=str, default="dualgraph")
    p.add_argument("--ligand_hidden_dim", type=int, default=512)
    p.add_argument("--ligand_num_layers", type=int, default=3)
    # Ligand pooling (SEP 기본)
    p.add_argument("--ligand_pool_type", type=str, default="sep", choices=["sep","diffpool","mincut","none"])
    p.add_argument("--tree_depth", type=int, default=3)
    p.add_argument("--sepool_hidden_dim", type=int, default=256)
    # Protein structure + pooling
    p.add_argument("--use_protein_graph_structure", type=bool_flag, default=True)
    p.add_argument("--protein_mp_layers", type=int, default=2)
    p.add_argument("--protein_pool_type", type=str, default="sag",
                   choices=["sag","asap","mincut","diffpool","none"])
    p.add_argument("--protein_pool_ratio", type=float, default=0.6)
    p.add_argument("--protein_pool_min_nodes", type=int, default=8)
    # Trainer flags that feed into forward
    p.add_argument("--use_sepool", type=bool_flag, default=True, help="Pass tree_batch to model (if provided)")
    p.add_argument("--cluster_level_for_readout", type=int, default=1)
    # Debug / NaN
    p.add_argument("--debug_nan", type=bool_flag, default=True)
    p.add_argument("--nan_policy", type=str, default="raise", choices=["raise","warn","silent"])
    return p


# ==================== Inference helpers for test set ====================
from typing import Any, Dict, List, Optional, Tuple, Union

def _looks_like_ppi(obj: Any) -> bool:
    """
    protein_pool_info 추정자.
    - 흔히 list[dict] 또는 dict (pool 단계별 인덱스/ratio/perm 등 메타)
    - 너무 빡세게 검증하지 않고, list-of-dict 이면 True, dict면 일부 키 힌트로 True
    """
    if isinstance(obj, list) and all(isinstance(x, dict) for x in obj):
        return True
    if isinstance(obj, dict):
        hint_keys = {"pool", "ratio", "perm", "cluster", "levels", "assign", "readout", "pool_info"}
        if any(k in obj for k in hint_keys):
            return True
    return False

def _extract_pred_and_ppi(outputs: Any):
    """
    (원본 제공 함수) + 살짝 보강:
    지원 형태:
    - Tensor
    - (y_pred, protein_pool_info)
    - (y_pred, (attn_pl, attn_lp), protein_pool_info)
    - (y_pred, level_feats, protein_pool_info)
    - (y_pred, (attn_pl, attn_lp), level_feats, protein_pool_info)
    - {'y_pred': ..., 'protein_pool_info': ...}
    - 기타 튜플/리스트 내 위치 불문 마지막 요소가 protein_pool_info인 케이스
    """
    y_pred, protein_pool_info = None, None

    # dict 케이스
    if isinstance(outputs, dict):
        if "y_pred" in outputs:
            y_pred = outputs["y_pred"]
        if "protein_pool_info" in outputs:
            protein_pool_info = outputs["protein_pool_info"]
        if protein_pool_info is None:
            for v in outputs.values():
                if _looks_like_ppi(v):
                    protein_pool_info = v
                    break

    # tuple/list 케이스
    if isinstance(outputs, (tuple, list)):
        # 후보군 중 첫 텐서를 y_pred로
        if len(outputs) > 0 and torch.is_tensor(outputs[0]):
            y_pred = outputs[0]
        # 마지막 요소부터 protein_pool_info 탐색(반환 규약상 맨 끝에 붙음)
        for obj in reversed(outputs[1:]):
            if _looks_like_ppi(obj):
                protein_pool_info = obj if isinstance(obj, list) else [obj]
                break

    # 순수 텐서
    if protein_pool_info is None and y_pred is None and torch.is_tensor(outputs):
        y_pred = outputs

    return y_pred, protein_pool_info

def _extract_everything(outputs: Any) -> Tuple[torch.Tensor,
                                               Optional[Tuple[Any, Any]],
                                               Optional[Any],
                                               Optional[Any]]:
    """
    y_pred, (attn_pl, attn_lp), level_feats, protein_pool_info 를 최대한 일반적으로 파싱
    - 모델의 다양한 반환 포맷(딕트/튜플/리스트/텐서)을 지원
    - attn과 level_feats는 없을 수 있으므로 Optional
    """
    y_pred, protein_pool_info = _extract_pred_and_ppi(outputs)
    attn_tuple: Optional[Tuple[Any, Any]] = None
    level_feats: Optional[Any] = None

    # dict 우선 시도
    if isinstance(outputs, dict):
        # attn 후보: ('attn', 'attn_weights') 또는 ('attn_pl','attn_lp')
        if "attn" in outputs:
            attn = outputs["attn"]
            if isinstance(attn, (tuple, list)) and len(attn) == 2:
                attn_tuple = (attn[0], attn[1])
        elif "attn_weights" in outputs:
            attn = outputs["attn_weights"]
            if isinstance(attn, (tuple, list)) and len(attn) == 2:
                attn_tuple = (attn[0], attn[1])
        elif "attn_pl" in outputs or "attn_lp" in outputs:
            attn_tuple = (outputs.get("attn_pl", None), outputs.get("attn_lp", None))

        # level feats 후보
        for k in ("level_feats", "cluster_level_feats", "levels"):
            if k in outputs:
                level_feats = outputs[k]
                break

    # tuple/list 의 경우: 관례상 (y_pred, (attn_pl,attn_lp), level_feats, protein_pool_info) 순서가 흔함
    if attn_tuple is None or level_feats is None:
        if isinstance(outputs, (tuple, list)):
            # y_pred 이후의 성분들만 검사
            tail = list(outputs[1:]) if len(outputs) > 1 else []
            # protein_pool_info는 보통 맨 끝이므로 스캔 시 제외
            if tail and _looks_like_ppi(tail[-1]):
                tail = tail[:-1]

            # (attn_pl, attn_lp) 후보: 길이 2의 튜플/리스트
            for obj in tail:
                if isinstance(obj, (tuple, list)) and len(obj) == 2:
                    a0, a1 = obj[0], obj[1]
                    # 대략적인 힌트: 텐서/리스트/딕트 조합을 허용
                    if (torch.is_tensor(a0) or isinstance(a0, (list, dict))) and \
                       (torch.is_tensor(a1) or isinstance(a1, (list, dict))):
                        attn_tuple = (a0, a1)
                        break

            # level_feats 후보: 남은 것 중 텐서/리스트/딕트 묶음
            if level_feats is None:
                for obj in tail:
                    if obj is attn_tuple:
                        continue
                    if isinstance(obj, (list, dict)) or torch.is_tensor(obj):
                        level_feats = obj
                        break

    return y_pred, attn_tuple, level_feats, protein_pool_info


def _maybe_load_best(model: torch.nn.Module, best_path: Optional[str]) -> None:
    """
    trainer.fit 이 저장한 체크포인트 로드(가능한 키들을 폭넓게 시도)
    """
    if not best_path:
        print("[WARN] best_path 가 비어 있어 현재 메모리의 모델로 추론합니다.")
        return
    ckpt = torch.load(best_path, map_location="cpu")
    # 다양한 포맷 대응
    sd = None
    for k in ("state_dict", "model", "model_state_dict", "net", "module"):
        if isinstance(ckpt, dict) and k in ckpt and isinstance(ckpt[k], dict):
            sd = ckpt[k]
            break
    if sd is None and isinstance(ckpt, dict):
        # 혹시 바로 state_dict 형태일 수도
        sd = ckpt
    if sd is None:
        print(f"[WARN] 체크포인트에서 state_dict 를 찾지 못했습니다: {best_path}. 그대로 진행합니다.")
        return
    # DataParallel 저장본 키 정리
    new_sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"[INFO] Missing keys({len(missing)}): {missing[:5]}{' ...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[INFO] Unexpected keys({len(unexpected)}): {unexpected[:5]}{' ...' if len(unexpected)>5 else ''}")


def run_test_inference(model: torch.nn.Module,
                       test_loader: torch.utils.data.DataLoader,
                       device: torch.device,
                       use_amp: bool = False,
                       use_sepool: bool = False,
                       save_dir: Union[str, Path] = "./ckpt/runs/exp1",
                       run_id: str = "0") -> Dict[str, Any]:

    model.eval()
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    all_preds: List[torch.Tensor] = []
    all_attn: List[Optional[Tuple[Any, Any]]] = []
    all_levels: List[Optional[Any]] = []
    all_ppi: List[Optional[Any]] = []

    jsonl_path = Path(save_dir) / f"inference_test_run_{run_id}.jsonl"
    pt_path    = Path(save_dir) / f"inference_test_run_{run_id}.pt"

    try:
        if jsonl_path.exists():
            jsonl_path.unlink()
    except Exception:
        pass

    # [ADD] 환경 변수로 디버그 온/오프
    infer_debug = os.getenv("INFER_DEBUG", "0") in ("1", "true", "True")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            sequences       = batch.get("protein_sequences")
            graphs          = batch.get("ligand_batch")
            protein_graphs  = batch.get("protein_graphs", None)
            tree_batch      = batch.get("tree_batch") if use_sepool else None

            graphs = move_to_device(graphs, device)
            protein_graphs = move_to_device(protein_graphs, device)
            tree_batch = move_to_device(tree_batch, device)

            if use_amp and device.type == "cuda":
                autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
            else:
                from contextlib import nullcontext
                autocast_ctx = nullcontext()

            with autocast_ctx:
                outputs = model(
                    sequences,
                    graphs,
                    protein_graphs=protein_graphs,
                    tree_batch=tree_batch,
                    return_protein_pool_info=True,
                    return_attn_weights=True,
                    return_cluster_levels=True,
                )

            # [ADD] 첫 배치이거나 디버그 모드라면 반환 구조 덤프
            if infer_debug or batch_idx < 1:
                dump_outputs_shape(outputs, prefix=f"[OUT][batch {batch_idx}]")

            # [CHANGE] 엄격 파서 우선
            y_pred, attn_tuple, level_feats, protein_pool_info = extract_all_strict(outputs)

            if attn_tuple is None:
                # [ADD] 배치별 경고 + 일부 배치는 구조 덤프
                print(f"[WARN] attn_tuple is None at batch {batch_idx}")
                if not (infer_debug or batch_idx < 1):
                    # 이미 위에서 첫 배치는 덤프했음. 추가로 2번째 배치도 한 번 덤프.
                    if batch_idx < 2:
                        dump_outputs_shape(outputs, prefix=f"[OUT][batch {batch_idx}][warn-dump]")

            # 수집
            if torch.is_tensor(y_pred):
                all_preds.append(y_pred.detach().cpu())
            else:
                all_preds.append(torch.empty(0))

            all_attn.append(attn_tuple)
            all_levels.append(level_feats)
            all_ppi.append(protein_pool_info)

            try:
                import json as _json
                if torch.is_tensor(y_pred) and y_pred.numel() > 0:
                    mean_pred = float(y_pred.detach().mean().item())
                    rec = {"batch_idx": batch_idx, "mean_pred": mean_pred,
                           "has_attn": attn_tuple is not None}
                else:
                    rec = {"batch_idx": batch_idx, "mean_pred": None, "has_attn": False}
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(_json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[WARN] jsonl 기록 실패(batch {batch_idx}): {e}")

    y_pred_all = torch.cat([t for t in all_preds if isinstance(t, torch.Tensor) and t.numel() > 0], dim=0) \
                 if any(t.numel() > 0 for t in all_preds if isinstance(t, torch.Tensor)) \
                 else torch.empty(0)

    out = {
        "y_pred": y_pred_all,
        "attn": all_attn,
        "level_feats": all_levels,
        "protein_pool_info": all_ppi,
    }

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    torch.save(out, pt_path)
    print(f"[OK] Test inference saved:\n- PT:   {pt_path}\n- JSONL:{jsonl_path}")
    return out

def _autodetect_checkpoint(ckpt_dir: Path, run_id: str) -> Optional[str]:
    """
    ckpt_dir 아래에서 가장 그럴듯한 .pt 체크포인트 자동 탐색:
    - 파일명에 'best'와 run_id를 모두 포함한 것을 최우선
    - 다음으로 run_id 포함
    - 다음으로 'best' 포함
    - 그 외에는 수정시각 최신
    """
    if not ckpt_dir.exists():
        return None
    cands = list(ckpt_dir.rglob("*.pt"))
    if not cands:
        return None

    def score(p: Path) -> tuple:
        name = p.name.lower()
        s0 = int(("best" in name) and (run_id in name))   # best & run_id
        s1 = int((run_id in name))                        # run_id
        s2 = int(("best" in name))                        # best
        s3 = p.stat().st_mtime                            # recent
        return (s0, s1, s2, s3)

    cands.sort(key=score, reverse=True)
    return str(cands[0])

# ------------------ main ------------------
# ------------------ main ------------------
def main():
    args = build_parser().parse_args()
    set_seed(args.seed)

    # 더 자세한 디버깅을 기본 ON (원하면 주석 처리)
    os.environ.setdefault("INFER_DEBUG", "1")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # HF (옵션)
    if args.hf_token:
        try:
            login(args.hf_token)
        except Exception as e:
            print(f"[WARN] HuggingFace login failed: {e}")
    else:
        print("[INFO] HF token not provided. Skipping login.")

    # ------------------ Dataset ------------------
    data = DTI(name=args.dataset_name)
    data.harmonize_affinities(mode='max_affinity')
    df = data.get_data()
    df = df[df['Y'] > 0].copy()
    # pKd = 9 - log10(Kd[nM])
    df['pKd'] = -df['Y'].apply(lambda x: torch.log10(torch.tensor(x * 1e-9)).item())
    df = df[df['Target'].apply(lambda x: len(x) <= args.max_target_len)].reset_index(drop=True)

    # 동일 분할 방식으로 test 셋 구성
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=args.seed)
    val_df,   test_df = train_test_split(temp_df, test_size=0.5, random_state=args.seed)

    # ------------------ Dataloaders ------------------
    test_dataset = BindingDataset(test_df, protein_weight_mode=args.protein_weight_mode)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_binding
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
        som_mode=args.som_mode,

        # Ligand encoder
        ligand_gnn_type=args.ligand_gnn_type,
        ligand_in_dim=None,
        ligand_hidden_dim=args.ligand_hidden_dim,
        ligand_num_layers=args.ligand_num_layers,

        # Ligand pooling
        ligand_pool_type=args.ligand_pool_type,
        tree_depth=args.tree_depth,
        sepool_hidden_dim=args.sepool_hidden_dim,

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

    # ------------------ Load checkpoint ------------------
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    autodetected = _autodetect_checkpoint(ckpt_dir, args.run_id)
    if autodetected is None:
        print(f"[WARN] No checkpoint found under: {ckpt_dir}. Proceeding with current (unloaded) weights.")
        best_path = None
    else:
        print(f"[INFO] Using checkpoint: {autodetected}")
        best_path = autodetected

    # >>>> 변경점 1: 키 리매핑 로더로 로드
    if best_path is not None:
        load_ckpt_with_remap(model, best_path, verbose=True)  # <-- _maybe_load_best 대신 이 함수 사용
    else:
        print("[WARN] Proceeding without loading checkpoint weights.")

    # >>>> 변경점 2: mincut 가중치 로드 여부 경고
    if getattr(model, "protein_pool_type", "") == "mincut":
        st = model.state_dict().keys()
        has_assign_params = any(k.startswith("protein_assign.") for k in st)
        if not has_assign_params:
            print("[WARN] protein_pool_type='mincut' 이지만 'protein_assign.*' 파라미터가 모델에 없습니다."
                  " (코드-체크포인트 버전차/리팩토링 가능성). mincut assigner는 랜덤 초기화 상태일 수 있습니다.")
        else:
            # 체크포인트에서 assigner가 실제로 로드되었는지 간접 확인
            # (로드 전후 파라미터 norm 비교 같은 정밀검사는 생략)
            print("[INFO] Found 'protein_assign.*' parameters in model. (Loaded or freshly initialized)")

    # >>>> 변경점 3: 어텐션/모듈 진단 출력
    debug_attn_capabilities(model, best_path)

    # ------------------ Inference ------------------
    _ = run_test_inference(
        model,
        test_loader,
        device,
        use_amp=args.amp,
        use_sepool=args.use_sepool,
        save_dir=str(ckpt_dir),
        run_id=args.run_id,
    )

    print("[DONE] Inference completed.")

if __name__ == "__main__":
    main()