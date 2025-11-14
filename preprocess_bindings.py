
# -*- coding: utf-8 -*-
"""
Preprocess BindingDB_Kd dataset into CPU-only graph/sequence/label shards (.pt)
"""

import os
import json
import math
from pathlib import Path
import os
import json
import math
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
from torch_geometric.utils import coalesce
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from rdkit import Chem
from tqdm import tqdm
import torch
from rdkit import Chem
from torch_geometric.utils import coalesce
from torch_geometric.data import Data
from tdc.multi_pred import DTI
from utils import to_cpu_tensor, map_nested_to_cpu, _to_1d_cpu_float

# === Import your own modules ===
from modules.dta_dataset import *
from protein_init import *


# ====== 단백질 그래프 정규화 ======
def normalize_protein_graph(prot: dict) -> Data:
    """prot dict -> PyG Data (CPU), edge_index/edge_weight 정합성 보장 + coalesce"""
    edge_index = to_cpu_tensor(prot["edge_index"], torch.long)
    if edge_index.dim() == 2 and edge_index.size(0) != 2 and edge_index.size(1) == 2:
        edge_index = edge_index.t().contiguous()
    E = edge_index.size(1)

    edge_weight = _to_1d_cpu_float(prot.get("edge_weight"))
    if E == 0:
        edge_weight = torch.empty(0, dtype=torch.float32)

    if edge_weight is None or edge_weight.numel() == 0:
        edge_weight = torch.ones(E, dtype=torch.float32)
    else:
        ew = edge_weight.view(-1)
        if ew.numel() != E:
            if ew.numel() * 2 == E:
                ew = ew.repeat(2)
            elif ew.numel() < E and (E % ew.numel() == 0):
                ew = ew.repeat(E // ew.numel())
            elif ew.numel() > E:
                ew = ew[:E]
            else:
                ew = torch.ones(E, dtype=torch.float32)
        edge_weight = ew

    x_tok = to_cpu_tensor(prot["token_representation"], torch.float32)  # [L, D]
    num_nodes = int(prot.get("num_nodes", x_tok.size(0)))
    num_nodes = min(num_nodes, x_tok.size(0))
    pos = to_cpu_tensor(prot.get("num_pos", None), torch.float32)

    ei, ew = coalesce(edge_index.contiguous(), edge_weight.contiguous(),
                      num_nodes, num_nodes)

    g = Data(
        x=x_tok,
        edge_index=ei,
        edge_weight=ew.to(torch.float32).contiguous(),
        num_nodes=num_nodes,
        pos=pos,
    )
    g.seq_feat = map_nested_to_cpu(prot.get("seq_feat"))
    g.secondary_structure = map_nested_to_cpu(prot.get("secondary_structure"))
    g.sasa = map_nested_to_cpu(prot.get("sasa"))
    g.function_logits = map_nested_to_cpu(prot.get("function_logits"))
    return g


# ====== 샤드 저장 ======
def save_split(df, split_name, out_root: Path, shard_size, k, prot_cache):
    """df의 각 row를 PyG 샘플로 전처리하여 shards/에 저장"""
    split_dir = out_root / split_name
    shard_dir = split_dir
    shard_dir.mkdir(parents=True, exist_ok=True)

    items = []
    shard_idx = 0
    total = len(df)

    for i, row in tqdm(df.iterrows(), total=total, desc=f"Preproc[{split_name}]"):
        smiles = row["Drug"]
        seq = row["Target"]
        label = float(row["pY"])

        # Ligand graph (CPU)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        lig = mol2graph(mol)  # 내부에서 CPU 텐서 리턴되도록 구현되어 있어야 함
        lig.smile = smiles
        # 부가정보(원하면)
        lig.atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        lig.bonds_idx = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]

        # Protein graph (CPU + 정규화)
        prot = prot_cache[seq]
        pg = normalize_protein_graph(prot)

        sample = {
            "sequence": seq,
            "smiles": smiles,
            "label": torch.tensor(label, dtype=torch.float32),
            "graph": lig,
            "protein_graph": pg,
        }
        items.append(sample)

        if len(items) >= shard_size:
            shard_path = shard_dir / f"{split_name}_shard_{shard_idx:05d}.pt"
            torch.save(items, shard_path)
            shard_idx += 1
            items = []

    if len(items) > 0:
        shard_path = shard_dir / f"{split_name}_shard_{shard_idx:05d}.pt"
        torch.save(items, shard_path)
        shard_idx += 1

    # meta.json for the split
    with open(split_dir / "meta.json", "w") as f:
        json.dump(
            {
                "split": split_name,
                "num_samples": int(total),
                "num_shards": int(shard_idx),
                "shard_size": int(shard_size),
                "k": int(k),
            },
            f,
            indent=2,
        )
    print(f"[OK] {split_name}: {total} samples -> {shard_idx} shard(s)")


# ====== 메인 ======
def main():
    parser = argparse.ArgumentParser("Preprocess BindingDB (Kd/Ki/IC50) into CPU shards with train/val/test split")
    parser.add_argument("--data_type", type=str, default="Kd", choices=["Kd", "Ki", "IC50"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_base", type=str, default="./precomputed")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--protein_contact_threshold", type=float, default=8.0)
    parser.add_argument("--protein_batch_size", type=int, default=64)
    parser.add_argument("--protein_weight_mode", type=str, default="inverse",
                        choices=["inverse", "linear", "gaussian", "binary", "raw"])
    parser.add_argument("--max_target_len", type=int, default=1000)
    parser.add_argument("--shard_size", type=int, default=2000)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # === 데이터셋 이름 선택 ===
    name_map = {
        "Kd": "BindingDB_Kd",
        "Ki": "BindingDB_Ki",
        "IC50": "BindingDB_IC50",
    }
    dataset_name = name_map[args.data_type]

    # === 출력 디렉토리: 유형+시드 노출 ===
    out_root = Path(args.out_base) / f"{args.data_type}_s{args.seed}"
    out_root.mkdir(parents=True, exist_ok=True)

    # === 원본 로드 & 가공 ===
    data = DTI(name=dataset_name)
    data.harmonize_affinities(mode="max_affinity")
    df = data.get_data()
    # BindingDB 값(Y)은 보통 nM로 제공 → pY = -log10(Y * 1e-9)
    df = df[df["Y"] > 0].copy()
    df["pY"] = -df["Y"].apply(lambda x: torch.log10(torch.tensor(x * 1e-9)).item())
    df = df[df["Target"].apply(lambda x: len(x) <= args.max_target_len)].reset_index(drop=True)
    print(f"[INFO] Loaded {dataset_name}: {len(df)} usable samples")

    # === split (80/10/10) with seed ===
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=args.seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=args.seed)
    print(f"[INFO] Split -> train {len(train_df)} / val {len(val_df)} / test {len(test_df)}")

    # === 단백질 캐시: 세 split 전체에서 등장하는 시퀀스들의 합집합 ===
    all_seqs = pd_concat_series([train_df["Target"], val_df["Target"], test_df["Target"]]).astype(str).unique().tolist()
    prot_cache_raw = protein_init(
        all_seqs,
        contact_threshold=args.protein_contact_threshold,
        batch_size=args.protein_batch_size,
        weight_mode=args.protein_weight_mode,
    )
    if not isinstance(prot_cache_raw, dict):
        raise RuntimeError("protein_init did not return a dict")
    prot_cache = map_nested_to_cpu(prot_cache_raw)
    print(f"[INFO] Protein cache ready: {len(prot_cache)} unique sequences")

    # === 샤드 저장 (train/val/test) ===
    save_split(train_df, "train", out_root, args.shard_size, args.k, prot_cache)
    save_split(val_df,   "val",   out_root, args.shard_size, args.k, prot_cache)
    save_split(test_df,  "test",  out_root, args.shard_size, args.k, prot_cache)

    # === top-level meta ===
    with open(out_root / "meta.json", "w") as f:
        json.dump(
            {
                "dataset_name": dataset_name,
                "data_type": args.data_type,
                "seed": args.seed,
                "k": args.k,
                "protein_contact_threshold": args.protein_contact_threshold,
                "protein_weight_mode": args.protein_weight_mode,
                "max_target_len": args.max_target_len,
                "shard_size": args.shard_size,
            },
            f,
            indent=2,
        )
    print(f"[DONE] All splits saved under: {out_root}")


# ====== helpers ======
def pd_concat_series(series_list):
    # pandas를 직접 임포트하지 않고 간단히 처리
    import pandas as pd
    return pd.concat(series_list, axis=0, ignore_index=True)


if __name__ == "__main__":
    main()