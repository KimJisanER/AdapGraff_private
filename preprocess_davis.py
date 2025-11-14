# -*- coding: utf-8 -*-
"""
Preprocess 5-Fold DAVIS pkl dataset into CPU-only graph/sequence/label shards (.pt)
"""

import os
import json
import math
import argparse
from pathlib import Path
from tqdm import tqdm
import pickle
import pandas as pd

import torch
from torch_geometric.utils import coalesce
from torch_geometric.data import Data
from rdkit import Chem
from utils import to_cpu_tensor, map_nested_to_cpu, _to_1d_cpu_float

# === Import your own modules ===
# (이 모듈들이 현재 스크립트와 같은 위치 또는 PYTHONPATH에 있어야 합니다)

from modules.dta_dataset import *
from protein_init import *

# ====== 단백질 그래프 정규화 (Input 2와 동일) ======
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
    # (부가 정보는 원본 데이터에 따라 없을 수 있으므로 .get() 사용)
    g.seq_feat = map_nested_to_cpu(prot.get("seq_feat"))
    g.secondary_structure = map_nested_to_cpu(prot.get("secondary_structure"))
    g.sasa = map_nested_to_cpu(prot.get("sasa"))
    g.function_logits = map_nested_to_cpu(prot.get("function_logits"))
    return g


# ====== [신규] PKL -> DataFrame 변환 헬퍼 ======
def convert_pkl_to_df(dataset_pkl, drug_map, protein_map):
    """
    (drug_id, protein_id, label) 튜플 리스트를
    (SMILES, Sequence, pY) DataFrame으로 변환합니다.
    """
    drugs_smiles = []
    proteins_seqs = []
    labels_py = []

    # Input 1의 load_dataset 로직 참조:
    # data_i[0] = drug_id, data_i[2] = protein_id, data_i[-1] = label
    for data in dataset_pkl:
        try:
            drug_id = str(data[0])
            protein_id = str(data[2])
            label = float(data[-1])

            drugs_smiles.append(drug_map[drug_id])
            proteins_seqs.append(protein_map[protein_id])
            labels_py.append(label)
        except KeyError as e:
            # print(f"Warning: ID {e} not found in mapping dicts. Skipping.")
            pass
        except Exception as e:
            # print(f"Error processing data {data}: {e}")
            pass

    return pd.DataFrame({
        "Drug": drugs_smiles,
        "Target": proteins_seqs,
        "pY": labels_py
    })


# ====== 샤드 저장 (Input 2와 동일) ======
def save_split(df, split_name, out_root: Path, shard_size, k, prot_cache):
    """df의 각 row를 PyG 샘플로 전처리하여 shards/에 저장"""
    split_dir = out_root / split_name
    shard_dir = split_dir
    shard_dir.mkdir(parents=True, exist_ok=True)

    items = []
    shard_idx = 0
    total = len(df)
    processed_count = 0

    for i, row in tqdm(df.iterrows(), total=total, desc=f"Preproc[{split_name}]"):
        smiles = row["Drug"]
        seq = row["Target"]
        label = float(row["pY"])

        # Ligand graph (CPU)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # print(f"Warning: Invalid SMILES {smiles}. Skipping.")
            continue

        try:
            lig = mol2graph(mol)  # 내부에서 CPU 텐서 리턴되도록 구현되어 있어야 함
            lig.smile = smiles
            # 부가정보(원하면)
            lig.atoms = [a.GetSymbol() for a in mol.GetAtoms()]
            lig.bonds_idx = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]

            # Protein graph (CPU + 정규화)
            if seq not in prot_cache:
                # print(f"Warning: Sequence for {smiles} not in prot_cache. Skipping.")
                continue
            prot = prot_cache[seq]
            pg = normalize_protein_graph(prot)

        except Exception as e:
            print(e)
            continue

        sample = {
            "sequence": seq,
            "smiles": smiles,
            "label": torch.tensor(label, dtype=torch.float32),
            "graph": lig,
            "protein_graph": pg,
        }
        items.append(sample)
        processed_count += 1

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
                "num_samples": int(processed_count),  # 실제 처리된 샘플 수
                "num_shards": int(shard_idx),
                "shard_size": int(shard_size),
                "k": int(k),
            },
            f,
            indent=2,
        )
    print(f"[OK] {split_name}: {processed_count} samples -> {shard_idx} shard(s)")


# ====== 메인 (수정됨) ======
def main():
    parser = argparse.ArgumentParser("Preprocess 5-Fold DAVIS PKL dataset into CPU shards")
    # [수정] 데이터셋 경로 인자
    parser.add_argument("--data_dir", type=str, default="./KIBA_datasets/DAVIS",
                        help="Path to the DAVIS .pkl files directory")
    parser.add_argument("--dataset_name", type=str, default="DAVIS", help="Dataset name (e.g., DAVIS)")

    # (Input 2의 인자들 유지)
    parser.add_argument("--out_base", type=str, default="./precomputed")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--protein_contact_threshold", type=float, default=8.0)
    parser.add_argument("--protein_batch_size", type=int, default=16)
    parser.add_argument("--protein_weight_mode", type=str, default="inverse",
                        choices=["inverse", "linear", "gaussian", "binary", "raw"])
    parser.add_argument("--max_target_len", type=int, default=2600)  #####2600#4200##############################
    parser.add_argument("--shard_size", type=int, default=2000)
    args = parser.parse_args()

    data_root = Path(args.data_dir)

    # === [수정] 1. 원본 ID <-> SMILES/Sequence 매핑 로드 ===
    print(f"Loading ID-to-Data maps from {data_root}...")
    try:
        # (가정: drug_ids.pkl는 {id: smiles} 딕셔너리)
        with open(data_root / "drug_ids.pkl", 'rb') as f:
            drug_map = pickle.load(f)
            # (가정: protein_ids.pkl는 {id: sequence} 딕셔너리)
        with open(data_root / "protein_ids.pkl", 'rb') as f:
            protein_map = pickle.load(f)
    except FileNotFoundError as e:
        print(f"[Error] Mapping files (drug_ids.pkl, protein_ids.pkl) not found in {data_root}.")
        print("These files are required to map IDs to SMILES and Sequences.")
        print(e)
        exit()

    # === [수정] 2. 5-Fold 루프 실행 ===
    for fold in range(1, 6):
        print(f"\n--- Processing Fold {fold}/5 ---")

        # === 출력 디렉토리: Fold별로 생성 ===
        out_root_fold = Path(args.out_base) / f"{args.dataset_name}_fold_{fold}"
        out_root_fold.mkdir(parents=True, exist_ok=True)

        # === 3. Fold 데이터 로드 (.pkl) ===
        print(f"Loading train/valid/test pkl files for Fold {fold}...")
        try:
            with open(data_root / f"train_{fold}.pkl", 'rb') as f:
                train_set = pickle.load(f)
            with open(data_root / f"valid_{fold}.pkl", 'rb') as f:
                valid_set = pickle.load(f)
            with open(data_root / f"test_{fold}.pkl", 'rb') as f:
                test_set = pickle.load(f)
        except FileNotFoundError as e:
            print(f"[Error] Fold {fold} .pkl files not found. Skipping fold.")
            print(e)
            continue

        # === 4. PKL -> DataFrame 변환 ===
        print("Converting pkl data to DataFrame (mapping IDs)...")
        train_df = convert_pkl_to_df(train_set, drug_map, protein_map)
        val_df = convert_pkl_to_df(valid_set, drug_map, protein_map)
        test_df = convert_pkl_to_df(test_set, drug_map, protein_map)

        # === 5. (Optional) 시퀀스 길이로 필터링 ===
        train_df = train_df[train_df["Target"].apply(lambda x: len(x) <= args.max_target_len)]
        val_df = val_df[val_df["Target"].apply(lambda x: len(x) <= args.max_target_len)]
        test_df = test_df[test_df["Target"].apply(lambda x: len(x) <= args.max_target_len)]

        print(f"[INFO] Fold {fold} Split -> train {len(train_df)} / val {len(val_df)} / test {len(test_df)}")
        if len(train_df) == 0:
            print(f"[Error] No training data found for fold {fold} after filtering/mapping. Skipping.")
            continue

        # === 6. 단백질 캐시: (T/V/T) 합집합 ===
        all_seqs = pd.concat([train_df["Target"], val_df["Target"], test_df["Target"]]).astype(str).unique().tolist()

        print(f"Building protein cache for {len(all_seqs)} unique sequences...")
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

        # === 7. 샤드 저장 (train/val/test) ===
        save_split(train_df, "train", out_root_fold, args.shard_size, args.k, prot_cache)
        save_split(val_df, "val", out_root_fold, args.shard_size, args.k, prot_cache)
        save_split(test_df, "test", out_root_fold, args.shard_size, args.k, prot_cache)

        # === 8. top-level meta for fold ===
        with open(out_root_fold / "meta.json", "w") as f:
            json.dump(
                {
                    "dataset_name": args.dataset_name,
                    "fold": fold,
                    "k": args.k,
                    "protein_contact_threshold": args.protein_contact_threshold,
                    "protein_weight_mode": args.protein_weight_mode,
                    "max_target_len": args.max_target_len,
                    "shard_size": args.shard_size,
                },
                f,
                indent=2,
            )
        print(f"[DONE] Fold {fold} splits saved under: {out_root_fold}")

    print(f"\n[ALL DONE] Preprocessing for {args.dataset_name} completed.")


# ====== helpers ======
def pd_concat_series(series_list):
    # pandas를 직접 임포트하지 않고 간단히 처리
    return pd.concat(series_list, axis=0, ignore_index=True)


if __name__ == "__main__":
    main()