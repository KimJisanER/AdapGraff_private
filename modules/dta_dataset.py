from modules.dualgraph.mol import smiles2graphwithface
import torch, random
import numpy as np
import torch_geometric.transforms as T
from rdkit.Chem import AllChem
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch, dense_to_sparse
from torch_geometric.utils import shuffle_node, mask_feature, dropout_node, dropout_edge
from torch_geometric.transforms import RandomNodeSplit
from modules.dualgraph.dataset import DGData
from torch.utils.data import Dataset
from protein_init import *

import os
import copy
import pickle
import itertools
import numpy as np
import networkx as nx
from utils import load_graph, to_cpu_tensor, map_nested_to_cpu, _to_1d_cpu_float
from multiprocessing import Pool
import torch
import os, json, math
from pathlib import Path
import torch
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected
from torch_geometric.utils import coalesce
from copy import deepcopy
import bisect
from itertools import accumulate
# import warnings
# warnings.filterwarnings('ignore')

#########################################################################################################################
##########################################################################################################################
def dgdata_to_nx(graph, atoms=None) -> nx.Graph:
    """
    DGData -> NetworkX. 노드 'tag'는 DGData.x에서 그대로 가져온다.
    SEP 전처리(one_hot_features)가 전체 label_set을 모아서 원-핫으로 바꿔주므로,
    여기서는 정수 라벨만 일관되게 넣어주면 됨.
    """
    G = nx.Graph()

    # 노드/엣지 추가
    num_nodes = int(getattr(graph, "n_nodes", getattr(graph, "num_nodes", graph.x.size(0))))
    G.add_nodes_from(range(num_nodes))

    ei = graph.edge_index
    if isinstance(ei, torch.Tensor):
        ei = ei.cpu().numpy()
    for u, v in zip(ei[0], ei[1]):
        if u != v:
            G.add_edge(int(u), int(v))

    # ---- 여기서 '원래 방식'으로 tag 주입 ----
    x = graph.x  # DGData.x
    # 케이스 1) x가 [N] 정수 라벨 벡터
    if x.dim() == 1 and x.dtype in (torch.long, torch.int64, torch.int32):
        tags = x.tolist()
    # 케이스 2) x가 [N, F] 이고 one-hot 라벨이 첫 채널이거나 one-hot 전체인 경우
    elif x.dim() == 2:
        # (a) one-hot이면 argmax로 라벨 획득
        if x.dtype in (torch.long, torch.int64, torch.int32) and (x.sum(dim=1) == 1).all():
            tags = x.argmax(dim=1).tolist()
        elif x.dtype in (torch.float16, torch.float32, torch.float64) and torch.isclose(x.sum(dim=1), torch.ones(x.size(0))).all():
            tags = x.argmax(dim=1).tolist()
        else:
            # (b) 첫 컬럼이 이미 정수 라벨인 경우
            if x[:, 0].dtype in (torch.long, torch.int64, torch.int32):
                tags = x[:, 0].tolist()
            else:
                raise ValueError("DGData.x에서 정수 라벨(tag)을 추출할 수 없습니다. 라벨 채널을 지정해 주세요.")
    else:
        raise ValueError("DGData.x 형식을 지원하지 않습니다.")

    for i, t in enumerate(tags):
        G.nodes[i]["tag"] = int(t)  # SEP 전처리에서 one-hot로 변환됨

    return G

#######################################################################################################################

def mol2graph(mol):
    data = DGData()
    graph = smiles2graphwithface(mol)

    data.__num_nodes__ = int(graph["num_nodes"])
    data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
    data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
    data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)

    data.ring_mask = torch.from_numpy(graph["ring_mask"]).to(torch.bool)
    data.ring_index = torch.from_numpy(graph["ring_index"]).to(torch.int64)
    data.nf_node = torch.from_numpy(graph["nf_node"]).to(torch.int64)
    data.nf_ring = torch.from_numpy(graph["nf_ring"]).to(torch.int64)
    data.num_rings = int(graph["num_rings"])
    data.n_edges = int(graph["n_edges"])
    data.n_nodes = int(graph["n_nodes"])
    data.n_nfs = int(graph["n_nfs"])

    return data

class BindingDataset(Dataset):
    def __init__(
        self,
        df,
        tree_depth: int = 3,
        k: int = 2,
        use_sep: bool = True,
        protein_contact_threshold: float = 8.0,
        protein_batch_size: int = 2,
        preload_protein: bool = True,
        protein_weight_mode: str = 'inverse',
        esm_structure: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.tree_depth = tree_depth
        self.k = k
        self.use_sep = use_sep

        self._tree_cache = {}
        self._ligand_cache = {}  # SMILES -> DGData 캐시
        self._protein_cache = {} # seq -> protein entry 캐시 (구조물은 미포함)

        self.protein_contact_threshold = protein_contact_threshold
        self.protein_batch_size = protein_batch_size
        self.protein_weight_mode = protein_weight_mode
        self.esm_structure = esm_structure  # ✅ 저장

        # ✅ 프리로드: 비용 때문에 구조예측은 생략(esm_structure=True여도)
        if preload_protein:
            unique_seqs = self.df["Target"].astype(str).unique().tolist()
            if len(unique_seqs) > 0:
                raw_cache = protein_init(
                    unique_seqs,
                    contact_threshold=self.protein_contact_threshold,
                    batch_size=self.protein_batch_size,
                    weight_mode=self.protein_weight_mode,
                    # esm_structure 생략 (프리로드에서 구조까지 만들지 않음)
                )
                if not isinstance(raw_cache, dict):
                    raise RuntimeError("protein_init did not return a dict as expected during preload.")
                self._protein_cache = map_nested_to_cpu(raw_cache)

    def __len__(self):
        return len(self.df)

    def _get_protein_entry(self, seq: str, esm_structure: bool = False):
        """
        반환:
          - esm_structure=False: entry(dict-like)
          - esm_structure=True : (entry, structure_output)
        """
        # 캐시에 있으면 entry만 즉시 반환 (구조물은 캐시에 안 넣음)
        if seq in self._protein_cache and not esm_structure:
            return self._protein_cache[seq]

        # 단일 시퀀스 호출
        result = protein_init(
            [seq],
            contact_threshold=self.protein_contact_threshold,
            batch_size=1,
            weight_mode=self.protein_weight_mode,
            esm_structure=esm_structure,
        )

        # 반환 형태 정규화
        structure_output = None
        if isinstance(result, tuple) and len(result) == 2:
            single, structure_output = result
        else:
            single = result

        if not isinstance(single, dict) or seq not in single:
            raise ValueError(f"protein_init failed or returned empty for seq[:10]={seq[:10]}...")

        single_cpu = map_nested_to_cpu(single)
        # 캐시에 entry 저장
        self._protein_cache.update(single_cpu)

        if esm_structure:
            return self._protein_cache[seq], structure_output
        else:
            return self._protein_cache[seq]

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        smiles = row["Drug"]
        seq = row["Target"]

        # --- Ligand graph (with cache) ---
        if smiles in self._ligand_cache:
            graph = self._ligand_cache[smiles]
        else:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES at index {idx}: {smiles}")
            graph = mol2graph(mol)  # 내부에서 CPU 텐서 생성되도록 구현되어 있어야 함
            self._ligand_cache[smiles] = graph

        # 부가 정보
        graph.smile = smiles
        mol = Chem.MolFromSmiles(smiles)  # 필요시 캐시해도 됨
        graph.atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        graph.bonds_idx = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]

        item = {
            "sequence": seq,
            "smiles": smiles,
            "label": to_cpu_tensor(row["pKd"], torch.float32),
            "graph": graph,
        }

        # --- Protein entry ---
        if self.esm_structure:
            prot_entry, structure_output = self._get_protein_entry(seq, esm_structure=True)
        else:
            prot_entry = self._get_protein_entry(seq, esm_structure=False)
            structure_output = None

        # edge_index 모양 통일 (2, E)
        edge_index = to_cpu_tensor(prot_entry["edge_index"], torch.long)
        if edge_index.dim() == 2 and edge_index.size(0) != 2 and edge_index.size(1) == 2:
            edge_index = edge_index.t().contiguous()
        E = edge_index.size(1)

        # weight / features / nodes
        edge_weight = _to_1d_cpu_float(prot_entry.get("edge_weight"))
        if E == 0:
            edge_weight = torch.empty(0, dtype=torch.float32)
        x_tok = to_cpu_tensor(prot_entry["token_representation"], torch.float32)  # [L, D]

        # 노드 수/좌표
        num_nodes = int(prot_entry.get("num_nodes", x_tok.size(0)))
        num_nodes = min(num_nodes, x_tok.size(0))
        pos = to_cpu_tensor(
            prot_entry.get("pos", prot_entry.get("num_pos", None)), torch.float32
        )

        # 길이 보정
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

        # coalesce
        ei, ew = coalesce(
            edge_index.contiguous(),
            edge_weight.contiguous(),
            num_nodes,
            num_nodes,
        )

        protein_graph = Data(
            x=x_tok,
            edge_index=ei,
            edge_weight=ew.to(torch.float32).contiguous(),
            num_nodes=num_nodes,
            pos=pos,
        )
        protein_graph.seq_feat = map_nested_to_cpu(prot_entry.get("seq_feat"))
        protein_graph.secondary_structure = map_nested_to_cpu(prot_entry.get("secondary_structure"))
        protein_graph.sasa = map_nested_to_cpu(prot_entry.get("sasa"))
        protein_graph.function_logits = map_nested_to_cpu(prot_entry.get("function_logits"))

        item["protein_graph"] = protein_graph
        item["protein_seq"] = seq

        if self.esm_structure and structure_output is not None:
            item["esm_structure"] = structure_output

        return item

class PrecomputedBindingDataset(Dataset):
    """
    Expect directory layout:
      split_dir/
        meta.json (optional; can contain shard_lengths)
        *.pt
    """
    def __init__(self, split_dir: str):
        self.split_dir = Path(split_dir)
        self.shard_dir = self.split_dir

        meta_path = self.split_dir / "meta.json"
        self._shard_files = sorted(list(self.shard_dir.glob("*.pt")))
        if not self._shard_files:
            raise RuntimeError(f"No shard files under {self.shard_dir}")

        self._cache_shard_id = None
        self._cache_data = None

        # --- 1) 길이 정보 가져오기: meta 우선, 없으면 한번 측정 ---
        self._shard_lengths = None
        self.num_shards = len(self._shard_files)
        self.shard_size = None  # keep for backward compat (first shard size)

        if meta_path.exists():
            with open(meta_path, "r") as f:
                m = json.load(f)
            # optional 필드 지원
            shard_lengths = m.get("shard_lengths", None)
            if isinstance(shard_lengths, list) and len(shard_lengths) == self.num_shards:
                self._shard_lengths = [int(x) for x in shard_lengths]

            # (참고) 기존 메타 필드도 읽어두되, 가변 샤드 대응은 _shard_lengths로만 처리
            self.num_samples = int(m.get("num_samples", 0)) if "num_samples" in m else None
            if "shard_size" in m:
                self.shard_size = int(m["shard_size"])

        if self._shard_lengths is None:
            # 메타에 없으면: 한 번만 로드해서 길이 측정 (weights_only=False)
            lengths = []
            for shp in self._shard_files:
                dl = torch.load(shp, map_location="cpu", weights_only=False)
                lengths.append(len(dl))
            self._shard_lengths = lengths

        # --- 2) 누적합 / 총길이 ---
        self._cumsums = list(accumulate(self._shard_lengths))
        self._total_len = self._cumsums[-1]
        if self.shard_size is None:
            self.shard_size = self._shard_lengths[0]  # reference only

    def __len__(self):
        return self._total_len

    def _locate(self, idx: int):
        if idx < 0 or idx >= self._total_len:
            raise IndexError(f"Index {idx} out of range [0, {self._total_len})")
        sid = bisect.bisect_right(self._cumsums, idx)
        prev_end = 0 if sid == 0 else self._cumsums[sid - 1]
        offset = idx - prev_end
        return sid, offset

    def _load_shard(self, shard_id: int):
        if self._cache_shard_id == shard_id and self._cache_data is not None:
            return self._cache_data
        path = self._shard_files[shard_id]
        try:
            data_list = torch.load(path, map_location="cpu")  # PyTorch 2.6 default weights_only=True
        except Exception as e:
            raise RuntimeError(
                f"Safe unpickling failed for shard {path}. "
                f"Add the missing class mentioned in the error to safe_globals."
            ) from e
        self._cache_shard_id = shard_id
        self._cache_data = data_list
        return data_list

    def __getitem__(self, idx: int):
        shard_id, offset = self._locate(idx)
        shard = self._load_shard(shard_id)
        sample = shard[offset]
        if isinstance(sample.get("label", None), torch.Tensor):
            sample["label"] = sample["label"].detach().to("cpu").contiguous()
        return sample


