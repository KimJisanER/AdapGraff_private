import torch
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss
from torch_geometric.data import Batch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, jaccard_score, precision_score, recall_score, roc_auc_score, average_precision_score
from torch_geometric.utils import unbatch
import warnings
warnings.filterwarnings('ignore', '')
warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')
import copy
import math
import heapq
import numba as nb
import numpy as np
import networkx as nx
from torch_geometric.data import Batch as PygBatch
import torch
import importlib, inspect

def load_graph(dname):
    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('datasets/%s/%s.txt' % (dname, dname.replace('-', '')), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if l not in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                g.add_node(j, tag=row[0])
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
            else:
                node_features = None

            assert len(g) == n
            g_list.append({'G': g, 'label': l})
    print("# data: %d\tlabel:%s" % (len(g_list), len(label_dict)))
    return g_list


def collate_binding(batch, use_sep: bool = True):
    # 기본 요소 수집
    seqs   = [b["sequence"] for b in batch]              # list[str]
    ligs   = [b["graph"]    for b in batch]              # list[PyG Data] (ligand)
    prots  = [b["protein_graph"] for b in batch]         # list[PyG Data] (protein)
    labels = torch.stack([b["label"] for b in batch], dim=0).view(-1)

    # PyG Batch로 묶기
    ligand_batch  = PygBatch.from_data_list(ligs)
    protein_batch = PygBatch.from_data_list(prots)

    out = {
        "protein_sequences": seqs,        # 모델 forward에서 protein_sequences로 사용
        "ligand_batch": ligand_batch,     # ligand 그래프 배치
        "protein_graphs": protein_batch,  # ✅ protein_graphs(복수)로 통일
        "y": labels,
    }

    # 선택: SEPooling 트리
    if use_sep and ("tree" in batch[0]):
        out["tree_batch"] = [b["tree"] for b in batch]

    return out

# ------------------ utilities ------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
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

def _try_import(path: str):
    try:
        m, name = path.rsplit(".", 1)
        mod = importlib.import_module(m)
        return getattr(mod, name, None)
    except Exception:
        return None

def register_safe_globals():
    candidates = [
        # Project custom
        "modules.dualgraph.dataset.DGData",

        # PyG common
        "torch_geometric.data.data.Data",
        "torch_geometric.data.batch.Batch",
        "torch_geometric.data.hetero_data.HeteroData",

        # PyG data helpers seen so far
        "torch_geometric.data.data.DataTensorAttr",
        "torch_geometric.data.data.DataEdgeAttr",

        # torch-sparse (가끔 섞임)
        "torch_sparse.tensor.SparseTensor",
    ]

    found = []
    for p in candidates:
        obj = _try_import(p)
        if obj is not None:
            found.append(obj)

    # 🔥 Storage 계열은 버전에 따라 많으니 모듈에서 자동 수집
    try:
        storage_mod = importlib.import_module("torch_geometric.data.storage")
        for name, obj in inspect.getmembers(storage_mod):
            # BaseStorage, NodeStorage, EdgeStorage, GlobalStorage, GraphStore 등 버전에 따라 다양
            if inspect.isclass(obj) and name.endswith("Storage"):
                found.append(obj)
    except Exception:
        pass

    if found:
        # 중복 제거
        uniq = []
        seen = set()
        for cls in found:
            if cls is None or cls in seen:
                continue
            uniq.append(cls)
            seen.add(cls)
        torch.serialization.add_safe_globals(uniq)
        print(f"[SAFE_GLOBALS] registered: {[c.__name__ for c in uniq]}")

# === utils ===
def to_cpu_tensor(x, dtype=None):
    if isinstance(x, torch.Tensor):
        t = x.detach().to('cpu')
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
        return t.contiguous()
    if x is None:
        return None
    t = torch.as_tensor(x)
    if dtype is not None and t.dtype != dtype:
        t = t.to(dtype)
    return t.contiguous()

def map_nested_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().to('cpu').contiguous()
    if isinstance(obj, dict):
        return {k: map_nested_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [map_nested_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(map_nested_to_cpu(v) for v in obj)
    return obj

def _to_1d_cpu_float(x):
    if x is None:
        return None
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    return x.detach().to('cpu').contiguous().view(-1).to(torch.float32)

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
