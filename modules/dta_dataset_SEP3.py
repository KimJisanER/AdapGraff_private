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
from utils import load_graph
from utils import PartitionTree
from multiprocessing import Pool
import torch
import os, json, math
from pathlib import Path
import torch
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected
from torch_geometric.utils import coalesce
from copy import deepcopy
# import warnings
# warnings.filterwarnings('ignore')

#########################################################################################################################

def trans_to_adj(graph):
    graph.remove_edges_from(nx.selfloop_edges(graph))
    nodes = range(len(graph.nodes))
    return nx.to_numpy_array(graph, nodelist=nodes)


def trans_to_tree(adj, k=2):
    undirected_adj = np.array(adj)
    y = PartitionTree(adj_matrix=undirected_adj)
    x = y.build_coding_tree(k)
    return y.tree_node


def update_depth(tree):
    # set leaf depth
    wait_update = [k for k, v in tree.items() if v.children is None]
    while wait_update:
        for nid in wait_update:
            node = tree[nid]
            if node.children is None:
                node.child_h = 0
            else:
                node.child_h = tree[list(node.children)[0]].child_h + 1
        wait_update = set([tree[nid].parent for nid in wait_update if tree[nid].parent])


def update_node(tree):
    update_depth(tree)
    d_id = [(v.child_h, v.ID) for k, v in tree.items()]
    d_id.sort()
    new_tree = {}
    for k, v in tree.items():
        n = copy.deepcopy(v)
        n.ID = d_id.index((n.child_h, n.ID))
        if n.parent is not None:
            n.parent = d_id.index((n.child_h+1, n.parent))
        if n.children is not None:
            n.children = [d_id.index((n.child_h-1, c)) for c in n.children]
        n = n.__dict__
        n['depth'] = n['child_h']
        new_tree[n['ID']] = n
    return new_tree


def pool_trans(input_):
    g, tree_depth = input_
    adj_mat = trans_to_adj(g['G'])
    tree = trans_to_tree(adj_mat, tree_depth)
    g['tree'] = update_node(tree)
    return g


def pool_trans_discon(input_):
    g, tree_depth = input_
    if nx.is_connected(g['G']):
        return pool_trans((g, tree_depth))
    trees = []
    for gi, sub_nodes in enumerate(nx.connected_components(g['G'])):
        if len(sub_nodes) == 1:
            node = list(sub_nodes)[0]
            js = [{'ID': node, 'parent': '%s_%s_0' % (gi, 1), 'depth': 0, 'children': None}]
            for d in range(1, tree_depth+1):
                js.append({'ID': '%s_%s_0' % (gi, d),
                           'parent': '%s_%s_0' % (gi, d+1) if d<tree_depth else None,
                           'depth': d,
                           'children': [js[-1]['ID']]
                          })
        else:
            sg = g['G'].subgraph(sub_nodes)
            nodes = list(sg.nodes)
            nodes.sort()
            nmap = {n: nodes.index(n) for n in nodes}
            sg = nx.relabel_nodes(sg, nmap)
            adj_mat = trans_to_adj(sg)
            tree = trans_to_tree(adj_mat, tree_depth)
            tree = update_node(tree)
            js = list(tree.values())
            rmap = {nodes.index(n): n for n in nodes}
            for j in js:
                if j['depth'] > 0:
                    rmap[j['ID']] = '%s_%s_%s' % (gi, j['depth'], j['ID'])
            for j in js:
                j['ID'] = rmap[j['ID']]
                j['parent'] = rmap[j['parent']] if j['depth']<tree_depth else None
                j['children'] = [rmap[c] for c in j['children']] if j['children'] else None
        trees.append(js)
    id_map = {}
    for d in range(0, tree_depth+1):
        for js in trees:
            for j in js:
                if j['depth'] == d:
                    id_map[j['ID']] = len(id_map) if d>0 else j['ID']
    tree = {}
    root_ids = []
    for js in trees:
        for j in js:
            n = copy.deepcopy(j)
            n['parent'] = id_map[n['parent']] if n['parent'] else None
            n['children'] = [id_map[c] for c in n['children']] if n['children'] else None
            n['ID'] = id_map[n['ID']]
            tree[n['ID']] = n
            if n['parent'] is None:
                root_ids.append(n['ID'])
    root_id = min(root_ids)
    root_children = list(itertools.chain.from_iterable([tree[i]['children'] for i in root_ids]))
    root_node = {'ID': root_id, 'parent': None, 'children': root_children, 'depth': tree_depth}
    [tree.pop(i) for i in root_ids]
    for c in root_children:
        tree[c]['parent'] = root_id
    tree[root_id] = root_node
    g['tree'] = tree
    return g


def struct_tree(dataset, tree_depth=3):
    if not os.path.exists('trees'):
        os.makedirs('trees')
    if os.path.exists('trees/%s_%s.pickle' % (dataset, tree_depth)):
        return
    g_list = load_graph(dataset)
    pool_func = pool_trans_discon if dataset in discon_datasets else pool_trans
    pool = Pool()
    g_list = pool.map(pool_func, [(g, tree_depth) for g in g_list])
    pool.close()
    pool.join()
    g_list = filter(lambda g: g is not None, g_list)
    with open('trees/%s_%s.pickle' % (dataset, tree_depth), 'wb') as fp:
        pickle.dump(list(g_list), fp)

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

def build_coding_tree_single(G: nx.Graph, tree_depth: int, k: int = 2):
    # 0..N-1 연속 인덱스 보장
    mapping = {n:i for i,n in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping, copy=True)
    # 자기 루프 제거
    G.remove_edges_from(nx.selfloop_edges(G))
    # adjacency
    adj = nx.to_numpy_array(G, nodelist=range(G.number_of_nodes()))
    # PartitionTree
    pt = PartitionTree(adj_matrix=adj)
    _ = pt.build_coding_tree(k)
    tree = pt.tree_node  # {id: node_obj}
    # update_node (네 코드 그대로)
    # -- child_h 계산
    wait = [k for k,v in tree.items() if v.children is None]
    while wait:
        for nid in wait:
            node = tree[nid]
            if node.children is None:
                node.child_h = 0
            else:
                node.child_h = tree[list(node.children)[0]].child_h + 1
        wait = set([tree[nid].parent for nid in wait if tree[nid].parent])
    # -- ID/parent/children 재매핑
    d_id = sorted([(v.child_h, v.ID) for _,v in tree.items()])
    new_tree = {}
    for _, v in tree.items():
        n = copy.deepcopy(v)
        n.ID = d_id.index((n.child_h, n.ID))
        if n.parent is not None:
            n.parent = d_id.index((n.child_h+1, n.parent))
        if n.children is not None:
            n.children = [d_id.index((n.child_h-1, c)) for c in n.children]
        n = n.__dict__
        n['depth'] = n['child_h']
        new_tree[n['ID']] = n
    return new_tree  # {ID: {"ID":..,"parent":..,"children":[..],"depth":..}}

def tree_to_treebatch_item(tree: dict, tree_depth: int):
    """
    tree: build_coding_tree_single() 결과
    return: {"node_size":[...], "edges":{level:[[p,c],...]}}
    """
    # depth별 노드 모으기
    levels = {}
    for nid, nd in tree.items():
        d = nd["depth"]
        levels.setdefault(d, []).append(nid)
    max_d = max(levels.keys())
    # 레벨 0..max_d 정렬/재인덱스
    for d in levels:
        levels[d].sort()
    id2idx = {d: {nid:i for i,nid in enumerate(levels[d])} for d in levels}
    node_size = [len(levels.get(d, [])) for d in range(max_d+1)]
    # edges dict (level >=1)
    edges = {}
    for l in range(1, max_d+1):
        e = []
        for p_nid in levels[l]:
            ch = tree[p_nid].get("children") or []
            for c_nid in ch:
                e.append([ id2idx[l][p_nid], id2idx[l-1][c_nid] ])  # (parent, child)
        edges[l] = e
    # tree_depth 체크(요청 깊이와 실제 생성 깊이를 맞춤)
    assert (max_d+1) == tree_depth, f"tree_depth mismatch: built {max_d+1}, expected {tree_depth}"
    return {"node_size": node_size, "edges": edges}


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


# === Dataset ===
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
    ):
        self.df = df.reset_index(drop=True)
        self.tree_depth = tree_depth
        self.k = k
        self.use_sep = use_sep

        self._tree_cache = {}
        self._ligand_cache = {}  # SMILES -> DGData 캐시(선택적, 속도 향상)

        self.protein_contact_threshold = protein_contact_threshold
        self.protein_batch_size = protein_batch_size
        self.protein_weight_mode = protein_weight_mode
        self._protein_cache = {}

        if preload_protein:
            unique_seqs = self.df["Target"].astype(str).unique().tolist()
            if len(unique_seqs) > 0:
                raw_cache = protein_init(
                    unique_seqs,
                    contact_threshold=self.protein_contact_threshold,
                    batch_size=self.protein_batch_size,
                    weight_mode=self.protein_weight_mode,
                )
                if not isinstance(raw_cache, dict):
                    raise RuntimeError("protein_init did not return a dict as expected.")
                self._protein_cache = map_nested_to_cpu(raw_cache)

    def __len__(self):
        return len(self.df)

    def _get_protein_entry(self, seq: str):
        if seq in self._protein_cache:
            return self._protein_cache[seq]
        single = protein_init(
            [seq],
            contact_threshold=self.protein_contact_threshold,
            batch_size=1,
            weight_mode=self.protein_weight_mode,
        )
        if not isinstance(single, dict) or seq not in single:
            raise ValueError(f"protein_init failed or returned empty for seq[:10]={seq[:10]}...")
        single_cpu = map_nested_to_cpu(single)
        self._protein_cache.update(single_cpu)
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
            graph = mol2graph(mol)              # 내부에서 CPU 텐서 생성되도록 구현되어 있어야 함
            self._ligand_cache[smiles] = graph  # 캐시

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

        # --- Ligand tree (optional) ---
        if self.use_sep:
            key = smiles
            if key not in self._tree_cache:
                nxG = dgdata_to_nx(graph)
                tree = build_coding_tree_single(nxG, tree_depth=self.tree_depth, k=self.k)
                tree_item = tree_to_treebatch_item(tree, tree_depth=self.tree_depth)
                self._tree_cache[key] = tree_item
            item["tree"] = self._tree_cache[key]

        # --- Protein graph ---
        prot = self._get_protein_entry(seq)  # CPU 텐서/넘파이만

        # edge_index 모양 통일 (2, E)
        edge_index = to_cpu_tensor(prot["edge_index"], torch.long)
        if edge_index.dim() == 2 and edge_index.size(0) != 2 and edge_index.size(1) == 2:
            edge_index = edge_index.t().contiguous()
        E = edge_index.size(1)

        # weight / features / nodes
        edge_weight = _to_1d_cpu_float(prot.get("edge_weight"))
        if E == 0:
            edge_weight = torch.empty(0, dtype=torch.float32)
        x_tok = to_cpu_tensor(prot["token_representation"], torch.float32)  # [L, D]
        num_nodes = int(prot.get("num_nodes", x_tok.size(0)))
        num_nodes = min(num_nodes, x_tok.size(0))
        pos = to_cpu_tensor(prot.get("num_pos", None), torch.float32)

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

        # coalesce로 최종 정규화
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
        protein_graph.seq_feat = map_nested_to_cpu(prot.get("seq_feat"))
        protein_graph.secondary_structure = map_nested_to_cpu(prot.get("secondary_structure"))
        protein_graph.sasa = map_nested_to_cpu(prot.get("sasa"))
        protein_graph.function_logits = map_nested_to_cpu(prot.get("function_logits"))

        item["protein_graph"] = protein_graph
        item["protein_seq"] = seq

        return item

class PrecomputedBindingDataset(Dataset):
    def __init__(self, root_dir: str, title: str = "bindingdb_kd"):
        self.root = Path(root_dir)
        with open(self.root / "meta.json", "r") as f:
            meta = json.load(f)
        self.meta = meta
        self.title = title
        self.num_samples = meta["num_samples"]
        self.shard_size = meta["shard_size"]
        self.num_shards = meta["num_shards"]

        # 샤드 경로들 미리 정리
        self.shards = [
            self.root / "shards" / f"{self.title}_shard_{i:05d}.pt"
            for i in range(self.num_shards)
        ]
        self._cache_shard_id = None
        self._cache_shard_data = None  # 최근 샤드만 메모리에 유지

    def __len__(self):
        return self.num_samples

    def _load_shard(self, shard_id: int):
        if self._cache_shard_id == shard_id and self._cache_shard_data is not None:
            return self._cache_shard_data
        path = self.shards[shard_id]
        data_list = torch.load(path, map_location="cpu")
        self._cache_shard_id = shard_id
        self._cache_shard_data = data_list
        return data_list

    def __getitem__(self, idx: int):
        shard_id = idx // self.shard_size
        offset = idx % self.shard_size
        shard = self._load_shard(shard_id)
        sample = shard[offset]
        # 안전상 CPU 보장
        sample["label"] = sample["label"].detach().to("cpu").contiguous()
        # graph / protein_graph은 저장 시 이미 CPU PyG Data로 저장됨
        return sample

