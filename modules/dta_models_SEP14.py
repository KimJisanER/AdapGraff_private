import torch
from torch.nn import Sequential, Dropout, Linear
from torch_geometric.nn.models import GCN, GIN, GAT
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import  add_self_loops
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from .dualgraph.gnn import GNN
from torch.nn.utils.rnn import pad_sequence
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from transformers import PreTrainedTokenizerFast, BertTokenizer
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from typing import Callable
from functools import reduce
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GINConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import ChebConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, Size
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_mean_pool as gap, global_add_pool as gsp
from torch_geometric.nn import SAGPooling, ASAPooling
from torch_geometric.nn.dense import dense_diff_pool, dense_mincut_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj, subgraph
from torch.cuda.amp import autocast, GradScaler

class SEPooling(MessagePassing):
    def __init__(self, nn: nn.Module, **kwargs):
        kwargs.setdefault('aggr', 'add')
        # kwargs.setdefault('flow', 'target_to_source')  # child -> parent
        kwargs.setdefault('flow', 'source_to_target')
        super().__init__(**kwargs)
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x: torch.Tensor, edge_index: Adj, size: Size = None) -> torch.Tensor:
        out = self.propagate(edge_index, x=x, size=size)
        return self.nn(out)  # SEP 코드와 달리 MLP 활성화

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: torch.Tensor) -> torch.Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x, reduce=self.aggr)

class MPNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin_node = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(13, out_channels)

    def forward(self, x, edge_index, edge_attr):
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=x.size(0))
        x = self.lin_node(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return x_j + self.lin_edge(edge_attr)

    def update(self, aggr_out):
        return aggr_out

class Attention(torch.nn.Module):
    def __init__(self, channels, dropout, n_classes):
        super().__init__()
        self.attn = Sequential(
                                Linear(channels, channels, bias=False),
                                torch.nn.PReLU(init=0.05),
                                Dropout(dropout),
                                torch.nn.Tanh()
                                )

        self.fc = Sequential(
            torch.nn.LayerNorm(channels),
            Linear(channels, channels),
            torch.nn.BatchNorm1d(channels),
            torch.nn.PReLU(init=0.05),
            Dropout(dropout),
            Linear(channels, n_classes)
        )

    def forward(self, x, return_attn=False):
        A = self.attn(x)
        mul_x = torch.mul(x, A)
        if return_attn:
            return mul_x, self.fc(mul_x)
        return self.fc(mul_x)

class LigandEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, gnn_type="gin", num_layers=3, dropout: float = 0.1):
        super().__init__()
        self.gnn_type = gnn_type.lower()
        self.output_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            in_c = in_dim if i == 0 else hidden_dim
            if self.gnn_type == "gcn":
                self.layers.append(GCNConv(in_c, hidden_dim))
            elif self.gnn_type == "gat":
                self.layers.append(GATConv(in_c, hidden_dim, heads=1, concat=False))
            elif self.gnn_type == "gin":
                nn_seq = nn.Sequential(
                    nn.Linear(in_c, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.layers.append(GINConv(nn_seq))
            else:
                raise ValueError(f"Unsupported GNN type: {gnn_type}")

        self.readout = global_add_pool  # from torch_geometric.nn

    def forward(
        self,
        data_or_x,
        edge_index=None,
        batch=None,
        edge_attr=None,
        return_node_embeddings: bool = False,
        **unused,
    ):
        # --- 입력 해석 ---
        if hasattr(data_or_x, "x") and hasattr(data_or_x, "edge_index"):
            x = data_or_x.x
            ei = data_or_x.edge_index
            bt = getattr(data_or_x, "batch", None)
            ea = getattr(data_or_x, "edge_attr", None)
        else:
            x = data_or_x
            ei = edge_index
            bt = batch
            ea = edge_attr

        if bt is None:
            bt = x.new_zeros(x.size(0), dtype=torch.long)

        # --- GNN 스택 ---
        for conv in self.layers:
            x = conv(x, ei)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)  # ✅ dropout 추가

        node_feat = x  # (N, hidden_dim)
        graph_feat = self.readout(node_feat, bt)  # (B, hidden_dim)

        if return_node_embeddings:
            return graph_feat, node_feat
        return graph_feat

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, return_attn_weights=False):
        attn_output, attn_weights = self.attn(query, key_value, key_value)
        output = self.norm(query + self.dropout(attn_output))
        if return_attn_weights:
            return output, attn_weights  # Shape: (B, L_query, L_key)
        return output


def _shape(x, name):
    print(f"{name}: {tuple(x.shape)}")


def _tstats(x: torch.Tensor):
    x = x.detach()
    f = torch.isfinite(x)
    return dict(
        shape=tuple(x.shape),
        min=float(x[f].min()) if f.any() else float('nan'),
        max=float(x[f].max()) if f.any() else float('nan'),
        mean=float(x[f].mean()) if f.any() else float('nan'),
        n_nan=int(torch.isnan(x).sum().item()),
        n_inf=int(torch.isinf(x).sum().item()),
    )

def _finite_or_handle(x: torch.Tensor, where: str, policy: str = "raise"):
    if not torch.is_tensor(x):
        return x
    if torch.isfinite(x).all():
        return x
    n_nan = int(torch.isnan(x).sum().item())
    n_inf = int(torch.isinf(x).sum().item())
    print(f"[NaNGuard] {where}: nan={n_nan}, inf={n_inf}, stats={_tstats(x)}")
    if policy == "replace":
        return torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    else:
        raise RuntimeError(f"Non-finite detected at {where} (nan={n_nan}, inf={n_inf})")

def attach_grad_nan_hooks(model: nn.Module):
    def make_hook(pname):
        def _hook(grad):
            if grad is None:
                return
            if not torch.isfinite(grad).all():
                print(f"[GradNaN] {pname}: stats={_tstats(grad)}")
        return _hook

    for name, p in model.named_parameters():
        if p.requires_grad:
            p.register_hook(make_hook(name))

def attach_forward_nan_hooks(model: nn.Module):
    watch_types = (nn.Linear, nn.LayerNorm, nn.MultiheadAttention, SAGPooling)
    for name, mod in model.named_modules():
        if not isinstance(mod, watch_types):
            continue

        def pre_hook(mod, inputs):
            for i, t in enumerate(inputs):
                if torch.is_tensor(t) and not torch.isfinite(t).all():
                    print(f"[Hook-Pre] {mod.__class__.__name__}({name}) arg{i}: {_tstats(t)}")

        def post_hook(mod, inputs, output):
            outs = output if isinstance(output, (tuple, list)) else (output,)
            for i, t in enumerate(outs):
                if torch.is_tensor(t) and not torch.isfinite(t).all():
                    print(f"[Hook-Post] {mod.__class__.__name__}({name}) out{i}: {_tstats(t)}")

        mod.register_forward_pre_hook(pre_hook)
        mod.register_forward_hook(post_hook)

# === 유틸 함수들 근처(클래스 바깥)에 추가 ===
def _validate_pg(x_cat, g):
    # x_cat: (N_total, D)
    assert x_cat.dim() == 2, f"x_cat shape {x_cat.shape}"
    # edge_index
    assert hasattr(g, "edge_index"), "protein_graphs.edge_index missing"
    assert g.edge_index.dtype == torch.long and g.edge_index.dim() == 2 and g.edge_index.size(0) == 2, \
        f"edge_index must be (2,E) long, got {g.edge_index.dtype}, {tuple(g.edge_index.shape)}"
    # batch
    assert hasattr(g, "batch"), "protein_graphs.batch missing"
    assert g.batch.numel() == x_cat.size(0), f"N mismatch: batch({g.batch.numel()}) vs x_cat({x_cat.size(0)})"
    # device align
    assert g.edge_index.device == x_cat.device and g.batch.device == x_cat.device, \
        f"device mismatch: x({x_cat.device}) ei({g.edge_index.device}) batch({g.batch.device})"

    # 배치 라벨 0..B-1로 연속화
    uniq, inv = torch.unique(g.batch, sorted=True, return_inverse=True)
    if not torch.equal(uniq, torch.arange(uniq.numel(), device=uniq.device)):
        # in-place 재매핑
        g.batch = inv

    # 엣지 정리(무향 + coalesce) — 선택사항이지만 권장
    from torch_geometric.utils import to_undirected, coalesce
    g.edge_index, _ = coalesce(to_undirected(g.edge_index), None, x_cat.size(0), x_cat.size(0))

    # edge_weight 체크
    if hasattr(g, "edge_weight") and g.edge_weight is not None:
        assert g.edge_weight.device == x_cat.device, "edge_weight device mismatch"
        assert g.edge_weight.numel() == g.edge_index.size(1), "edge_weight length != E"
        assert torch.isfinite(g.edge_weight).all(), "edge_weight has NaN/Inf"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import SAGPooling  # ✅ 새로 추가
# 나머지 GNN/SEPooling/CrossAttention/ESMC, ESMProtein, LogitsConfig 등은 기존과 동일하게 import

class MultimodalBindingAffinityModel(nn.Module):
    def __init__(
        self,
        in_dim=None,
        hidden_dim=128,
        fusion_dim=256,
        ddi=None,
        esm_model_name="esmc_300m",
        dropout_fc=0.1,
        device="cuda",
        freeze_esm=True,
        num_unfrozen_layers=4,
        # GNN (dualgraph)
        mlp_hidden_size=512,
        mlp_layers=2,
        use_layer_norm=False,
        use_face=True,
        dropedge_rate=0.1,
        dropnode_rate=0.1,
        dropout=0.1,
        dropnet=0.1,
        global_reducer="sum",
        node_reducer="sum",
        face_reducer="sum",
        graph_pooling="sum",
        node_attn=True,
        face_attn=True,
        som_mode=False,
        tree_depth: int = 3,
        sepool_hidden_dim: int = None,
        # LigandEncoder
        ligand_gnn_type="dualgraph",
        ligand_in_dim=None,
        ligand_hidden_dim=128,
        ligand_num_layers=6,
        # --- NEW: Ligand pooling 옵션 ---
        ligand_pool_type: str = "sep",  # "sep" | "diffpool" | "mincut" | "none"
        ligand_pool_ratio: float = 0.5,  # Diff/MinCut에서 선택할 클러스터 개수 비율
        ligand_pool_min_clusters: int = 4,  # 최소 유지 클러스터 수
        ligand_max_clusters: int | None = None,  # Diff/MinCut용 고정 최대 클러스터 수(C)
        # --- Protein structure 사용 옵션 ---
        use_protein_graph_structure: bool = True,
        protein_mp_layers: int = 2,
        # --- NEW: SAGPooling 옵션 ---
        protein_pool_type: str = "sag",     # "sag"|"asap"|"mincut"|"diffpool"|"none"
        protein_pool_ratio: float = 0.5,
        protein_pool_min_nodes: int = 8,
        protein_max_clusters: int | None = None,  # Diff/MinCut용 고정 최대 클러스터 수
        debug_nan: bool = False,  # << 추가
        nan_policy: str = "raise",
        return_protein_pool_info: bool = False,
            # 그래프 당 최소 유지 노드수 (보호 장치)
    ):
        super().__init__()
        self.device = device
        self.ligand_gnn_type = ligand_gnn_type.lower()
        self.ddi = ddi
        self.tree_depth = tree_depth

        self.use_protein_graph_structure = use_protein_graph_structure
        self.protein_mp_layers = max(0, int(protein_mp_layers))
        # ▼ 새 파라미터 저장
        self.protein_pool_type = str(protein_pool_type).lower()
        # (하위호환) 예전 플래그가 켜져 있고 타입이 지정 안되면 sag로
        self.protein_pool_ratio = float(protein_pool_ratio)
        self.protein_pool_min_nodes = int(protein_pool_min_nodes)
        self.protein_max_clusters = int(protein_max_clusters) if protein_max_clusters is not None else None

        self.nan_policy = {
            "mode": "warn",    # "error" | "warn" | "silent"
            "replace": 0.0,    # NaN을 이 값으로 대체
            "clip": 1e6        # ±clip 으로 값 클리핑 (Inf 방지)
        }
        # __init__
        self.pool_loss_weights = dict(
            prot_diff_link=1.0,  # DiffPool A 재구성
            prot_diff_ent=1e-3,  # DiffPool 클러스터 엔트로피
            prot_mc=1.0,  # MinCut min-cut 항
            prot_ortho=1.0,  # MinCut orthogonality
            lig_diff_link=1.0,
            lig_diff_ent=1e-3,
            lig_mc=1.0,
            lig_ortho=1.0,
        )

        self.pool_loss_weights.update({
            "prot_sag_smooth": 1.0,
            "prot_sag_ent": 1e-3,
            "lig_sep_cons": 1.0,
            "lig_sep_ortho": 1e-3,
        })

        self._aux_losses = {}  # forward마다 갱신

        # ----- Ligand encoder (동일) -----
        if self.ligand_gnn_type == "dualgraph":
            self.gnn = GNN(
                mlp_hidden_size=mlp_hidden_size,
                mlp_layers=mlp_layers,
                latent_size=hidden_dim,
                use_layer_norm=use_layer_norm,
                use_face=use_face,
                ddi=ddi,
                dropedge_rate=dropedge_rate,
                dropnode_rate=dropnode_rate,
                dropout=dropout,
                dropnet=dropnet,
                global_reducer=global_reducer,
                node_reducer=node_reducer,
                face_reducer=face_reducer,
                graph_pooling=graph_pooling,
                node_attn=node_attn,
                face_attn=face_attn,
                som_mode=som_mode,
            )
            ligand_output_dim = hidden_dim
        else:
            self.gnn = LigandEncoder(
                in_dim=ligand_in_dim if ligand_in_dim is not None else in_dim,
                hidden_dim=ligand_hidden_dim,
                gnn_type=ligand_gnn_type,
                num_layers=ligand_num_layers,
                dropout=dropout,
            )
            ligand_output_dim = self.gnn.output_dim

        # ----- Protein (ESMC features) -----
        self.esm = ESMC.from_pretrained(esm_model_name).to(device)
        if freeze_esm:
            for p in self.esm.parameters():
                p.requires_grad = False
            if hasattr(self.esm, "transformer") and hasattr(self.esm.transformer, "blocks"):
                total = len(self.esm.transformer.blocks)
                for i in range(total - num_unfrozen_layers, total):
                    for p in self.esm.transformer.blocks[i].parameters():
                        p.requires_grad = True

        # ESMC residue-embedding(960) → fusion_dim
        self.protein_proj = nn.Sequential(
            nn.Linear(960, fusion_dim),
            nn.ReLU(),
        )

        # 구조 반영용 간단 MP(ESMC 임베딩 위)
        self.protein_mp_self = nn.ModuleList([nn.Linear(fusion_dim, fusion_dim) for _ in range(self.protein_mp_layers)])
        self.protein_mp_nei  = nn.ModuleList([nn.Linear(fusion_dim, fusion_dim) for _ in range(self.protein_mp_layers)])
        self.protein_mp_ln   = nn.ModuleList([nn.LayerNorm(fusion_dim) for _ in range(self.protein_mp_layers)])
        self.protein_mp_drop = nn.Dropout(dropout)

        # ✅ Pooling 모듈 준비
        if self.protein_pool_type == "sag":
            self.protein_pool = SAGPooling(in_channels=fusion_dim, ratio=self.protein_pool_ratio)
        elif self.protein_pool_type == "asap":
            self.protein_pool = ASAPooling(in_channels=fusion_dim, ratio=self.protein_pool_ratio)
        elif self.protein_pool_type in ("diffpool", "mincut"):
            # 고정 클러스터 채널 수(배치 간 동일해야 합니다)
            self.protein_max_clusters = self.protein_max_clusters or 64
            self.protein_assign = nn.Linear(fusion_dim, self.protein_max_clusters)
        elif self.protein_pool_type == "none":
            self.protein_pool = None
        else:
            raise ValueError(f"Unknown protein_pool_type={self.protein_pool_type}")

        self.ligand_proj = nn.Linear(ligand_output_dim, fusion_dim)

        # --- NEW: Ligand pooling 설정 ---
        self.ligand_pool_type = str(ligand_pool_type).lower()
        self.ligand_pool_ratio = float(ligand_pool_ratio)
        self.ligand_pool_min_clusters = int(ligand_pool_min_clusters)
        self.ligand_max_clusters = int(ligand_max_clusters) if ligand_max_clusters is not None else 32
        if self.ligand_pool_type in ("diffpool", "mincut"):
            # 간단한 클러스터 할당기 (원하면 GNN으로 대체 가능)
            self.ligand_assign = nn.Linear(fusion_dim, self.ligand_max_clusters)

        # --- SEP (동일) ---
        if self.ligand_pool_type == "sep":
            self.sepools = nn.ModuleList()
            _in = fusion_dim
            _out = fusion_dim if sepool_hidden_dim is None else sepool_hidden_dim
            for _ in range(self.tree_depth - 1):
                mlp = nn.Sequential(
                    nn.Linear(_in, _out),
                    nn.ReLU(),
                    nn.Linear(_out, _out),
                    nn.ReLU(),
                    nn.LayerNorm(_out),
                )
                self.sepools.append(SEPooling(mlp))
                _in = _out
            self._fusion_dim_after_sepool = _out
        else:
            self._fusion_dim_after_sepool = fusion_dim

        self._fusion_dim_after_sepool = fusion_dim if sepool_hidden_dim is None else sepool_hidden_dim
        self.sepool_kv_proj = None
        if self.ligand_pool_type == "sep" and self._fusion_dim_after_sepool != fusion_dim:
            self.sepool_kv_proj = nn.Linear(self._fusion_dim_after_sepool, fusion_dim)

        self.cross_attn = CrossAttention(fusion_dim, num_heads=4, dropout=dropout_fc)

        self.cross_attn_pl = CrossAttention(fusion_dim, num_heads=4, dropout=dropout_fc)  # protein  ← ligand
        self.cross_attn_lp = CrossAttention(fusion_dim, num_heads=4, dropout=dropout_fc)  # ligand   ← protein
        self.ca_drop = nn.Dropout(dropout_fc)

        def _ffn(dim):
            return nn.Sequential(
                nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout_fc),
                nn.Linear(dim * 4, dim), nn.Dropout(dropout_fc)
            )

        self.prot_ln1 = nn.LayerNorm(fusion_dim)
        self.prot_ln2 = nn.LayerNorm(fusion_dim)
        self.prot_ffn = _ffn(fusion_dim)

        self.lig_ln1 = nn.LayerNorm(fusion_dim)
        self.lig_ln2 = nn.LayerNorm(fusion_dim)
        self.lig_ffn = _ffn(fusion_dim)

        self.pre_norm_prot = nn.LayerNorm(fusion_dim)
        self.pre_norm_lig = nn.LayerNorm(fusion_dim)

        self.regressor = nn.Sequential(
            nn.LayerNorm(fusion_dim * 2),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.PReLU(init=0.05),
            nn.Dropout(dropout_fc),
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.PReLU(init=0.05),
            nn.Dropout(dropout_fc),
            nn.Linear(fusion_dim // 2, 1),
        )

    # ---------------- util ----------------
    @staticmethod
    def _edge_aggregate(x, edge_index, edge_weight=None):
        if edge_index.numel() == 0:
            return torch.zeros_like(x)
        src, dst = edge_index[0], edge_index[1]
        if edge_weight is None:
            m = x[src]
        else:
            w = edge_weight
            if w.dim() == 2 and w.size(1) == 1:
                w = w.view(-1)
            w = 1.0 / (w + 1e-6)   # 거리 → 역수 가중
            w = torch.clamp(w, max=10.0)
            m = x[src] * w.unsqueeze(-1)
        out = torch.zeros_like(x)
        out.index_add_(0, dst, m)
        return out

    @staticmethod
    def _split_by_batch(x, batch, B: int):
        return [x[batch == b] for b in range(B)]

    @staticmethod
    def _build_sep_edge_index(tree_batch, level: int, device):
        """
        tree_batch[g]['edges'][level] : (E_g, 2) 형태의 [parent, child] 인덱스
        반환: edge_index (2, E) with [src=child(level-1), dst=parent(level)]  ← SEPooling에 맞춘 방향
        """
        pieces = []
        src_off, dst_off = 0, 0
        for g in tree_batch:
            # g['edges'][level]은 각 그래프의 parent-child 쌍
            e = torch.as_tensor(g['edges'][level], dtype=torch.long, device=device)  # (E_g, 2)
            if e.numel() > 0:
                # child는 level-1의 노드, parent는 level의 노드
                src = e[:, 1] + src_off  # child + offset(level-1)
                dst = e[:, 0] + dst_off  # parent + offset(level)
                pieces.append(torch.stack([src, dst], dim=0))  # (2, E_g)
            # 다음 그래프로 offsets 갱신
            src_off += int(g['node_size'][level - 1])
            dst_off += int(g['node_size'][level])
        if not pieces:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        return torch.cat(pieces, dim=1)  # (2, E)

    @staticmethod
    def _aux_sag_asap_losses(x_pool, ei_pool, batch_out, score):
        device = x_pool.device
        B = int(batch_out.max().item()) + 1 if batch_out.numel() else 0
        loss_smooth = torch.zeros((), device=device, dtype=x_pool.dtype)
        loss_ent    = torch.zeros((), device=device, dtype=x_pool.dtype)

        for b in range(B):
            mask_b = (batch_out == b)
            idx_b = mask_b.nonzero(as_tuple=True)[0]
            if idx_b.numel() <= 1:
                continue

            # 풀링 후 그래프에서 배치 b의 서브그래프
            ei_b, *_ = subgraph(idx_b, ei_pool, relabel_nodes=True, num_nodes=x_pool.size(0))
            s_b = score[mask_b]  # (n_b,)

            # (1) 점수 매끄러움: Σ_ij A_ij (s_i - s_j)^2
            if ei_b.numel() > 0:
                loss_smooth = loss_smooth + ((s_b[ei_b[0]] - s_b[ei_b[1]])**2).mean()

            # (2) 이진화 유도: 시그모이드 엔트로피 최소화
            p_b = torch.sigmoid(s_b)
            ent_b = -(p_b * torch.log(p_b + 1e-8) + (1 - p_b) * torch.log(1 - p_b + 1e-8)).mean()
            loss_ent = loss_ent + ent_b

        # 배치 평균
        denom = max(1, B)
        return loss_smooth / denom, loss_ent / denom

    @staticmethod
    def _aux_sep_losses(level_feats, tree_batch, device):
        import torch.nn.functional as F
        if (level_feats is None) or (len(level_feats) <= 1):
            # 레벨 1개(=클러스터 없음)면 0 반환
            zero = torch.zeros((), device=device)
            return zero, zero

        loss_cons = torch.zeros((), device=device)
        loss_ortho = torch.zeros((), device=device)

        num_levels = 0
        for lvl in range(1, len(level_feats)):
            # parent-children 인덱스
            # child → parent 방향(edge_index[0]=child, edge_index[1]=parent)
            # (이 함수는 forward의 _build_sep_edge_index와 동일 포맷 사용 가정)
            # forward에서 쓰는 것을 재사용하려면 호출부에서 edge_index를 전달하거나,
            # 여기서 동일 로직을 간단히 inline할 수도 있음.
            # 간결성을 위해 호출부에서 ei_lvl을 만들어 전달하는 방식으로도 변경 가능.
            # 여기서는 tree_batch에서 다시 만들자:
            pieces = []
            src_off, dst_off = 0, 0
            for g in tree_batch:
                e = torch.as_tensor(g['edges'][lvl], dtype=torch.long, device=device) if g['edges'][lvl] is not None else None
                if e is not None and e.numel() > 0:
                    src = e[:, 1] + src_off  # child indices(level-1)
                    dst = e[:, 0] + dst_off  # parent indices(level)
                    pieces.append(torch.stack([src, dst], dim=0))
                src_off += int(g['node_size'][lvl - 1])
                dst_off += int(g['node_size'][lvl])
            if not pieces:
                continue
            ei_lvl = torch.cat(pieces, dim=1)  # (2, E)
            src, dst = ei_lvl[0], ei_lvl[1]

            Zc = level_feats[lvl - 1]  # children
            Zp = level_feats[lvl]      # parents
            # parent별 children 평균 (scatter-mean)
            mean_child = torch.zeros_like(Zp)
            mean_child.index_add_(0, dst, Zc[src])
            counts = torch.zeros(Zp.size(0), device=device, dtype=Zp.dtype)
            one = torch.ones_like(dst, dtype=Zp.dtype)
            counts.index_add_(0, dst, one)
            counts = counts.clamp_min(1.0).unsqueeze(-1)
            mean_child = mean_child / counts

            # (1) parent ≈ mean(children)
            loss_cons = loss_cons + F.mse_loss(Zp, mean_child)

            # (2) 같은 그래프 내 parent 간 직교
            sizes_p = [int(g['node_size'][lvl]) for g in tree_batch]
            off = 0
            for sz in sizes_p:
                if sz >= 2:
                    Zpb = Zp[off:off+sz]
                    Zpb = F.normalize(Zpb, dim=1)
                    G = Zpb @ Zpb.t()
                    offdiag = G - torch.eye(sz, device=device)
                    loss_ortho = loss_ortho + (offdiag**2).mean()
                off += sz

            num_levels += 1

        denom = max(1, num_levels)
        return loss_cons / denom, loss_ortho / denom

    @staticmethod
    def _sep_size(tree_batch, level: int):
        """
        메시지패싱 size 힌트: (num_src=level-1 총 노드수, num_dst=level 총 노드수)
        """
        num_src = sum(int(g['node_size'][level - 1]) for g in tree_batch)
        num_dst = sum(int(g['node_size'][level]) for g in tree_batch)
        return (num_src, num_dst)

    def _chk(self, x, name):
        # 모델 메서드로 쓰기 편하게 래핑
        return _finite_or_handle(x, name, self.nan_policy)

    def _compute_lengths_offsets(self, seq_feats):
        lengths = [t.size(0) for t in seq_feats]
        offsets = [0]
        for L in lengths[:-1]:
            offsets.append(offsets[-1] + L)
        return lengths, offsets

    def pool_regularizer(self):
        reg = None
        for k, v in self._aux_losses.items():
            w = float(self.pool_loss_weights.get(k, 0.0))
            if w != 0.0:
                term = v * w
                reg = term if reg is None else reg + term
        if reg is None:
            p = next(self.parameters(), None)
            device = p.device if p is not None else torch.device("cpu")
            dtype = p.dtype if p is not None else torch.float32
            reg = torch.zeros((), device=device, dtype=dtype)
        return reg

    def _protein_pool(self, x_cat, protein_graphs, seq_feats, dbg=False):
        """
        선택된 pooling 방식으로 단백질 토큰을 생성.
        반환: prot_tokens(list[Tensor[B_i, D]]), protein_pool_info(list[dict])
        """
        B = len(seq_feats)
        lengths_esmc = [t.size(0) for t in seq_feats]
        lengths, offsets = self._compute_lengths_offsets(seq_feats)
        ei = protein_graphs.edge_index
        ew = getattr(protein_graphs, "edge_weight", None)
        batch = protein_graphs.batch

        prot_tokens = []
        protein_pool_info = []

        if self.protein_pool_type in ("sag", "asap"):
            # (선택) edge weight 변환
            if ew is not None:
                tau = 5.0
                ew = self._chk(ew, "edge_weight_raw")
                ew = torch.exp(-ew / tau).clamp_(0.0, 1.0)

            with torch.cuda.amp.autocast(enabled=False):
                # ASAP도 동일 시그니처로 동작 (edge_attr=None 가능)
                x_pool, ei_pool, ew_pool, batch_out, perm, score = self.protein_pool(
                    x_cat.float(), ei, ew, batch=batch
                )

            # ⬇ 추가
            loss_smooth, loss_ent = self._aux_sag_asap_losses(x_pool, ei_pool, batch_out, score)
            self._aux_losses["prot_sag_smooth"] = loss_smooth
            self._aux_losses["prot_sag_ent"] = loss_ent

            if dbg:
                print(f"[POOL-{self.protein_pool_type.upper()}] in={x_cat.size(0)} out={x_pool.size(0)} "
                      f"ratio={x_pool.size(0) / max(1, x_cat.size(0)):.2f}", flush=True)

            x_pool = self._chk(x_pool, "POOL/x_pool")
            score = self._chk(score, "POOL/score")

            # 그래프별로 다시 나누기
            for b in range(B):
                mask_b = (batch_out == b)
                if mask_b.sum() == 0:
                    # fallback: 최소 노드 유지
                    src_mask = (batch == b)
                    src_idx = src_mask.nonzero(as_tuple=True)[0]
                    keep = max(1, min(self.protein_pool_min_nodes, int(src_idx.numel())))
                    idx_keep = src_idx[:keep]
                    prot_tokens.append(x_cat[idx_keep])
                    res_idx_fallback = (idx_keep - offsets[b]).detach().cpu()
                    protein_pool_info.append(dict(residue_idx=res_idx_fallback, score=None))
                    continue

                x_b = x_pool[mask_b]
                perm_b = perm[mask_b]
                score_b = score[mask_b]
                res_idx = perm_b - offsets[b]
                order = torch.argsort(res_idx)
                prot_tokens.append(x_b[order])
                protein_pool_info.append(dict(
                    residue_idx=res_idx[order].detach().cpu(),
                    score=score_b[order].detach().cpu()
                ))

            return prot_tokens, protein_pool_info

        elif self.protein_pool_type in ("diffpool", "mincut"):
            # --- Dense 변환 ---
            X_dense, mask = to_dense_batch(x_cat, batch)  # [B, Nmax, D], [B, Nmax]
            A_dense = to_dense_adj(ei, batch=batch)  # [B, Nmax, Nmax]

            # 고정 클러스터 채널 수(배치 공통)
            C = self.protein_max_clusters
            S_logits = self.protein_assign(X_dense)  # [B, Nmax, C]
            S = F.softmax(S_logits, dim=-1)
            S = S * mask.unsqueeze(-1)  # 패딩 노드 무효화

            if self.protein_pool_type == "diffpool":
                Xp, Ap, link_loss, ent_loss = dense_diff_pool(X_dense, A_dense, S, mask)
                # 학습 그래프에 남김 (detach 금지)
                self._aux_losses["prot_diff_link"] = link_loss
                self._aux_losses["prot_diff_ent"] = ent_loss
            else:  # "mincut"
                Xp, Ap, mc_loss, o_loss = dense_mincut_pool(X_dense, A_dense, S, mask)
                self._aux_losses["prot_mc"] = mc_loss
                self._aux_losses["prot_ortho"] = o_loss

            # --- 그래프별 유지 클러스터 수/선택
            N_per_graph = mask.sum(dim=1).long().tolist()  # [B]
            mass = S.sum(dim=1)  # [B, C]
            for b in range(B):
                Lb = int(N_per_graph[b])
                kb = max(self.protein_pool_min_nodes, int(self.protein_pool_ratio * Lb))
                kb = min(kb, C)

                topk = torch.topk(mass[b], k=kb, largest=True).indices  # kept clusters
                x_b = Xp[b, topk, :]  # [kb, D]
                prot_tokens.append(x_b)

                # --- NEW: residue→cluster 매핑과 점수까지 기록 ---
                S_b = S[b]  # [Nmax, C]
                z_b = S_b.argmax(dim=-1)  # [Nmax] residue별 argmax cluster
                # 실제 존재하는 residue만 (mask)
                valid = mask[b]  # [Nmax] 0/1
                z_b = z_b[valid]  # [L_b]
                S_b = S_b[valid]  # [L_b, C]

                # # kept cluster에 속한 residue만
                # kept_set = set(topk.detach().cpu().tolist())
                # z_cpu = z_b.detach().cpu()
                # keep_mask = torch.tensor([int(c in kept_set) for c in z_cpu], dtype=torch.bool, device=z_b.device)
                # res_idx = keep_mask.nonzero(as_tuple=True)[0]  # [L_kept]
                # res_clu = z_b[res_idx]  # [L_kept]
                # res_scr = S_b[res_idx, res_clu]  # [L_kept]
                
                # 유효 노드만 남긴 soft-assign / argmax
                S_b = S[b]  # [Nmax, C]
                valid = mask[b].bool()  # [Nmax]
                z_b = S_b.argmax(dim=-1)[valid]  # [Lb]
                S_bv = S_b[valid]  # [Lb, C]

                # kept cluster 멤버십: torch.isin 사용 (CPU로 맞춰 비교)
                keep_mask_cpu = torch.isin(z_b.detach().cpu(), topk.detach().cpu())
                res_idx = keep_mask_cpu.nonzero(as_tuple=True)[0].to(z_b.device)  # [L_kept]
                res_clu = z_b[res_idx]  # [L_kept]
                res_scr = S_bv[res_idx, res_clu]  # [L_kept]

                # ── Fallback: 만약 여전히 비면 전체를 기록(디버그용)
                if res_idx.numel() == 0:
                    # (a) 확률이 아주 작은 argmax만 남았을 수 있으니, max prob 기준 임계값으로 한 번 더 걸러보기
                    max_prob = S_bv.max(dim=-1).values  # [Lb]
                    thr_mask = (max_prob > 1e-8)
                    if thr_mask.any():
                        res_idx = thr_mask.nonzero(as_tuple=True)[0]
                        res_clu = z_b[res_idx]
                        res_scr = max_prob[res_idx]
                    else:
                        # (b) 완전 비정상일 때는 디버그를 위해 전체 기록
                        res_idx = torch.arange(z_b.size(0), device=z_b.device)
                        res_clu = z_b
                        res_scr = S_bv[torch.arange(S_bv.size(0), device=z_b.device), z_b]

                protein_pool_info.append(dict(
                    residue_idx=res_idx.detach().cpu(),  # kept residue indices (0..L_b-1)
                    cluster_idx=res_clu.detach().cpu(),  # residue별 cluster id
                    score=res_scr.detach().cpu(),  # residue별 argmax 확률
                    kept_clusters=topk.detach().cpu()  # 선택된 클러스터 id 목록
                ))

            return prot_tokens, protein_pool_info

        elif self.protein_pool_type == "none":
            # 풀링 생략: 원 토큰 그대로
            return [t for t in seq_feats], [dict(residue_idx=None, score=None) for _ in range(B)]

        else:
            raise ValueError(f"Unknown protein_pool_type={self.protein_pool_type}")

    def _ligand_pool_tokens(self, node_feat, ligand_batch, dbg=False):
        """
        DiffPool/MinCut에서 리간드 토큰(클러스터 임베딩들)을 그래프별 variable-length 리스트로 반환.
        반환: tokens_list = [Tensor(k_b, D) for b in batch]
        """
        node_batch = ligand_batch.batch  # [N]
        B = int(node_batch.max().item()) + 1

        # Dense 변환
        X, mask = to_dense_batch(node_feat, node_batch)  # [B, Nmax, D], [B, Nmax]
        A = to_dense_adj(ligand_batch.edge_index, node_batch)  # [B, Nmax, Nmax]

        # 할당 행렬 S: [B, Nmax, C]
        C = self.ligand_max_clusters
        S_logits = self.ligand_assign(X)
        S = F.softmax(S_logits, dim=-1)
        S = S * mask.unsqueeze(-1)  # 패딩 무효화

        # 풀링
        if self.ligand_pool_type == "diffpool":
            Xp, Ap, link_loss, ent_loss = dense_diff_pool(X, A, S, mask)
            self._aux_losses["lig_diff_link"] = link_loss
            self._aux_losses["lig_diff_ent"] = ent_loss
        else:
            Xp, Ap, mc_loss, o_loss = dense_mincut_pool(X, A, S, mask)
            self._aux_losses["lig_mc"] = mc_loss
            self._aux_losses["lig_ortho"] = o_loss

        # 각 그래프별 유지할 클러스터 수 k_b 결정 후 상위 질량 클러스터 선택
        mass = S.sum(dim=1)  # [B, C], 클러스터 질량(할당 합)
        N_per_graph = mask.sum(dim=1).long().tolist()

        tokens_list = []
        for b in range(B):
            Lb = int(N_per_graph[b])
            kb = max(self.ligand_pool_min_clusters, int(round(self.ligand_pool_ratio * Lb)))
            kb = min(kb, C)
            topk = torch.topk(mass[b], k=kb, largest=True).indices
            tokens_list.append(Xp[b, topk, :])  # (k_b, D)

        if dbg:
            kept = [t.size(0) for t in tokens_list]
            print(f"[Lig-{self.ligand_pool_type}] kept clusters per graph: {kept}", flush=True)

        return tokens_list

    # --------------- forward ----------------
    def forward(self, protein_sequences, ligand_batch, tree_batch=None,
                cluster_level_for_readout: int | None = None,
                return_cluster_levels=False, return_attn_weights=False,
                return_protein_pool_info: bool = False,
                protein_graphs=None):

        self._aux_losses.clear()
        dbg = getattr(self, "debug", False)

        # [1] Protein ESMC 임베딩
        # (옵션) ESM가 완전 동결이면 메모리 절약
        requires_esm_grad = any(p.requires_grad for p in self.esm.parameters())

        # --- NEW: batched ESMC inference ---
        seq0 = protein_sequences[0]
        if isinstance(seq0, str) and all(isinstance(s, str) for s in protein_sequences):
            seq_list = list(protein_sequences)
        elif isinstance(seq0, list) and all(isinstance(s, list) for s in protein_sequences):
            j = ''.join
            seq_list = [j(s) for s in protein_sequences]
        else:
            j = ''.join
            seq_list = [s if isinstance(s, str) else j(s) for s in protein_sequences]

        tokenizer = getattr(self.esm, "tokenizer", None)
        assert tokenizer is not None, "ESMC tokenizer not found. Attach tokenizer or implement encode_tokens."

        with torch.set_grad_enabled(requires_esm_grad):
            seq_feats = esmc_encode_batched(
                esmc_model=self.esm,
                seqs=seq_list,
                tokenizer=tokenizer,
                max_len_cap=1024,
                max_tokens_per_batch=65536,
                device=next(self.parameters()).device,
                requires_grad=requires_esm_grad,  # ★ 추가
                use_autocast=requires_esm_grad,  # ★ 선택: 미세조정 시 혼합정밀
            )
        # seq_feats: List[Tensor(L_i, 960)]
        # 이후 동일하게 projection 적용
        seq_feats = [self.protein_proj(x) for x in seq_feats]  # list of (L, D)

        B = len(seq_feats)

        # [2] (선택) 구조 반영 + 풀링
        use_struct = (protein_graphs is not None) and self.use_protein_graph_structure
        if use_struct:
            protein_graphs = protein_graphs.to(seq_feats[0].device)

            lengths_esmc = [t.size(0) for t in seq_feats]
            x_cat = torch.cat(seq_feats, dim=0)  # (N_total, D)
            _validate_pg(x_cat, protein_graphs)

            lengths_pg = torch.bincount(protein_graphs.batch, minlength=B).tolist()
            if lengths_esmc == lengths_pg:
                for i in range(self.protein_mp_layers):
                    nei = self._edge_aggregate(
                        x_cat, protein_graphs.edge_index, getattr(protein_graphs, "edge_weight", None)
                    )
                    h = self.protein_mp_self[i](x_cat) + self.protein_mp_nei[i](nei)
                    x_cat = self.protein_mp_ln[i](F.gelu(h))
                    x_cat = self._chk(self.protein_mp_drop(x_cat), f"protein_mp_block{i}")

            # ✅ 여기서만 protein 풀링 호출 (중복 호출 제거)
            prot_tokens, protein_pool_info = self._protein_pool(
                x_cat, protein_graphs, seq_feats, dbg=dbg
            )
        else:
            # 그래프 구조 미사용: 원 토큰 그대로
            prot_tokens = seq_feats
            protein_pool_info = [dict(residue_idx=None, score=None) for _ in range(B)]

        # [3] Cross-Attention 입력 준비
        protein_embed = pad_sequence(prot_tokens, batch_first=True)  # (B, Lp, D)
        protein_embed = self._chk(protein_embed, "pad/protein_embed")
        protein_embed = self._chk(self.pre_norm_prot(protein_embed), "pre_norm_prot")
        # (선택) 평균 pooled (현재 사용 안 하면 남겨두어도 무해)
        pooled_prot = torch.stack(
            [t.mean(dim=0) if t.numel() > 0 else torch.zeros(protein_embed.size(-1), device=protein_embed.device)
             for t in prot_tokens], dim=0
        )

        # [4] Ligand 인코딩
        graph_feat, node_feat = self.gnn(ligand_batch, return_node_embeddings=True)
        graph_feat = self._chk(self.ligand_proj(graph_feat), "lig_graph_proj")
        node_feat = self._chk(self.ligand_proj(node_feat), "lig_node_proj")
        node_feat = self._chk(self.pre_norm_lig(node_feat), "pre_norm_lig")

        # [5] Ligand Pooling (SEP/DiffPool/MinCut/none)
        node_batch = ligand_batch.batch
        level_feats = None
        pooled_level_feat = None
        has_sep = False

        if (self.ligand_pool_type == "sep") and (tree_batch is not None):
            x_level = node_feat
            level_feats = [x_level]
            for lvl in range(1, self.tree_depth):
                ei = self._build_sep_edge_index(tree_batch, level=lvl, device=node_feat.device)
                size = self._sep_size(tree_batch, level=lvl)
                x_level = F.relu(self.sepools[lvl - 1](x_level, ei, size=size))
                level_feats.append(x_level)

            if cluster_level_for_readout is not None:
                l = int(cluster_level_for_readout)
                sizes_l = [int(g['node_size'][l]) for g in tree_batch]
                chunks = torch.split(level_feats[l], sizes_l, dim=0)
                pooled = [(c.mean(dim=0, keepdim=True) if c.size(0) > 0
                           else torch.zeros(1, level_feats[l].size(1), device=level_feats[l].device))
                          for c in chunks]
                pooled_level_feat = torch.cat(pooled, dim=0)  # (B, D')
                has_sep = True

            lig_tokens_list = [node_feat[node_batch == i] for i in range(B)]
            lig_tokens = pad_sequence(lig_tokens_list, batch_first=True)  # (B, Ln, D)
            loss_cons, loss_ortho = self._aux_sep_losses(level_feats, tree_batch, device=node_feat.device)
            self._aux_losses["lig_sep_cons"] = loss_cons
            self._aux_losses["lig_sep_ortho"] = loss_ortho

            if has_sep:
                kv_vec = self.sepool_kv_proj(
                    pooled_level_feat) if self.sepool_kv_proj is not None else pooled_level_feat
                kv_vec = self._chk(kv_vec, "sepool_kv_proj")
                sep_token = kv_vec.unsqueeze(1)  # (B,1,D)
                lig_tokens_aug = torch.cat([lig_tokens, sep_token], dim=1)
            else:
                lig_tokens_aug = lig_tokens

        elif self.ligand_pool_type in ("diffpool", "mincut"):
            lig_tokens_list = self._ligand_pool_tokens(node_feat, ligand_batch, dbg=dbg)  # list[(k_b, D)]
            lig_tokens_aug = pad_sequence(lig_tokens_list, batch_first=True)  # (B, k_max, D)
        else:
            lig_tokens_list = [node_feat[node_batch == i] for i in range(B)]
            lig_tokens_aug = pad_sequence(lig_tokens_list, batch_first=True)

        lig_tokens_aug = self._chk(lig_tokens_aug, "lig_tokens_aug")

        # [6] Cross-Attention (protein ← ligand, ligand ← protein)
        if return_attn_weights:
            prot_msg, attn_pl = self.cross_attn_pl(protein_embed, lig_tokens_aug, return_attn_weights=True)
        else:
            prot_msg = self.cross_attn_pl(protein_embed, lig_tokens_aug)
            attn_pl = None

        # ✅ FFN 중복 제거: 한 번만 적용
        prot_upd = self.prot_ln1(protein_embed + self.ca_drop(prot_msg))
        prot_upd = self._chk(self.prot_ln2(prot_upd + self.prot_ffn(prot_upd)), "prot_upd")

        if return_attn_weights:
            lig_msg, attn_lp = self.cross_attn_lp(lig_tokens_aug, prot_upd, return_attn_weights=True)
        else:
            lig_msg = self.cross_attn_lp(lig_tokens_aug, prot_upd)
            attn_lp = None

        lig_upd = self.lig_ln1(lig_tokens_aug + self.ca_drop(lig_msg))
        lig_upd = self._chk(self.lig_ln2(lig_upd + self.lig_ffn(lig_upd)), "lig_upd")  # ✅ 중복 제거

        # [7] Pooling & Fusion
        # prot_upd = self._chk(prot_upd.mean(dim=1), "prot_upd")

        if has_sep:
            sep_token_upd = self._chk(lig_upd[:, -1, :], "sep_token_upd")
            pooled_lig = self._chk(lig_upd[:, :-1, :].mean(dim=1), "pooled_lig")
        else:
            sep_token_upd = None
            pooled_lig = self._chk(lig_upd.mean(dim=1), "pooled_lig")

        graph_side = self._chk(0.5 * graph_feat + 0.5 * pooled_lig, "graph_side_mix1")
        if has_sep:
            graph_side = self._chk(0.5 * graph_side + 0.5 * sep_token_upd, "graph_side_mix2")

        lengths_prot = torch.tensor([t.size(0) for t in prot_tokens], device=protein_embed.device)
        prot_upd = masked_mean(prot_upd, lengths_prot) ##14
        fused = torch.cat([prot_upd, graph_side], dim=-1)
        fused = self._chk(fused, "fused_in")
        out = self._chk(self.regressor(fused), "regressor_out")

        # [8] 반환 구성
        if return_attn_weights and return_cluster_levels:
            result = (out, (attn_pl, attn_lp), level_feats)
        elif return_attn_weights:
            result = (out, (attn_pl, attn_lp))
        elif return_cluster_levels:
            result = (out, level_feats)
        else:
            result = out

        if return_protein_pool_info:
            if isinstance(result, tuple):
                return (*result, protein_pool_info)
            else:
                return result, protein_pool_info

        return result


#######################################################################################################################
def bucketize_indices_by_length(lengths, max_tokens_per_batch=65536, sort_desc=True):
    """
    lengths: List[int]  (BOS/EOS 포함 길이)
    max_tokens_per_batch: 토큰 예산(길이합 상한). GPU 메모리 한계 따라 조절.

    Returns: List[List[int]]  (각 배치에 들어갈 샘플 인덱스 리스트)
    """
    idxs = list(range(len(lengths)))
    idxs.sort(key=lambda i: lengths[i], reverse=sort_desc)
    batches = []
    cur = []
    cur_tokens = 0
    for i in idxs:
        L = lengths[i]
        if cur and (cur_tokens + L > max_tokens_per_batch):
            batches.append(cur)
            cur, cur_tokens = [], 0
        cur.append(i)
        cur_tokens += L
    if cur:
        batches.append(cur)
    return batches


def esmc_encode_batched(esmc_model, seqs, tokenizer,
                        max_len_cap=1024, max_tokens_per_batch=65536,
                        device="cuda",
                        requires_grad=False,
                        use_autocast=False):
    """
    seqs: List[str]  (B개 시퀀스)
    tokenizer: ESMC 토크나이저
    return_embeddings=True이면 sequence-wise 임베딩 리턴.

    Returns:
      per_seq_embeddings: List[Tensor[L_i-2, D]]  (BOS/EOS 제거된 형태로 반환)
    """
    # 1) 우선 한 번 토크나이즈/패딩해서 길이 정보만 얻자(버킷을 위해)
    seq_ids, _, _, true_lens = build_sequence_tokens(seqs, tokenizer, max_len_cap=max_len_cap)

    batches = bucketize_indices_by_length(true_lens, max_tokens_per_batch=max_tokens_per_batch)
    per_seq_embeddings = [None] * len(seqs)

    for batch_idxs in batches:
        # 배치 시퀀스 뽑기
        batch_seqs = [seqs[i] for i in batch_idxs]

        # 다시 해당 배치만 토크나이즈/패딩
        ids, attn_mask, struct, lens = build_sequence_tokens(batch_seqs, tokenizer, max_len_cap=max_len_cap)

        ids      = ids.to(device)
        attn_mask= attn_mask.to(device)
        struct   = struct.to(device)

        ctx_grad = torch.enable_grad() if requires_grad else torch.no_grad()
        with ctx_grad:
            if use_autocast:
                ac = autocast()
            else:
                # no-op context
                from contextlib import nullcontext
                ac = nullcontext()

            with ac:
                try:
                    proteins = [ESMProtein(sequence=s) for s in batch_seqs]
                    outputs = esmc_model.encode(proteins)
                    logits = esmc_model.logits(outputs, LogitsConfig(sequence=True, return_embeddings=True))
                    emb = logits.embeddings  # [B, L_max, D]
                except Exception:
                    # 폴백(개별 호출): 그래도 grad가 필요하면 grad 켠 상태로
                    emb_list = []
                    for s in batch_seqs:
                        pt = esmc_model.encode(ESMProtein(sequence=s))
                        lg = esmc_model.logits(pt, LogitsConfig(sequence=True, return_embeddings=True))
                        emb_list.append(lg.embeddings)  # [1, L_i, D]
                    # pad_sequence는 autograd-safe
                    from torch.nn.utils.rnn import pad_sequence
                    emb = pad_sequence([e.squeeze(0) for e in emb_list], batch_first=True)

        # BOS/EOS 제거 시에도 detach 금지
        for j, L in enumerate(lens):
            per_seq_embeddings[batch_idxs[j]] = emb[j, 1:L - 1, :].contiguous()  # ★ detach 없음

    return per_seq_embeddings

def masked_mean(x, valid_lengths):
    # x: (B, L, D), valid_lengths: (B,)
    B, L, D = x.size()
    mask = torch.arange(L, device=x.device).unsqueeze(0) < valid_lengths.unsqueeze(1)  # (B,L)
    mask = mask.unsqueeze(-1).float()  # (B,L,1)
    s = (x * mask).sum(dim=1)  # (B,D)
    denom = mask.sum(dim=1).clamp_min(1.0)  # (B,1)
    return s / denom

MODEL_MAX_TOKENS = 1024
SPECIAL_TOKENS   = 2   # BOS/EOS
MAX_RAW_SEQ_LEN  = MODEL_MAX_TOKENS - SPECIAL_TOKENS  # 1022
SAFETY_CAP       = 1000  # 실제 truncate는 1000 정도로
BOS_ID = 4098
EOS_ID = 4097
PAD_ID = 0            # <- ESMC 토크나이저의 실제 pad id로 교체
MASK_ID = 4096        # 구조 트랙 마스킹 id (네 예시 값)


def build_sequence_tokens(seqs, tokenizer, max_len_cap=1024):
    """
    seqs: List[str]  (원본 아미노산 서열; SPECIAL 토큰 제외된 길이 L_i)
    tokenizer: ESMC 토크나이저(또는 model.esm.tokenizer). 반드시 BOS/EOS를 붙일 수 있어야 함.
    max_len_cap: 배치 내 패딩 길이 상한(최대 버킷 길이). 모델 하드 한계 포함.

    Returns:
      seq_ids_padded: LongTensor [B, L_max]  (BOS/EOS 포함)
      attn_mask     : BoolTensor [B, L_max]  (True=유효토큰; PAD=False)
      structure_pad : LongTensor [B, L_max]  (예시처럼 구조트랙 마스킹 및 일부 인덱스 채움)
      true_lens     : List[int] (BOS/EOS 포함한 각 샘플 실길이)
    """
    seq_ids = []
    structure = []
    true_lens = []

    for s in seqs:
        # 1) 길이 제한: BOS/EOS 포함 여유를 둔다. (모델 한계가 1024라면, 원시 서열은 최대 1022가 안전)
        #    안전하게 원시 서열을 1000~1022 선으로 truncate 하는 것을 권고.
        s_trim = s[: max_len_cap - 2]  # BOS/EOS 2토큰 고려

        # 2) 토크나이즈(+스페셜)
        #    아래는 예시: tokenizer.encode(s_trim, add_special_tokens=True) 같은 API를 가정
        #    실제 ESMC 토크나이저 API에 맞춰 교체 필요.
        ids = tokenizer.encode(s_trim, add_special_tokens=True)  # [BOS] ... [EOS]
        ids = torch.tensor(ids, dtype=torch.long)

        # 3) 구조 트랙 (네 예시를 배치 형태로 일반화)
        st = torch.full_like(ids, MASK_ID)
        st[0] = BOS_ID
        st[-1] = EOS_ID

        # --- 템플릿을 써서 일부 residue 위치 구조토큰을 채우고 싶다면 여기서 지정 ---
        # 예) GFP 템플릿(인덱스 주의: BOS 포함 인덱스 기준) 참고 카피
        # if template is not None:
        #     st[55:70] = template.structure[56:71]
        #     st[93]    = template.structure[93]
        #     st[219]   = template.structure[219]

        seq_ids.append(ids)
        structure.append(st)
        true_lens.append(int(ids.size(0)))

    # 4) 패딩 (좌/우 패딩 모두 가능하나, 여기선 우측 PAD)
    seq_ids_padded = pad_sequence(seq_ids, batch_first=True, padding_value=PAD_ID)          # [B, L_max]
    structure_pad  = pad_sequence(structure, batch_first=True, padding_value=MASK_ID)       # [B, L_max]

    # 5) 어텐션 마스크 (PAD=False)
    attn_mask = (seq_ids_padded != PAD_ID)                                                  # [B, L_max]

    return seq_ids_padded, attn_mask, structure_pad, true_lens