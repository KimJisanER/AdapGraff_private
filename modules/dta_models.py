import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn.utils.rnn import pad_sequence

from esm.models.esmc import ESMC

from torch_geometric.data import Batch  # forward에서 그래프 재배치를 위해 필요할 수 있음
from torch_geometric.nn import SAGPooling
from torch_geometric.nn.dense import dense_diff_pool, dense_mincut_pool
from torch_geometric.utils import subgraph, to_dense_adj, to_dense_batch

from .dualgraph.gnn import GNN
from modules.layers import LigandEncoder, CrossAttention
from modules.nan_guard import _finite_or_handle, _validate_pg
from modules.esmc_embedding import esmc_encode_batched, masked_mean


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
        max_len_cap=1024,
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
        # LigandEncoder
        ligand_gnn_type="dualgraph",
        ligand_in_dim=None,
        ligand_hidden_dim=128,
        ligand_num_layers=6,
        # --- Protein structure 사용 옵션 ---
        use_protein_graph_structure: bool = True,
        protein_mp_layers: int = 2,
        # --- NEW: SAGPooling 옵션 ---
        protein_pool_type: str = "sag",     # "sag"|"mincut"|"diffpool"|"none"
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
        )

        self.pool_loss_weights.update({
            "prot_sag_smooth": 1.0,
            "prot_sag_ent": 1e-3,
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
                som_mode=False,
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
        elif self.protein_pool_type in ("diffpool", "mincut"):
            # 고정 클러스터 채널 수(배치 간 동일해야 합니다)
            self.protein_max_clusters = self.protein_max_clusters or 64
            self.protein_assign = nn.Linear(fusion_dim, self.protein_max_clusters)
        elif self.protein_pool_type == "none":
            self.protein_pool = None
        else:
            raise ValueError(f"Unknown protein_pool_type={self.protein_pool_type}")

        self.ligand_proj = nn.Linear(ligand_output_dim, fusion_dim)
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

        if self.protein_pool_type in ("sag"):
            # (선택) edge weight 변환
            if ew is not None:
                tau = 5.0
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
                max_len_cap=max_len_cap,
                max_tokens_per_batch=65536,
                device=next(self.parameters()).device,
                requires_grad=requires_esm_grad,  # ★ 추가
                use_autocast=requires_esm_grad,  # ★ 선택: 미세조정 시 혼합정밀
            )

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

        lig_tokens_list = [node_feat[node_batch == i] for i in range(B)]
        lig_tokens_aug = pad_sequence(lig_tokens_list, batch_first=True)

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

        pooled_lig = self._chk(lig_upd.mean(dim=1), "pooled_lig")
        graph_side = self._chk(0.5 * graph_feat + 0.5 * pooled_lig, "graph_side_mix1")

        lengths_prot = torch.tensor([t.size(0) for t in prot_tokens], device=protein_embed.device)
        prot_upd = masked_mean(prot_upd, lengths_prot) ##14
        fused = torch.cat([prot_upd, graph_side], dim=-1)
        out = self._chk(self.regressor(fused), "regressor_out")

        # [8] 반환 구성
        if return_attn_weights:
            result = (out, (attn_pl, attn_lp))
        else:
            result = out

        if return_protein_pool_info:
            if isinstance(result, tuple):
                return (*result, protein_pool_info)
            else:
                return result, protein_pool_info

        return result
