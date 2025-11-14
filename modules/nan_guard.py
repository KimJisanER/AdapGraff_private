import torch
import torch.nn as nn

from torch_geometric.nn import SAGPooling
from torch_geometric.utils import coalesce, to_undirected

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
    g.edge_index, _ = coalesce(to_undirected(g.edge_index), None, x_cat.size(0), x_cat.size(0))

    # edge_weight 체크
    if hasattr(g, "edge_weight") and g.edge_weight is not None:
        assert g.edge_weight.device == x_cat.device, "edge_weight device mismatch"
        assert g.edge_weight.numel() == g.edge_index.size(1), "edge_weight length != E"
        assert torch.isfinite(g.edge_weight).all(), "edge_weight has NaN/Inf"
