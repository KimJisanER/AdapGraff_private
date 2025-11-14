import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Linear, Sequential

from torch_geometric.nn import (
    GATConv, GCNConv, GINConv, MessagePassing, global_add_pool
)
from torch_geometric.utils import add_self_loops

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

