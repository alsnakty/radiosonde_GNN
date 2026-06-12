# -*- coding: utf-8 -*-
"""GNN model definitions for radiosonde imputation.

Single entry point: RadiosondeSpatioTemporalGNN(model_type=...). Each model_type
selects a triple of Vertical/Horizontal/Temporal MessagePassing layers combined in
SpatioTemporalGNNLayer; flat_graphsage is the exception (all three edge sets merged
into one homogeneous graph, a single GraphSAGE conv, no fusion).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
from torch_geometric.utils import softmax

# MS-GraphSAGE
class MultiScaleVerticalSAGE(MessagePassing):
    def __init__(self, in_channels, out_channels):
        # Otomatik agregasyonu kapatıyoruz (kendimiz yapacağız)
        super().__init__(aggr=None)
        
        # PNA Mantığı: Giriş boyutu 3 katına çıkıyor (Mean + Max + Min)
        self.lin_neigh = nn.Linear(in_channels * 3, out_channels)
        self.lin_self = nn.Linear(in_channels, out_channels)
        
        # Edge Gating Mantığı: Kenar özellikleri boyutu 1 (basınç farkı)
        self.lin_edge = nn.Linear(1, in_channels) 

    def forward(self, x, edge_index, edge_attr):
        # 1. Propagate: Mesajlaşma ve Agregasyon
        # Çıktı boyutu: [N, in_channels * 3]
        neigh_agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # 2. Update: Boyut indirgeme ve Self-Connection
        # [N, in_channels * 3] -> [N, out_channels]
        out = self.lin_neigh(neigh_agg) + self.lin_self(x)
        
        return F.relu(out)

    def message(self, x_j, edge_attr):
        # Edge özelliğini işle (Mesafe/Basınç farkı)
        edge_emb = self.lin_edge(edge_attr)
        # Komşudan gelen bilgiyi kapıdan geçir (Gating)
        # Not: Burada projeksiyon yapmıyoruz, ham veriyi modüle ediyoruz
        return x_j * torch.sigmoid(edge_emb)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        # inputs: [E, in_channels] (Modüle edilmiş komşu bilgileri)
        
        # 1. Mean (Ortalama)
        mean_agg = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='mean')
        # 2. Max (En güçlü sinyal - örn: fırtına)
        max_agg = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='max')
        # 3. Min (En düşük sinyal - örn: inversiyon)
        min_agg = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='min')
        
        # Hepsini birleştir: [N, in_channels * 3]
        return torch.cat([mean_agg, max_agg, min_agg], dim=-1)

class MultiScaleHorizontalSAGE(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr=None)
        self.lin_neigh = nn.Linear(in_channels * 3, out_channels)
        self.lin_self = nn.Linear(in_channels, out_channels)
        # Yatay kenar özellikleri boyutu 2 (Mesafe + Yön)
        self.lin_edge = nn.Linear(2, in_channels)

    def forward(self, x, edge_index, edge_attr):
        neigh_agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.lin_neigh(neigh_agg) + self.lin_self(x)
        return F.relu(out)

    def message(self, x_j, edge_attr):
        edge_emb = self.lin_edge(edge_attr)
        return x_j * torch.sigmoid(edge_emb)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        mean_agg = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='mean')
        max_agg = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='max')
        min_agg = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='min')
        return torch.cat([mean_agg, max_agg, min_agg], dim=-1)


class MultiScaleTemporalSAGE(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr=None)
        self.lin_neigh = nn.Linear(in_channels * 3, out_channels)
        self.lin_self = nn.Linear(in_channels, out_channels)
        # Zamansal kenar özellikleri boyutu 1 (Zaman farkı)
        self.lin_edge = nn.Linear(1, in_channels)

    def forward(self, x, edge_index, edge_attr):
        neigh_agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.lin_neigh(neigh_agg) + self.lin_self(x)
        return F.relu(out)

    def message(self, x_j, edge_attr):
        edge_emb = self.lin_edge(edge_attr)
        return x_j * torch.sigmoid(edge_emb)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        mean_agg = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='mean')
        max_agg = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='max')
        min_agg = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='min')
        return torch.cat([mean_agg, max_agg, min_agg], dim=-1)

# GraphSAGE    
class VerticalEdgeConvVHT(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        self.lin_neigh = nn.Linear(in_channels, out_channels)
        self.lin_self = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(1, out_channels)

    def forward(self, x, edge_index, edge_attr):
        neigh_agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.lin_self(x) + neigh_agg
        return F.relu(out)

    def message(self, x_j, edge_attr):
        edge_emb = self.lin_edge(edge_attr)
        return self.lin_neigh(x_j) * torch.sigmoid(edge_emb)

class HorizontalEdgeConvVHT(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        self.lin_neigh = nn.Linear(in_channels, out_channels)
        self.lin_self = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(2, out_channels)

    def forward(self, x, edge_index, edge_attr):
        neigh_agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.lin_self(x) + neigh_agg
        return F.relu(out)

    def message(self, x_j, edge_attr):
        edge_emb = self.lin_edge(edge_attr)
        return self.lin_neigh(x_j) * torch.sigmoid(edge_emb)
    
class TemporalEdgeConvVHT(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        self.lin_neigh = nn.Linear(in_channels, out_channels)
        self.lin_self = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(1, out_channels)

    def forward(self, x, edge_index, edge_attr):
        neigh_agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.lin_self(x) + neigh_agg
        return F.relu(out)

    def message(self, x_j, edge_attr):
        edge_emb = self.lin_edge(edge_attr)
        return self.lin_neigh(x_j) * torch.sigmoid(edge_emb)


# Ablation: VHT without sigmoid-gated edge conditioning.
# Same three edge types and same conv structure as VHT, but messages
# are not modulated by edge attributes -- pure neighbor projection.
class VerticalEdgeConvVHTNoGating(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        self.lin_neigh = nn.Linear(in_channels, out_channels)
        self.lin_self  = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        neigh_agg = self.propagate(edge_index, x=x)
        out = self.lin_self(x) + neigh_agg
        return F.relu(out)

    def message(self, x_j):
        return self.lin_neigh(x_j)


class HorizontalEdgeConvVHTNoGating(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        self.lin_neigh = nn.Linear(in_channels, out_channels)
        self.lin_self  = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        neigh_agg = self.propagate(edge_index, x=x)
        out = self.lin_self(x) + neigh_agg
        return F.relu(out)

    def message(self, x_j):
        return self.lin_neigh(x_j)


class TemporalEdgeConvVHTNoGating(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        self.lin_neigh = nn.Linear(in_channels, out_channels)
        self.lin_self  = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        neigh_agg = self.propagate(edge_index, x=x)
        out = self.lin_self(x) + neigh_agg
        return F.relu(out)

    def message(self, x_j):
        return self.lin_neigh(x_j)


# MPNN
class VerticalEdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(1, out_channels)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        edge_emb = self.lin_edge(edge_attr)
        return self.lin(x_j) * torch.sigmoid(edge_emb)

class HorizontalEdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(2, out_channels)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        edge_emb = self.lin_edge(edge_attr)
        return self.lin(x_j) * torch.sigmoid(edge_emb)

class TemporalEdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(1, out_channels)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        edge_emb = self.lin_edge(edge_attr)
        return self.lin(x_j) * torch.sigmoid(edge_emb)

# Vanilla GraphSAGE  
class VanillaVerticalSAGE(MessagePassing):
    """Orijinal GraphSAGE - edge attribute yok, gating yok"""
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        self.lin = nn.Linear(in_channels * 2, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        neigh_agg = self.propagate(edge_index, x=x)
        out = self.lin(torch.cat([x, neigh_agg], dim=-1))
        return F.relu(out)

    def message(self, x_j):
        return x_j


class VanillaHorizontalSAGE(MessagePassing):
    """Orijinal GraphSAGE - edge attribute yok, gating yok"""
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        self.lin = nn.Linear(in_channels * 2, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        neigh_agg = self.propagate(edge_index, x=x)
        out = self.lin(torch.cat([x, neigh_agg], dim=-1))
        return F.relu(out)

    def message(self, x_j):
        return x_j


class VanillaTemporalSAGE(MessagePassing):
    """Orijinal GraphSAGE - edge attribute yok, gating yok"""
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        self.lin = nn.Linear(in_channels * 2, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        neigh_agg = self.propagate(edge_index, x=x)
        out = self.lin(torch.cat([x, neigh_agg], dim=-1))
        return F.relu(out)

    def message(self, x_j):
        return x_j


class FlatGraphSAGEConv(MessagePassing):
    """Flat baseline conv: standard GraphSAGE over a single homogeneous graph
    (vertical/horizontal/temporal edges merged). No edge attributes, no
    per-relation distinction."""
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        self.lin = nn.Linear(in_channels * 2, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        neigh_agg = self.propagate(edge_index, x=x)
        out = self.lin(torch.cat([x, neigh_agg], dim=-1))
        return F.relu(out)

    def message(self, x_j):
        return x_j

# GAT
class VerticalEdgeConvGAT(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=4):
        super().__init__(aggr='mean')
        self.heads = heads
        self.head_dim = out_channels // heads
        
        self.W = nn.Linear(in_channels, out_channels, bias=False)
        self.att = nn.Linear(2 * self.head_dim + 1, 1, bias=False)  # +1 for edge_attr
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, x, edge_index, edge_attr):
        x = self.W(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr, index):
        # x_i, x_j: (E, out_channels)
        # Multi-head reshape
        x_i = x_i.view(-1, self.heads, self.head_dim)
        x_j = x_j.view(-1, self.heads, self.head_dim)
        
        # Attention score per head
        edge_attr_expanded = edge_attr.unsqueeze(1).expand(-1, self.heads, -1)
        att_input = torch.cat([x_i, x_j, edge_attr_expanded], dim=-1)
        alpha = self.leaky_relu(self.att(att_input))  # (E, heads, 1)
        alpha = softmax(alpha.squeeze(-1), index)  # (E, heads)
        
        # Weighted message
        out = x_j * alpha.unsqueeze(-1)  # (E, heads, head_dim)
        return out.view(-1, self.heads * self.head_dim)

class HorizontalEdgeConvGAT(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=4):
        super().__init__(aggr='mean')
        self.heads = heads
        self.head_dim = out_channels // heads
        
        self.W = nn.Linear(in_channels, out_channels, bias=False)
        self.att = nn.Linear(2 * self.head_dim + 2, 1, bias=False)  # +2 for edge_attr
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, x, edge_index, edge_attr):
        x = self.W(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr, index):
        x_i = x_i.view(-1, self.heads, self.head_dim)
        x_j = x_j.view(-1, self.heads, self.head_dim)
        
        edge_attr_expanded = edge_attr.unsqueeze(1).expand(-1, self.heads, -1)
        att_input = torch.cat([x_i, x_j, edge_attr_expanded], dim=-1)
        alpha = self.leaky_relu(self.att(att_input))
        alpha = softmax(alpha.squeeze(-1), index)
        
        out = x_j * alpha.unsqueeze(-1)
        return out.view(-1, self.heads * self.head_dim)

class TemporalEdgeConvGAT(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=4):
        super().__init__(aggr='mean')
        self.heads = heads
        self.head_dim = out_channels // heads
        
        self.W = nn.Linear(in_channels, out_channels, bias=False)
        self.att = nn.Linear(2 * self.head_dim + 1, 1, bias=False)  # +1 for edge_attr
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, x, edge_index, edge_attr):
        x = self.W(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr, index):
        x_i = x_i.view(-1, self.heads, self.head_dim)
        x_j = x_j.view(-1, self.heads, self.head_dim)
        
        edge_attr_expanded = edge_attr.unsqueeze(1).expand(-1, self.heads, -1)
        att_input = torch.cat([x_i, x_j, edge_attr_expanded], dim=-1)
        alpha = self.leaky_relu(self.att(att_input))
        alpha = softmax(alpha.squeeze(-1), index)
        
        out = x_j * alpha.unsqueeze(-1)
        return out.view(-1, self.heads * self.head_dim)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.pressure_encoder = nn.Linear(1, d_model // 4)
        self.lat_encoder = nn.Linear(1, d_model // 4)
        self.lon_encoder = nn.Linear(1, d_model // 4)
        self.time_encoder = nn.Linear(2, d_model // 4)

    def forward(self, pressure, lat, lon, time_features):
        return torch.cat([
            self.pressure_encoder(pressure),
            self.lat_encoder(lat),
            self.lon_encoder(lon),
            self.time_encoder(time_features)
        ], dim=-1)


class SpatioTemporalGNNLayer(nn.Module):
    def __init__(self, hidden_dim, model_type=None, heads=4):
        super().__init__()
        
        self.model_type = model_type

        if model_type == 'multiscale_graphsage':
            self.convs = nn.ModuleDict({
                'vertical': MultiScaleVerticalSAGE(hidden_dim, hidden_dim),
                'horizontal': MultiScaleHorizontalSAGE(hidden_dim, hidden_dim),
                'temporal': MultiScaleTemporalSAGE(hidden_dim, hidden_dim)
            })
            
        elif model_type in ('vht_gnn', 'vht_gnn_no_temporal', 'vht_gnn_fixed_fusion',
                             'vht_gnn_no_vertical', 'vht_gnn_no_horizontal'):
            # Same convs as vht_gnn; the ablation variants differ in the outer GNN
            # (no_temporal: skip temporal self-attention), the fusion step
            # (fixed_fusion: uniform 1/3 instead of learned alpha), or which edge
            # type is dropped in the forward (no_vertical / no_horizontal).
            self.convs = nn.ModuleDict({
                'vertical':   VerticalEdgeConvVHT(hidden_dim, hidden_dim),
                'horizontal': HorizontalEdgeConvVHT(hidden_dim, hidden_dim),
                'temporal':   TemporalEdgeConvVHT(hidden_dim, hidden_dim),
            })

        elif model_type == 'vht_gnn_no_gating':
            # VHT structure (three edge types, edge-type-specific convs)
            # but without sigmoid edge gating.
            self.convs = nn.ModuleDict({
                'vertical':   VerticalEdgeConvVHTNoGating(hidden_dim, hidden_dim),
                'horizontal': HorizontalEdgeConvVHTNoGating(hidden_dim, hidden_dim),
                'temporal':   TemporalEdgeConvVHTNoGating(hidden_dim, hidden_dim),
            })

        elif model_type == 'gat':
            self.convs = nn.ModuleDict({
                'vertical': VerticalEdgeConvGAT(hidden_dim, hidden_dim, heads=heads),
                'horizontal': HorizontalEdgeConvGAT(hidden_dim, hidden_dim, heads=heads),
                'temporal': TemporalEdgeConvGAT(hidden_dim, hidden_dim, heads=heads)
            })
            
        elif model_type == 'mpnn':
            self.convs = nn.ModuleDict({
                'vertical': VerticalEdgeConv(hidden_dim, hidden_dim),
                'horizontal': HorizontalEdgeConv(hidden_dim, hidden_dim),
                'temporal': TemporalEdgeConv(hidden_dim, hidden_dim)
            })
        elif model_type == 'vanilla_graphsage':
            self.convs = nn.ModuleDict({
                'vertical': VanillaVerticalSAGE(hidden_dim, hidden_dim),
                'horizontal': VanillaHorizontalSAGE(hidden_dim, hidden_dim),
                'temporal': VanillaTemporalSAGE(hidden_dim, hidden_dim)
            })

        elif model_type == 'flat_graphsage':
            # Flat baseline: a single conv over the merged (V+H+T) graph; the
            # forward concatenates the three edge sets and skips view fusion.
            self.convs = nn.ModuleDict({
                'flat': FlatGraphSAGEConv(hidden_dim, hidden_dim)
            })

        else:
            raise ValueError(
                f"Invalid model type: {model_type}. Options: vanilla_graphsage, "
                "flat_graphsage, multiscale_graphsage, vht_gnn, vht_gnn_no_temporal, "
                "vht_gnn_no_gating, vht_gnn_fixed_fusion, vht_gnn_no_vertical, "
                "vht_gnn_no_horizontal, gat, mpnn"
            )
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        # Learnable edge weights
        self.edge_weight_raw = nn.Parameter(torch.zeros(3))  

    def forward(self, x, edge_indices, edge_attrs):
        edge_types = ['vertical', 'horizontal', 'temporal']

        if self.model_type == 'flat_graphsage':
            # Merge the three relations into one homogeneous graph; a single
            # GraphSAGE conv, no per-relation fusion.
            edge_list = [edge_indices[t] for t in edge_types
                         if t in edge_indices and edge_indices[t].shape[1] > 0]
            if len(edge_list) > 0:
                merged_edge_index = torch.cat(edge_list, dim=1)
                aggregated = self.convs['flat'](x, merged_edge_index)
            else:
                aggregated = torch.zeros_like(x)
            return self.norm(x + self.dropout(aggregated))

        if self.model_type == 'vht_gnn_fixed_fusion':
            # Ablation: equal weights instead of learned softmax(alpha)
            edge_weights = torch.full((3,), 1.0 / 3.0, device=x.device, dtype=x.dtype)
        else:
            edge_weights = F.softmax(self.edge_weight_raw, dim=0)

        # Ablation: drop one spatial relation entirely (no renormalization,
        # matching how an absent edge type is skipped below).
        dropped_edge = None
        if self.model_type == 'vht_gnn_no_vertical':
            dropped_edge = 'vertical'
        elif self.model_type == 'vht_gnn_no_horizontal':
            dropped_edge = 'horizontal'

        aggregated = torch.zeros_like(x)
        for i, edge_type in enumerate(edge_types):
            if edge_type == dropped_edge:
                continue
            if edge_type in edge_indices and edge_indices[edge_type].shape[1] > 0:
                msg = self.convs[edge_type](x, edge_indices[edge_type], edge_attrs[edge_type])
                aggregated = aggregated + edge_weights[i] * msg

        return self.norm(x + self.dropout(aggregated))


class TemporalAttentionFixed(nn.Module):
    """Padding + Attention Mask ile çalışan Temporal Attention"""
    skip_count = 0
    success_count = 0
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    @classmethod
    def reset_counters(cls):
        """Sayaçları sıfırla"""
        cls.skip_count = 0
        cls.success_count = 0
    
    @classmethod
    def get_stats(cls):
        """İstatistikleri döndür"""
        total = cls.skip_count + cls.success_count
        if total == 0:
            return "Henüz temporal attention çağrısı yapılmadı."
        skip_pct = 100 * cls.skip_count / total
        return (f"Temporal Attention İstatistikleri:\n"
                f"  Toplam çağrı: {total:,}\n"
                f"  Başarılı: {cls.success_count:,} ({100-skip_pct:.1f}%)\n"
                f"  Atlanan: {cls.skip_count:,} ({skip_pct:.1f}%)")

    def forward(self, x, node_metadata):
        """Temporal attention over W time-steps, per (batch-item, spatial-node).
        Input x: (B*W*S, h) where B = batch_size, W = window_size, S = nodes_per_step.
        Layout: item-major (B items concatenated), each item time-major (S nodes
        for t=0, then S for t=1, ..., S for t=W-1).
        Doing view(W, ?, h) directly would mix items together at the same
        time-index; correct shape is (B, W, S, h) -> permute -> attention over W."""
        W = len(node_metadata.get('time_steps', []))
        B = int(node_metadata.get('batch_size', 1))
        total_nodes = x.size(0)

        # Layout sanity: total_nodes must factor as B * W * S exactly.
        if W == 0 or B == 0 or (total_nodes % (B * W)) != 0:
            TemporalAttentionFixed.skip_count += 1
            return torch.zeros_like(x)

        S = total_nodes // (B * W)
        TemporalAttentionFixed.success_count += 1

        # (B*W*S, h) -> (B, W, S, h) -> (B, S, W, h) so attention runs over W.
        x4d = x.view(B, W, S, -1).permute(0, 2, 1, 3)

        Q = self.query(x4d)
        K = self.key(x4d)
        V = self.value(x4d)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale     # (B, S, W, W)
        attn   = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)                                # (B, S, W, h)

        output = self.out_proj(context)
        # (B, S, W, h) -> (B, W, S, h) -> (B*W*S, h)
        output = output.permute(0, 2, 1, 3).contiguous().view(total_nodes, -1)
        return output


class RadiosondeSpatioTemporalGNN(nn.Module):    
    def __init__(
            self,
            input_dim=6,
            hidden_dim=64,
            num_gnn_layers=1,
            dropout=0.1,
            model_type='multiscale_graphsage',
            **kwargs  
        ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.model_type = model_type 

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim)
        self.feat_combine = nn.Linear(hidden_dim * 2, hidden_dim)

        self.gnn_layers = nn.ModuleList([
            SpatioTemporalGNNLayer(
                hidden_dim, 
                model_type=model_type,  
                **kwargs
            ) 
            for _ in range(num_gnn_layers)
        ])

        self.temporal_attention = TemporalAttentionFixed(hidden_dim)

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_info, edge_indices, edge_attrs, node_metadata=None, mask=None):
        device = x.device

        if torch.isnan(x).all():
            return torch.zeros_like(x), torch.zeros(x.size(0), self.hidden_dim, device=device)

        x_filled = torch.nan_to_num(x, nan=0.0)
        h = self.input_proj(x_filled)

        # CUDA 
        pos_info_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in pos_info.items()}
        edge_indices_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in edge_indices.items()}
        edge_attrs_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in edge_attrs.items()}

        pos_enc = self.pos_encoding(
            pos_info_device['pressure'],
            pos_info_device['lat'],
            pos_info_device['lon'],
            pos_info_device['time']
        )
        h = self.feat_combine(torch.cat([h, pos_enc], dim=-1))
        h = self.dropout(h)

        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h, edge_indices_device, edge_attrs_device)

        if (node_metadata is not None
                and 'time_steps' in node_metadata
                and self.model_type != 'vht_gnn_no_temporal'):
            h_temporal = self.temporal_attention(h, node_metadata)
            h = h + h_temporal

        imputed = self.output_proj(h)

        if mask is not None:
            imputed = torch.where(mask.bool(), imputed, x)

        return imputed, h

    def get_num_parameters(self):
        """Toplam parametre sayısını döndürür."""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_parameters(self):
        """Eğitilebilir parametre sayısını döndürür."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_edge_weights(self):
        """Her layer'ın öğrendiği edge ağırlıklarını döndürür."""
        weights_dict = {}
        for i, layer in enumerate(self.gnn_layers):
            raw_weights = layer.edge_weight_raw.detach().cpu()
            softmax_weights = F.softmax(raw_weights, dim=0).numpy()
            weights_dict[f'layer_{i+1}'] = {
                'vertical': softmax_weights[0],
                'horizontal': softmax_weights[1],
                'temporal': softmax_weights[2]
            }
        return weights_dict

    def print_edge_weights(self):
        """Edge ağırlıklarını güzel formatta yazdırır."""
        weights = self.get_edge_weights()
        print("\n" + "="*60)
        print("LEARNABLE EDGE WEIGHTS")
        print("="*60)
        print(f"{'Layer':<10} | {'Vertical':>10} | {'Horizontal':>10} | {'Temporal':>10}")
        print("-"*60)
        for layer_name, w in weights.items():
            print(f"{layer_name:<10} | {w['vertical']:>10.2%} | {w['horizontal']:>10.2%} | {w['temporal']:>10.2%}")
        print("="*60)

# BACKWARD COMPATIBILITY - ESKİ İSİMLER
# Eski kodlar TemporalAttention adını kullanıyorsa
TemporalAttention = TemporalAttentionFixed 