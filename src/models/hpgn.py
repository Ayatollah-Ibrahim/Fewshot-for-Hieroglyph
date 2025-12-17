"""
Hierarchical Prototype Graph Network (HPGN) implementation.

HPGN combines multi-scale CNN encoding, patch-level prototype extraction,
and graph neural network refinement for few-shot learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from src.models.encoders import ResNetEncoder


class SimplifiedPatchPrototype(nn.Module):
    """
    Patch-level prototype extraction module.
    
    Uses attention mechanism to extract multiple prototypical patches
    from local feature maps.
    """
    
    def __init__(self, local_channels=64, patch_dim=64, num_prototypes=2):
        """
        Initialize patch prototype module.
        
        Args:
            local_channels: Channels in input feature map
            patch_dim: Dimension of patch embeddings
            num_prototypes: Number of prototypes to extract per image
        """
        super().__init__()
        self.num_prototypes = num_prototypes
        self.patch_dim = patch_dim

        # Project patches to embedding space
        self.patch_proj = nn.Sequential(
            nn.Linear(local_channels, patch_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(patch_dim, patch_dim)
        )

        # Learnable prototype queries
        self.proto_queries = nn.Parameter(torch.randn(num_prototypes, patch_dim))
        
        # Multi-head attention for prototype extraction
        self.attn = nn.MultiheadAttention(
            patch_dim, 
            num_heads=2, 
            batch_first=True, 
            dropout=0.1
        )

    def forward(self, local_feat):
        """
        Extract patch prototypes from local features.
        
        Args:
            local_feat: Local feature maps [B, C, H, W]
            
        Returns:
            prototypes: Extracted prototypes [B, num_prototypes, patch_dim]
        """
        B, C, H, W = local_feat.shape
        
        # Flatten spatial dimensions to get patches
        patches = local_feat.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        
        # Project patches to embedding space
        patch_emb = self.patch_proj(patches)  # [B, H*W, patch_dim]
        
        # Expand queries for batch
        queries = self.proto_queries.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]
        
        # Extract prototypes via attention
        prototypes, _ = self.attn(queries, patch_emb, patch_emb)  # [B, K, D]
        
        return prototypes


class SimplifiedPrototypeGNN(nn.Module):
    """
    Graph neural network for prototype refinement.
    
    Refines prototypes by aggregating information from similar prototypes
    using graph attention networks.
    """
    
    def __init__(self, proto_dim=64, hidden_dim=64):
        """
        Initialize prototype GNN.
        
        Args:
            proto_dim: Dimension of input prototypes
            hidden_dim: Dimension of hidden representations
        """
        super().__init__()

        # Graph attention layer
        self.gat = GATConv(
            proto_dim, 
            hidden_dim, 
            heads=2, 
            concat=False, 
            dropout=0.1
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def build_graph_fast(self, prototypes, labels, k_neighbors=2):
        """
        Build k-NN graph connecting similar prototypes.
        
        Args:
            prototypes: Prototype embeddings [N, D]
            labels: Prototype labels [N]
            k_neighbors: Number of nearest neighbors to connect
            
        Returns:
            edge_index: Graph edges [2, E]
        """
        N = prototypes.shape[0]
        k = min(k_neighbors + 1, N)  # +1 to include self

        # Compute pairwise distances
        dist = torch.cdist(prototypes, prototypes)
        
        # Find k nearest neighbors
        _, indices = torch.topk(dist, k=k, largest=False, dim=1)

        # Create edge list
        src = torch.arange(N, device=prototypes.device).unsqueeze(1).expand(-1, k)
        dst = indices

        src_flat = src.reshape(-1)
        dst_flat = dst.reshape(-1)
        
        # Remove self-loops
        mask = src_flat != dst_flat

        edge_index = torch.stack([src_flat[mask], dst_flat[mask]], dim=0)
        
        return edge_index

    def forward(self, prototypes, labels):
        """
        Refine prototypes using graph neural network.
        
        Args:
            prototypes: Input prototypes [N, D]
            labels: Prototype labels [N]
            
        Returns:
            refined: Refined prototypes [N, D]
        """
        # Build graph
        edge_index = self.build_graph_fast(prototypes, labels, k_neighbors=2)
        
        # Apply GAT layer
        x = prototypes
        x_new = self.gat(x, edge_index)
        x_new = self.layer_norm(x_new)
        x_new = F.relu(x_new)
        x_new = self.dropout(x_new)
        
        # Residual connection
        x = x + x_new
        
        return x


class HPGN_Small(nn.Module):
    """
    Hierarchical Prototype Graph Network for Few-Shot Learning.
    
    Simplified version designed for small datasets with strong regularization
    to prevent overfitting.
    """
    
    def __init__(self, global_dim=128, local_channels=64, patch_dim=64, num_prototypes=2):
        """
        Initialize HPGN model.
        
        Args:
            global_dim: Dimension of global embeddings
            local_channels: Channels in local feature maps (compatibility param)
            patch_dim: Dimension of patch prototypes
            num_prototypes: Number of prototypes per image
        """
        super().__init__()
        self.num_prototypes = num_prototypes
        self.patch_dim = patch_dim

        # Feature encoder (ResNet18-based)
        self.encoder = ResNetEncoder(
            global_dim=global_dim,
            local_channels=local_channels
        )

        # Patch prototype extraction
        # Note: ResNet encoder outputs 512 channels
        self.patch_module = SimplifiedPatchPrototype(
            local_channels=512,  # ResNet18 output dimension
            patch_dim=patch_dim,
            num_prototypes=num_prototypes
        )

        # Graph-based prototype refinement
        self.gnn = SimplifiedPrototypeGNN(
            proto_dim=patch_dim,
            hidden_dim=patch_dim
        )

    def extract_features(self, x):
        """
        Extract global and patch-level features.
        
        Args:
            x: Input images [B, 1, H, W]
            
        Returns:
            global_emb: Global embeddings [B, global_dim]
            patch_prototypes: Patch prototypes [B, num_prototypes, patch_dim]
        """
        global_emb, local_feat = self.encoder(x)
        patch_prototypes = self.patch_module(local_feat)
        return global_emb, patch_prototypes

    def forward(self, support_x, support_y, query_x):
        """
        Forward pass for episode-based training.
        
        Args:
            support_x: Support images [N*K, 1, H, W]
            support_y: Support labels [N*K] (values 0 to N-1)
            query_x: Query images [Q, 1, H, W]
            
        Returns:
            logits: Classification logits [Q, N]
        """
        n_way = torch.unique(support_y).numel()

        # Extract support prototypes
        _, support_prototypes = self.extract_features(support_x)
        B_support, K_proto, D = support_prototypes.shape

        # Flatten prototypes across support samples
        support_proto_flat = support_prototypes.reshape(-1, D)
        support_labels_expanded = support_y.unsqueeze(1).expand(-1, K_proto).contiguous().reshape(-1)

        # Refine prototypes using GNN
        refined_support_proto = self.gnn(support_proto_flat, support_labels_expanded)

        # Compute class prototypes (mean of refined prototypes per class)
        class_prototypes = []
        for c in range(n_way):
            class_mask = support_labels_expanded == c
            class_proto = refined_support_proto[class_mask].mean(dim=0)
            class_prototypes.append(class_proto)
        class_prototypes = torch.stack(class_prototypes)  # [N, D]

        # Extract query features
        _, query_prototypes = self.extract_features(query_x)
        query_emb = query_prototypes.mean(dim=1)  # [Q, D]

        # Compute distances to class prototypes
        dists = torch.cdist(query_emb, class_prototypes)  # [Q, N]
        
        # Convert distances to logits (negative distances)
        logits = -dists

        return logits
