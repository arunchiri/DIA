# model.py
import torch
from torch import nn
from dataclasses import dataclass, field
from typing import Dict, List
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, dim, expansion, dropout):
        super().__init__()
        hidden = int(dim * expansion)
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(dim, hidden)
        self.w3 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.w3(self.drop(F.silu(self.w1(x)) * self.w2(x)))

class HRMBlock(nn.Module):
    def __init__(self, dim, expansion, dropout):
        super().__init__()
        self.mlp = SwiGLU(dim, expansion, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.mlp(x))

class HRMReasoning(nn.Module):
    def __init__(self, dim, layers, expansion, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [HRMBlock(dim, expansion, dropout) for _ in range(layers)]
        )

    def forward(self, z, inj):
        z = z + inj
        for l in self.layers:
            z = l(z)
        return z

@dataclass
class TabularHRMConfig:
    numeric_dim: int
    binary_dim: int
    cat_vocab_sizes: List[int]
    cat_emb_dims: List[int]
    hidden_size: int = 128
    expansion: float = 2.0
    dropout: float = 0.05
    H_layers: int = 2
    L_layers: int = 2
    H_cycles: int = 2
    L_cycles: int = 2
    output_heads: Dict[str, int] = field(default_factory=lambda: {"y": 1})

class TabularHRM(nn.Module):
    def __init__(self, cfg: TabularHRMConfig):
        super().__init__()
        self.cfg = cfg

        # Create embedding layers for categorical features
        self.cat_embs = nn.ModuleList(
            [nn.Embedding(v, d) for v, d in zip(cfg.cat_vocab_sizes, cfg.cat_emb_dims)]
        )
        
        # Calculate input dimension
        in_dim = cfg.numeric_dim + cfg.binary_dim + sum(cfg.cat_emb_dims)
        self.input_proj = nn.Linear(in_dim, cfg.hidden_size)

        # HRM reasoning modules
        self.H = HRMReasoning(cfg.hidden_size, cfg.H_layers, cfg.expansion, cfg.dropout)
        self.L = HRMReasoning(cfg.hidden_size, cfg.L_layers, cfg.expansion, cfg.dropout)

        # Learnable initial states
        self.H_init = nn.Parameter(torch.zeros(cfg.hidden_size))
        self.L_init = nn.Parameter(torch.zeros(cfg.hidden_size))

        # Output heads
        self.heads = nn.ModuleDict(
            {k: nn.Linear(cfg.hidden_size, v) for k, v in cfg.output_heads.items()}
        )

    def forward(self, batch):
        # Process categorical features
        if self.cat_embs and len(batch["cat"]) > 0:
            cats = torch.cat(
                [emb(batch["cat"][i]) for i, emb in enumerate(self.cat_embs)], dim=1
            )
        else:
            cats = torch.zeros(batch["num"].size(0), 0, device=batch["num"].device)

        # Concatenate all features
        x = torch.cat([batch["num"], batch["bin"], cats], dim=1)
        x = self.input_proj(x)

        # Initialize states
        zH = self.H_init.unsqueeze(0).expand(x.size(0), -1)
        zL = self.L_init.unsqueeze(0).expand(x.size(0), -1)

        # HRM cycles
        for _ in range(self.cfg.H_cycles):
            for _ in range(self.cfg.L_cycles):
                zL = self.L(zL, zH + x)
            zH = self.H(zH, zL)

        # Apply output heads
        return {k: head(zH) for k, head in self.heads.items()}