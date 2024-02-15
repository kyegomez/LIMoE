import torch 
from torch import nn, Tensor, einsum


class LiMoE(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_experts: int,
        dim_head: int,
        dropout: float,
        *args,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_experts = num_experts
        self.dim_head = dim_head
        self.dropout = dropout
        self.heads = self.dim // self.dim_head
        self.scale = self.dim_head ** -0.5
        
        