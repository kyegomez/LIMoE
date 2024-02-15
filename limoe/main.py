import torch 
from torch import nn, Tensor, einsum
from zeta.nn import MoERouter, MixtureOfExperts, FeedForward, PreNorm, Attention


class DenseEncoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        num_experts: int,
        dim_head: int,
        dropout: int,
        ff_mult: int,
        *args,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_experts = num_experts
        self.dim_head = dim_head
        self.dropout = dropout
        self.ff_mult = ff_mult
        
        self.heads = self.dim // self.dim_head
        self.scale = self.dim_head ** -0.5
        
        gpu = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Experts
        self.experts = MixtureOfExperts(
            dim=self.dim,
            num_experts=self.num_experts,
            dim_head=self.dim_head,
            dropout=self.dropout,
            ff_mult=ff_mult
        )
        
        # Attention
        self.attn = Attention(
            dim,
            dim_head,
            heads,
            True,
            flash=gpu,
            qk_norm=True,
            *args,
            **kwargs
        )
        
    def forward(self, x: Tensor):
        # Attention
        x = self.attn(x)
        
        # Expert
        x = self.experts(x)
            
        return x
    
# Tensor
x = torch.randn(1, 64, 512)
model = DenseEncoderLayer(512, 4, 8, 4, 64, 0.1, 4)
print(model(x).shape)


# LiMoE: Linear Mixture of Experts
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
        
    def forward(self, x: Tensor):
        # Encoder
        for _ in range(self.depth):
            x = DenseEncoderLayer(
                dim=self.dim,
                depth=self.depth,
                num_experts=self.num_experts,
                dim_head=self.dim_head,
                dropout=self.dropout
            )(x)
            
        return x
        
        