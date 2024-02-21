import torch
from torch import nn, Tensor
from zeta.nn import (
    Attention,
)
from swarms_torch import SimpleMoE


class DenseEncoderLayer(nn.Module):
    """
    DenseEncoderLayer is a module that represents a single layer of a dense encoder.

    Args:
        dim (int): The input dimension.
        depth (int): The depth of the encoder layer.
        heads (int): The number of attention heads.
        num_experts (int): The number of experts in the mixture of experts.
        dim_head (int): The dimension of each attention head.
        dropout (int): The dropout rate.
        ff_mult (int): The multiplier for the feed-forward network dimension.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Attributes:
        dim (int): The input dimension.
        depth (int): The depth of the encoder layer.
        num_experts (int): The number of experts in the mixture of experts.
        dim_head (int): The dimension of each attention head.
        dropout (int): The dropout rate.
        ff_mult (int): The multiplier for the feed-forward network dimension.
        heads (int): The number of attention heads.
        scale (float): The scaling factor for the attention weights.
        experts (MixtureOfExperts): The mixture of experts module.
        attn (Attention): The attention module.

    """

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
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_experts = num_experts
        self.dim_head = dim_head
        self.dropout = dropout
        self.ff_mult = ff_mult

        self.heads = self.dim // self.dim_head
        self.scale = self.dim_head**-0.5

        gpu = "cuda" if torch.cuda.is_available() else "cpu"

        # Experts
        self.experts = SimpleMoE(
            dim,
            dim * ff_mult,
            dim,
            num_experts,
            ff_mult,
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
            **kwargs,
        )

    def forward(self, x: Tensor):
        """
        Forward pass of the DenseEncoderLayer.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        # Attention
        x, _ = self.attn(x)

        # Expert
        x = self.experts(x)

        return x


# Tensor
x = torch.randn(1, 64, 512)
model = DenseEncoderLayer(
    dim=512,
    depth=3,
    heads=8,
    dim_head=64,
    num_experts=4,
    # dim_head = 64,
    dropout=0.1,
    ff_mult=4,
)
print(model(x))
