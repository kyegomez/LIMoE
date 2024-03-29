import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from swarms_torch import SimpleMoE
from torch import Tensor, nn
from zeta.nn import Attention, OutputHead, threed_to_text


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def contrastive_loss(
    output1: Tensor,
    output2: Tensor,
    target: Tensor,
    margin: float = 1.0,
):
    """
    Computes the contrastive loss between two output tensors.

    Args:
        output1 (Tensor): The first output tensor.
        output2 (Tensor): The second output tensor.
        target (Tensor): The target tensor.
        margin (float, optional): The margin value for the contrastive loss. Defaults to 1.0.

    Returns:
        Tensor: The computed contrastive loss.
    """
    # Compute the pairwise distance of the outputs
    distance = F.pairwise_distance(output1, output2)

    # Compute the contrastive loss
    loss_contrastive = torch.mean(
        (1 - target) * torch.pow(distance, 2)
        + (target) * torch.pow(torch.clamp(margin - distance, min=0.0), 2)
    )

    return loss_contrastive


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, target):
        # Check that inputs are tensors
        assert isinstance(output1, torch.Tensor), "output1 needs to be a PyTorch Tensor"
        assert isinstance(output2, torch.Tensor), "output2 needs to be a PyTorch Tensor"
        assert isinstance(target, torch.Tensor), "target needs to be a PyTorch Tensor"

        # Check that all tensors have the same length
        assert (
            output1.shape[0] == output2.shape[0] == target.shape[0]
        ), "output1, output2, and target need to have the same length"

        # Compute the pairwise distance of the outputs
        euclidean_distance = F.pairwise_distance(output1, output2)

        # Compute contrastive loss
        loss_contrastive = torch.mean(
            (1 - target) * torch.pow(euclidean_distance, 2)
            + (target)
            * torch.pow(
                torch.clamp(self.margin - euclidean_distance, min=0.0),
                2,
            )
        )

        return loss_contrastive


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
        dropout: float,
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

        # Norm
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor):
        """
        Forward pass of the DenseEncoderLayer.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        x = self.norm(x)

        # Attention
        x, _ = self.attn(x)
        x = self.norm(x)

        # Expert
        x = self.experts(x)
        x = self.norm(x)
        print(f"Output shape: {x.shape}")

        return x


class LiMoE(nn.Module):
    """
    LiMoE (Linearly Modularized Encoder) module.

    Args:
        dim (int): The input dimension.
        depth (int): The number of layers in the encoder.
        heads (int): The number of attention heads.
        num_experts (int): The number of experts in the mixture of experts.
        dim_head (int): The dimension of each attention head.
        dropout (int): The dropout rate.
        ff_mult (int): The multiplier for the feed-forward layer dimension.

    Attributes:
        dim (int): The input dimension.
        depth (int): The number of layers in the encoder.
        num_experts (int): The number of experts in the mixture of experts.
        dim_head (int): The dimension of each attention head.
        dropout (int): The dropout rate.
        ff_mult (int): The multiplier for the feed-forward layer dimension.
        layers (nn.ModuleList): The list of DenseEncoderLayer instances.

    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        num_tokens: int,
        seq_length: int,
        num_experts: int,
        dim_head: int,
        dropout: float,
        ff_mult: int,
        patch_size: int,
        image_size: int,
        channels: int,
        dense_encoder_depth: int,
        stride: int = 1,
        padding: int = 1,
        kernel_size: int = 3,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.num_tokens = num_tokens
        self.seq_length = seq_length
        self.num_experts = num_experts
        self.dim_head = dim_head
        self.dropout = dropout
        self.ff_mult = ff_mult
        self.image_size = image_size
        self.channels = channels
        self.patch_size = patch_size
        self.dense_encoder_depth = dense_encoder_depth

        # Patch utils
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        patch_dim = channels * patch_height * patch_width

        # Layers
        self.layers = nn.ModuleList([])

        # Add Dense Encoder layers
        for _ in range(depth):
            self.layers.append(
                DenseEncoderLayer(
                    dim,
                    dense_encoder_depth,
                    heads,
                    num_experts,
                    dim_head,
                    dropout,
                    ff_mult,
                    *args,
                    **kwargs,
                )
            )

        # Text embedding
        self.embed = nn.Embedding(num_tokens, dim)

        # Patch embedding for images using einops
        self.img_patch = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # Output head that applies a layer norm -> linear -> softmax for logits
        self.head = OutputHead(dim, 1)

        # Norm
        self.norm = nn.LayerNorm(dim)

        # Average pooling
        self.avg_pool = nn.AvgPool1d(kernel_size, stride, padding)

    def forward(self, x: Tensor = None, image: Tensor = None, *args, **kwargs):
        """
        Forward pass of the LiMoE module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        # Text embedding
        t_tensors = self.embed(x)
        b, s, d = t_tensors.shape
        t_tensors = self.norm(t_tensors)

        # If image then patch embed
        if image is not None:
            image = self.img_patch(image)
            print(f"text shape: {t_tensors.shape}")
            print(f"Image shape: {image.shape}")
            image = threed_to_text(image, s, d)
            print(f"Image shape after threed_to_text: {image.shape}")

            t_tensors = torch.cat((t_tensors, image), dim=1)
            print(f"Concatenated shape: {t_tensors.shape}")

        for layer in self.layers:
            tokens = layer(t_tensors)

        logits = self.avg_pool(tokens)
        print(f"average pooling shape: {logits.shape}")
        logits = self.head(tokens)
        return logits
