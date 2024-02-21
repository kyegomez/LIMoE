[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# LiMoE
Implementation of the "the first large-scale multimodal mixture of experts models." from the paper: "Multimodal Contrastive Learning with LIMoE: the Language-Image Mixture of Experts". [CLICK HERE FOR THE PAPER LINK:](https://arxiv.org/abs/2206.02770)


## install
`pip install limoe`

## usage
```python
import torch

from limoe.main import LiMoE

# Text tokens (batch, sequence length)
text = torch.randint(0, 100, (1, 64))

# image (batch, channels, height, width)
image = torch.randn(1, 3, 224, 224)

# Create an instance of LiMoE with the specified parameters
model = LiMoE(
    dim=64,  # Dimension of the input and output tensors
    depth=3,  # Number of layers in the encoder
    heads=8,  # Number of attention heads
    num_tokens=100,  # Number of tokens in the vocabulary
    seq_length=64,  # Length of the input sequence
    num_experts=4,  # Number of experts in the mixture-of-experts layer
    dim_head=64,  # Dimension of each attention head
    dropout=0.1,  # Dropout rate
    ff_mult=4,  # Multiplier for the dimension of the feed-forward layer
    patch_size=16,  # Patch size
    image_size=224,  # Image size
    channels=3,  # Number of image channels
    dense_encoder_depth=5,
)

# Pass the input tensor through the model and print the output
out = model(text, image)

# Print
print(out)
```

# License
MIT


## Citation
```bibtex
@misc{mustafa2022multimodal,
    title={Multimodal Contrastive Learning with LIMoE: the Language-Image Mixture of Experts}, 
    author={Basil Mustafa and Carlos Riquelme and Joan Puigcerver and Rodolphe Jenatton and Neil Houlsby},
    year={2022},
    eprint={2206.02770},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```