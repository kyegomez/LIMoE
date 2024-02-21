import torch
from limoe.main import DenseEncoderLayer

# Create a random tensor of shape (1, 64, 512)
x = torch.randn(1, 64, 512)

# Create an instance of DenseEncoderLayer with the specified parameters
model = DenseEncoderLayer(
    dim=512,  # Dimension of the input and output tensors
    depth=3,  # Number of layers in the encoder
    heads=8,  # Number of attention heads
    dim_head=64,  # Dimension of each attention head
    num_experts=4,  # Number of experts in the mixture-of-experts layer
    dropout=0.1,  # Dropout rate
    ff_mult=4,  # Multiplier for the dimension of the feed-forward layer
)

# Pass the input tensor through the model and print the output
print(model(x))
