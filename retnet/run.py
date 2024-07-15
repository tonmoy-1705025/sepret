# Creating a RetNet model
import time
import torch
from config import RetNetConfig
from RetNetDecoder import RetNetDecoder

betch_size = 82
sequence_len = 250
embed_dim = 256


config = RetNetConfig(
    decoder_embed_dim=embed_dim,
    decoder_value_embed_dim=embed_dim,
    decoder_retention_heads=8,
    decoder_ffn_embed_dim=128,
    decoder_layers=4,
    dropout=0.1,
)

retnet = RetNetDecoder(config).cuda()

print("{:.3f} million".format(sum([param.nelement() for param in retnet.parameters()]) / 1e6))

input = torch.randn(betch_size, sequence_len, embed_dim).long().cuda()

now = time.time()
output = retnet(input)
print(time.time() - now)

print(output.shape)