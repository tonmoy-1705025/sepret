import torch
import torch.nn as nn

class RetNetRelPos(nn.Module):
    
    def __init__(self, args):

        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, args.decoder_embed_dim // args.decoder_retention_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        decay = torch.log(1 - 2 ** (-5 - torch.arange(args.decoder_retention_heads, dtype=torch.float)))
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)
        
    def forward(self, slen, ):

        index = torch.arange(slen).to(self.decay)
        sin = torch.sin(index[:, None] * self.angle[None, :])
        cos = torch.cos(index[:, None] * self.angle[None, :])
        mask = torch.tril(torch.ones(slen, slen).to(self.decay))
        mask = torch.masked_fill(index[:, None] - index[None, :], ~mask.bool(), float("inf"))
        mask = torch.exp(mask * self.decay[:, None, None])
        mask = torch.nan_to_num(mask)
        mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()
        retention_rel_pos = ((sin, cos), mask)

        return retention_rel_pos


# class Args:
#     decoder_embed_dim = 512
#     decoder_retention_heads = 8

# args = Args()
# model = RetNetRelPos(args)
# print("{:.3f} million".format(sum([param.nelement() for param in model.parameters()]) / 1e6))
# output = model(10)


