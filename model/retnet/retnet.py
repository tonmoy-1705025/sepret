import math
import time
import torch
import numpy as np
import torch.nn as nn

from .component import make_experts, DropPath, GLU, MOELayer, Top1Gate, Top2Gate, RMSNorm
from .multiscale_retention import MultiScaleRetention


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


class DecoderLayer(nn.Module):

    def __init__( self, args, depth, is_moe_layer=False, ):

        super().__init__()
        self.args = args
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = torch.nn.Dropout(args.dropout)
        
        # Drop path rate
        if args.drop_path_rate > 0:
            drop_path_prob = np.linspace(0, args.drop_path_rate, args.decoder_layers)[
                depth
            ]
            self.drop_path = DropPath(drop_path_prob)
        else:
            self.drop_path = None

        # Retension Layer Building
        self.retention = MultiScaleRetention(args,
                                            self.embed_dim,
                                            args.decoder_value_embed_dim,
                                            args.decoder_retention_heads,
                                            )
        
        # If Normalize before
        self.normalize_before = args.decoder_normalize_before

        # Retension Layer Normalization
        self.retention_layer_norm = RMSNorm(self.embed_dim, eps=args.layernorm_eps)

        # If moe layer
        self.is_moe_layer = is_moe_layer
        self.ffn_dim = args.decoder_ffn_embed_dim

        # If moe layer exist => 
        if not self.is_moe_layer:

            self.ffn = GLU (
                self.embed_dim,
                self.ffn_dim,
                args.activation_fn,
                args.dropout,
                args.activation_dropout,
            )
        else:
            if args.moe_top1_expert:
                gate = Top1Gate(
                    self.embed_dim,
                    args.moe_expert_count,
                    use_fp32=args.moe_gating_use_fp32,
                    moe_eval_capacity_token_fraction=args.moe_eval_capacity_token_fraction,
                    use_xmoe=args.use_xmoe,
                )
            else:
                gate = Top2Gate(
                    self.embed_dim,
                    args.moe_expert_count,
                    args.moe_gating_use_fp32,
                    args.moe_second_expert_policy,
                    args.moe_normalize_gate_prob_before_dropping,
                    args.moe_eval_capacity_token_fraction,
                    use_xmoe=args.use_xmoe,
                )

            experts = make_experts(args, self.embed_dim, self.ffn_dim)
            self.moe_layer = MOELayer(gate, experts, args)

        # Final Layer Normalyzation
        self.final_layer_norm = RMSNorm(self.embed_dim, eps=args.layernorm_eps)

        # If Deep Norm Exist
        if args.deepnorm:
            self.alpha = math.pow(2.0 * args.decoder_layers, 0.25)
        else:
            self.alpha = 1.0

    # Residual Connection
    def residual_connection(self, x, residual):
        return residual * self.alpha + x

    # Forward Pass
    def forward( self, x, retention_rel_pos=None ):

        # Action for Retension
        residual = x
        if self.normalize_before:
            x = self.retention_layer_norm(x)

        x = self.retention( x, rel_pos=retention_rel_pos, )
        x = self.dropout_module(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.retention_layer_norm(x)

        # Action for Feed Forward Netword
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        if not self.is_moe_layer:
            x = self.ffn(x)
            l_aux = None
        else:
            x, l_aux = self.moe_layer(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        return x, l_aux


class RetNetDecoder(nn.Module):

    def __init__(self, args, **kwargs ):

        super().__init__(**kwargs)
        self.args = args

        self.embed_dim = args.decoder_embed_dim

        # Define Decoder Layers
        self.decoderlayers = nn.ModuleList([])

        moe_freq = args.moe_freq
        for i in range(args.decoder_layers):
            is_moe_layer = moe_freq != 0 and (i + 1) % moe_freq == 0

            self.decoderlayers.append(
                DecoderLayer(
                    args,                        # args
                    i,                           # Depth
                    is_moe_layer=is_moe_layer,
                )
            )

        # RMS Normalization
        if args.decoder_normalize_before:
            self.layer_norm = RMSNorm(self.embed_dim, eps=args.layernorm_eps)
        else:
            self.layer_norm = None

        # RetNet Relation Position
        self.retnet_rel_pos = RetNetRelPos(args)

    # Forward pass
    def forward( self, x ):

        # Getting Sequence Lenth
        slen = x.size(1)

        # relative position
        retention_rel_pos = self.retnet_rel_pos(slen)


        for idx, decoderlayer in enumerate(self.decoderlayers):
            x, l_aux_i = decoderlayer( x, retention_rel_pos=retention_rel_pos )
            
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return x
    

# betch_size = 82
# sequence_len = 256
# embed_dim = 768

# config = RetNetConfig(
#     decoder_embed_dim=embed_dim,
#     decoder_value_embed_dim=embed_dim,
#     decoder_retention_heads=8,
#     decoder_ffn_embed_dim=128,
#     decoder_layers=4,
#     dropout=0.1,
# )

# retnet = RetNetDecoder(config).cuda()

# print("{:.3f} million".format(sum([param.nelement() for param in retnet.parameters()]) / 1e6))

# input = torch.randn(betch_size, sequence_len, embed_dim).long().cuda()

# now = time.time()
# output = retnet(input)
# print(time.time() - now)

# print(output.shape)