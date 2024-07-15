import math
import torch
import numpy as np
import torch.nn as nn

from component.feedforward_network import make_experts
from component.droppath import DropPath
from component.gate_linear_unit import GLU
from component.xmoe.moe_layer import MOELayer
from component.xmoe.routing import Top1Gate, Top2Gate
from component.rms_norm import RMSNorm
from multiscale_retention import MultiScaleRetention


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

