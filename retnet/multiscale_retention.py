# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import torch
from torch import nn
import torch.nn.functional as F
from component.rms_norm import RMSNorm
from component.multiway_network import MultiwayWrapper


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\


def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)


def get_activation_fn(activation):
    if activation == "swish":
        return F.silu
    elif activation == "gelu":
        return F.gelu
    else:
        raise NotImplementedError


class MultiScaleRetention(nn.Module):
    
    def __init__( self, args, embed_dim, value_dim, num_heads, gate_fn="swish",):
        
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.head_dim = self.value_dim // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        
        self.gate_fn = get_activation_fn(activation=str(gate_fn))

        self.q_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=False))
        self.k_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=False))
        self.v_proj = MultiwayWrapper(args, nn.Linear(embed_dim, value_dim, bias=False))
        self.g_proj = MultiwayWrapper(args, nn.Linear(embed_dim, value_dim, bias=False))
        
        self.out_proj = MultiwayWrapper(args, nn.Linear(value_dim, embed_dim, bias=False))

        self.group_norm = MultiwayWrapper(args, RMSNorm(self.head_dim, eps=args.layernorm_eps, elementwise_affine=False))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.g_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=2 ** -1)

    def parallel_forward(self, qr, kr, v, mask):
        bsz, tgt_len, embed_dim = v.size()

        vr = v.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        qk_mat = qr @ kr.transpose(-1, -2) # bsz * m * tgt_len * tgt_len
        qk_mat = qk_mat * mask
        # invariant after normalization
        qk_mat = qk_mat / qk_mat.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1, max=5e4)
        output = torch.matmul(qk_mat, vr)
        output = output.transpose(1, 2)
        return output

    def forward(self, x, rel_pos=None):

        bsz, tgt_len, _ = x.size()      # Betch Size, Target_length, em_dim
        
        (sin, cos), inner_mask = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        g = self.g_proj(x)

        k *= self.scaling
        q = q.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        k = k.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)

        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)

        output = self.parallel_forward(qr, kr, v, inner_mask)
        
        output = self.group_norm(output).reshape(bsz, tgt_len, self.head_dim * self.num_heads)

        output = self.gate_fn(g) * output

        output = self.out_proj(output)

        return output

        
