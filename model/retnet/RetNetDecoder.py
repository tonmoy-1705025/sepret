import math
import torch.nn as nn
from fairscale.nn import checkpoint_wrapper, wrap
from component.rms_norm import RMSNorm

from DecoderLayer import DecoderLayer
from RetNetRelPos import RetNetRelPos


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
                self.build_decoder_layer( args, depth=i, is_moe_layer=is_moe_layer, )
            )

        # RMS Normalization
        if args.decoder_normalize_before:
            self.layer_norm = RMSNorm(self.embed_dim, eps=args.layernorm_eps)
        else:
            self.layer_norm = None

        # RetNet Relation Position
        self.retnet_rel_pos = RetNetRelPos(args)
        
        # Deep Normalization
        if args.deepnorm:
            init_scale = math.pow(8.0 * args.decoder_layers, 0.25)
            for name, p in self.named_parameters():
                if (
                    "fc1" in name
                    or "fc2" in name
                    or "out_proj" in name
                    or "v_proj" in name
                ):
                    p.data.div_(init_scale)


    # Build Decoder Layers
    def build_decoder_layer( self, args, depth, is_moe_layer=False ):

        layer = DecoderLayer(
            args,
            depth,
            is_moe_layer=is_moe_layer,
        )
        if args.checkpoint_activations:
            layer = checkpoint_wrapper(layer)
        if args.fsdp:
            layer = wrap(layer)
        return layer


    # Forward pass
    def forward( self, x ):

        # Getting Sequence Lenth
        slen = x.size(1)

        # relative position
        retention_rel_pos = self.retnet_rel_pos(slen)

        # decoder layers
        inner_states = [x]

        l_aux = []

        for idx, decoderlayer in enumerate(self.decoderlayers):

            x, l_aux_i = decoderlayer( x, retention_rel_pos=retention_rel_pos )
            l_aux.append(l_aux_i)
            inner_states.append(x)
            
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return x
    
