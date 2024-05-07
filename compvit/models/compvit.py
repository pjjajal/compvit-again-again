from functools import partial
from typing import Literal, Union, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dinov2.layers import Mlp
from dinov2.layers import NestedTensorBlock as Block
from dinov2.layers import PatchEmbed, SwiGLUFFNFused
from dinov2.models.vision_transformer import DinoVisionTransformer
from ..layers.compressor import Compressor, CompressorQueryBank
from ..layers.block import CompBlock
from ..layers.attention import CrossAttention


class CompViT(DinoVisionTransformer):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0,
        drop_path_uniform=False,
        init_values=None,
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        num_compressed_tokens=[0],
        num_patches=256,
        bottleneck_loc=[5],
        **kwargs,
    ):
        super().__init__(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            qkv_bias,
            ffn_bias,
            proj_bias,
            drop_path_rate,
            drop_path_uniform,
            init_values,
            embed_layer,
            act_layer,
            block_fn,
            ffn_layer,
            block_chunks,
            num_register_tokens,
            interpolate_antialias,
            interpolate_offset,
        )
        # Boilerplate from DINOv2 implementation
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        if ffn_layer == "mlp":
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        self.total_tokens = num_patches + self.num_tokens + self.num_register_tokens
        # self.total_tokens = num_patches
        self.num_compressed_tokens = num_compressed_tokens  # Add CLS Token
        self.bank_size = kwargs.get("bank_size", 0)
        if self.bank_size:
            self.bank_token = nn.Parameter(
                torch.zeros(1, 1, embed_dim), requires_grad=True
            )

        # Add compressor.
        if num_compressed_tokens:
            self.compress = True

            # Set the blocks where bottleneck will be with None
            self.bottleneck_loc = bottleneck_loc

            compressors = []
            for loc, comp_tokens in zip(bottleneck_loc, num_compressed_tokens):
                compressors.append(
                    Compressor(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        proj_bias=proj_bias,
                        ffn_bias=ffn_bias,
                        norm_layer=norm_layer,
                        act_layer=act_layer,
                        ffn_layer=ffn_layer,
                        init_values=init_values,
                        num_compressed_tokens=comp_tokens,
                        num_tokens=self.total_tokens,
                        num_register_tokens=self.num_register_tokens,
                    )
                )
            self.compressors = nn.ModuleList(compressors)

    def forward_features(self, x, masks=None, get_attn=False, use_decoder=False):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)
        if self.bank_size:
            x = torch.cat((x, self.bank_token.expand(x.shape(0), -1, -1)), dim=1)

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.compress and i in self.bottleneck_loc:
                x = self.compressors[self.bottleneck_loc.index(i)](x, get_attn)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_norm": x_norm,
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.compress and i == self.bottleneck_loc:
                x = self.compressor(x)

            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(
            blocks_to_take
        ), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output


if __name__ == "__main__":
    print(
        CompViT(num_compressed_tokens=1, block_chunks=0, bottleneck="mixer_bottleneck")
    )
