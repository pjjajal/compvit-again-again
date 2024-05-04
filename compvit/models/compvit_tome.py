from functools import partial
from typing import Literal, Union, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dinov2.layers import Mlp
from dinov2.layers import NestedTensorBlock as Block
from dinov2.layers import PatchEmbed, SwiGLUFFNFused
from dinov2.models.vision_transformer import DinoVisionTransformer
from .compvit import CompViT

from ..layers.compressor import Compressor, CompressorV2, CompressorV3
from ..layers.block import CompBlock
from ..layers.attention import CrossAttention
from thirdparty.tome.utils import parse_r
from thirdparty.tome.patch.dinov2 import ToMeDinoV2Block


class CompViTToMe(CompViT):
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
        num_compressed_tokens=0,
        num_patches=256,
        bottleneck: Literal[
            "mixer_bottleneck",
            "mixer_bottleneck_relu",
            "mixer_bottleneck_multi",
            "mixer_bottleneck_multi_v2",
            "conv_bottleneck",
        ] = "conv_bottleneck",
        bottleneck_loc=5,
        bottleneck_size=1,
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
            num_compressed_tokens,
            num_patches,
            bottleneck,
            bottleneck_loc,
            bottleneck_size,
            **kwargs,
        )

        self.total_tokens = num_patches + self.num_tokens + self.num_register_tokens
        # self.total_tokens = num_patches
        self.num_compressed_tokens = num_compressed_tokens  # Add CLS Token

        self.r = 0
        self._tome_info = {
            "r": self.r,
            "size": None,
            "source": None,
            "trace_source": False,
            "prop_attn": False,
            "class_token": True,
            "distill_token": False,
        }

        for i in self.blocks:
            print(f"Block {i}")

    def forward(self, *args, is_training=False, **kwargs):
        ### Update self._tome_info
        self._tome_info["r"] = parse_r(len(self.blocks), self.r)
        self._tome_info["size"] = None
        self._tome_info["source"] = None
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


if __name__ == "__main__":
    CompViTToMe(num_compressed_tokens=16, block_chunks=0)
