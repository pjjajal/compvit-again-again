from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dinov2.layers import MemEffAttention, Mlp
from dinov2.layers import NestedTensorBlock as Block
from dinov2.layers import PatchEmbed, SwiGLUFFNFused

from .attention import AttentionComp, CrossAttention
from .block import Block, CompBlock
from .query_bank import QueryBank


class Compressor(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        init_values=None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        num_compressed_tokens: int = 16,
        num_tokens: int = 196,
        num_register_tokens: int = 0,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.num_compressed_tokens = num_compressed_tokens
        self.num_register_tokens = num_register_tokens

        self.norm = norm_layer(dim)
        self.queries = nn.Parameter(
            torch.randn((1, num_compressed_tokens, dim)), requires_grad=True
        )
        self.ca_block = CompBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            ffn_layer=ffn_layer,
            init_values=init_values,
            attn_class=CrossAttention,
        )
        self.ca_block_2 = CompBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            ffn_layer=ffn_layer,
            init_values=init_values,
            attn_class=CrossAttention,
        )

    def forward(self, x, get_attn=False):
        B, N, C = x.shape

        if self.num_register_tokens > 0:
            cls_token, registers, x = (
                x[:, 0:1],
                x[:, 1 : self.num_register_tokens + 1],
                x[:, self.num_register_tokens + 1 :],
            )
        else:
            cls_token, x = x[:, 0:1], x[:, 1:]

        x = self.norm(x)

        compressed_tokens = self.ca_block(x, self.queries, get_attn)
        x = torch.cat([x, compressed_tokens], dim=1)
        compressed_tokens = self.ca_block_2(x, compressed_tokens, get_attn)

        if self.num_register_tokens > 0:
            compressed_tokens = torch.cat(
                [cls_token, registers, compressed_tokens], dim=1
            )
        else:
            compressed_tokens = torch.cat([cls_token, compressed_tokens], dim=1)

        return compressed_tokens


class CompressorQueryBank(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        init_values=None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        num_compressed_tokens: int = 16,
        num_tokens: int = 196,
        num_register_tokens: int = 0,
        bottleneck: nn.Module = None,
        bank_size: int = 0,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_tokens = num_tokens
        self.num_compressed_tokens = num_compressed_tokens
        self.num_register_tokens = num_register_tokens

        self.norm = norm_layer(dim)
        self.queries = QueryBank(
            dim, banksize=bank_size if bank_size > 0 else num_compressed_tokens
        )

        self.ca_block = CompBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            ffn_layer=ffn_layer,
            init_values=init_values,
            attn_class=CrossAttention,
        )
        self.sa_block = Block(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            ffn_layer=ffn_layer,
            init_values=init_values,
            attn_class=AttentionComp,
        )
        self.ca_block_2 = CompBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            ffn_layer=ffn_layer,
            init_values=init_values,
            attn_class=CrossAttention,
        )

    def forward(self, x, get_attn=False):
        B, N, C = x.shape

        x = self.norm(x)

        if self.num_register_tokens > 0:
            cls_token, registers, x = (
                x[:, 0:1],
                x[:, 1 : self.num_register_tokens + 1],
                x[:, self.num_register_tokens + 1 :],
            )
        else:
            cls_token, x = x[:, 0:1], x[:, 1:]

        x, queries = self.queries.forward(x, self.num_compressed_tokens)

        compressed_tokens = self.ca_block(x, queries, get_attn)
        x = torch.cat([x, compressed_tokens], dim=1)
        compressed_tokens = self.sa_block(compressed_tokens)
        compressed_tokens = self.ca_block_2(x, compressed_tokens, get_attn)

        if self.num_register_tokens > 0:
            compressed_tokens = torch.cat(
                [cls_token, registers, compressed_tokens], dim=1
            )
        else:
            compressed_tokens = torch.cat([cls_token, compressed_tokens], dim=1)

        return compressed_tokens


class CompressorMlp(Compressor):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        init_values=None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        num_compressed_tokens: int = 16,
        num_tokens: int = 196,
        num_register_tokens: int = 0,
    ) -> None:
        super().__init__(
            dim,
            num_heads,
            mlp_ratio,
            qkv_bias,
            proj_bias,
            ffn_bias,
            init_values,
            act_layer,
            norm_layer,
            ffn_layer,
            num_compressed_tokens,
            num_tokens,
            num_register_tokens,
        )

        self.mlp = Mlp(dim, int(dim * mlp_ratio), num_compressed_tokens)

    def forward(self, x, get_attn=False):
        B, N, C = x.shape

        if self.num_register_tokens > 0:
            cls_token, registers, x = (
                x[:, 0:1],
                x[:, 1 : self.num_register_tokens + 1],
                x[:, self.num_register_tokens + 1 :],
            )
        else:
            cls_token, x = x[:, 0:1], x[:, 1:]

        x = self.norm(x)

        queries = self.mlp(x)
        queries = F.softmax(queries.mT, dim=-1)
        queries = queries @ x # (B, num_compressed_tokens, C)
        queries = queries + self.queries

        compressed_tokens = self.ca_block(x, queries, get_attn)
        x = torch.cat([x, compressed_tokens], dim=1)
        compressed_tokens = self.ca_block_2(x, compressed_tokens, get_attn)

        if self.num_register_tokens > 0:
            compressed_tokens = torch.cat(
                [cls_token, registers, compressed_tokens], dim=1
            )
        else:
            compressed_tokens = torch.cat([cls_token, compressed_tokens], dim=1)

        return compressed_tokens


if __name__ == "__main__":
    compressor = Compressor(384, 8, 4)
    print(compressor(torch.randn((1, 196, 384))).shape)
