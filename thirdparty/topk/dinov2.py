import torch

### Standard
from typing import Any, Tuple, List, Dict, Callable
import os
import warnings
import copy
from types import SimpleNamespace
import argparse

### DinoV2
from dinov2.layers.block import (
    Block,
    NestedTensorBlock,
    drop_add_residual_stochastic_depth,
)
from dinov2.layers.attention import (
    Attention,
    MemEffAttention,
    XFORMERS_AVAILABLE,
    XFORMERS_ENABLED,
)

### Local
from .timm import HATPMetric, HATPForward, TopKMetric, TopKForward

# ### Copied from DinoV2
# XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
# try:
#     if XFORMERS_ENABLED:
#         from xformers.ops import memory_efficient_attention, unbind

#         XFORMERS_AVAILABLE = True
#         #warnings.warn("xFormers is available (Attention)")
#     else:
#         #warnings.warn("xFormers is disabled (Attention)")
#         raise ImportError
# except ImportError:
#     XFORMERS_AVAILABLE = False
#     #warnings.warn("xFormers is not available (Attention)")

###
### Custom Attention Mechanisms
###
class HATPDinoV2Attention(Attention):
    def forward(self, x: torch.Tensor, hatp_info : SimpleNamespace) -> Tuple:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        ### Q, K, V have shapes
        ### B, # heads, # tokens, # features per head
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        ### Get the metric!
        metric = HATPMetric(attn, v, hatp_info)

        ### Continue as normal
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        ### Modification - this is our metric to return
        return x, metric


###
### Custom Attention Mechanisms
###
class TopKDinoV2Attention(Attention):
    def forward(self, x: torch.Tensor, hatp_info : SimpleNamespace) -> Tuple:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        ### Q, K, V have shapes
        ### B, # heads, # tokens, # features per head
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        ### Get the metric!
        metric = TopKMetric(attn, hatp_info)

        ### Continue as normal
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        ### Modification - this is our metric to return
        return x, metric


### NOTE: Does not work for our method 
class TopKDinoV2MemEffAttention(TopKDinoV2Attention):
    def forward(self, x: torch.Tensor, hatp_info : SimpleNamespace, attn_bias=None) -> Tuple:
        if True or not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x, hatp_info)

        raise AssertionError("HATP does not support MemEffAttention")

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        ### Each has shape B, #tokens, #heads, # features per head
        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)

        ### Get the metric!
        ### NOTE: Not sure this works. Disable it for now.
        ### x: doesn't have the proper shape - it happens after the softmax(q @ k.T) @ v multiplication
        ### Sad!
        metric = HATPMetric(x, v, hatp_info)
        
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, metric


### NOTE: Does not work for our method 
class HATPDinoV2MemEffAttention(HATPDinoV2Attention):
    def forward(self, x: torch.Tensor, hatp_info : SimpleNamespace, attn_bias=None) -> Tuple:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x, hatp_info)

        raise AssertionError("HATP does not support MemEffAttention")

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        ### Each has shape B, #tokens, #heads, # features per head
        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)

        ### Get the metric!
        ### NOTE: Not sure this works. Disable it for now.
        ### x: doesn't have the proper shape - it happens after the softmax(q @ k.T) @ v multiplication
        ### Sad!
        metric = HATPMetric(x, v, hatp_info)
        
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, metric


###
### Custom Blocks
###
class TopKDinoV2Block(Block):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def attn_residual_func(x: torch.Tensor) -> Tuple:
            attention, metric = self.attn(self.norm1(x), hatp_info=self._hatp_info)
            attention_scaled = self.ls1(attention)
            return attention_scaled, metric

        def ffn_residual_func(x: torch.Tensor) -> torch.Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # Raise error
            raise AssertionError(
                "HATP wrapping does not support the drop_add_residaul_stochastic_depth flag during training"
            )

            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            residual, metric = attn_residual_func(x)
            x = x + self.drop_path1(residual)

            #print(f"x shape pre: {x.shape}")

            ### Apply metric
            x = TopKForward(x, metric, self._hatp_info)

            #print(f"x shape post: {x.shape}")

            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            residual, metric = attn_residual_func(x)
            x = x + residual

            #print(f"x shape pre: {x.shape}")

            ### Apply metric
            x = TopKForward(x, metric, self._hatp_info)

            #print(f"x shape post: {x.shape}")

            x = x + ffn_residual_func(x)
        return x


###
### Custom Blocks
###
class HATPDinoV2Block(Block):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def attn_residual_func(x: torch.Tensor) -> Tuple:
            attention, metric = self.attn(self.norm1(x), hatp_info=self._hatp_info)
            attention_scaled = self.ls1(attention)
            return attention_scaled, metric

        def ffn_residual_func(x: torch.Tensor) -> torch.Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # Raise error
            raise AssertionError(
                "HATP wrapping does not support the drop_add_residaul_stochastic_depth flag during training"
            )

            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            residual, metric = attn_residual_func(x)
            x = x + self.drop_path1(residual)

            #print(f"x shape pre: {x.shape}")

            ### Apply metric
            x = HATPForward(x, metric, self._hatp_info)

            #print(f"x shape post: {x.shape}")

            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            residual, metric = attn_residual_func(x)
            x = x + residual

            #print(f"x shape pre: {x.shape}")

            ### Apply metric
            x = HATPForward(x, metric, self._hatp_info)

            #print(f"x shape post: {x.shape}")

            x = x + ffn_residual_func(x)
        return x


class HATPDinoV2NestedTensorBlock(HATPDinoV2Block):
    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, torch.Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            ### NOTE: Not supported yet
            raise AssertionError(
                "HATP wrapping does not support list of tensors / nested tensors"
            )
            if not XFORMERS_AVAILABLE:
                raise AssertionError("xFormers is required for using nested tensors")
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError


class TopKDinoV2NestedTensorBlock(TopKDinoV2Block):
    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, torch.Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            ### NOTE: Not supported yet
            raise AssertionError(
                "HATP wrapping does not support list of tensors / nested tensors"
            )
            if not XFORMERS_AVAILABLE:
                raise AssertionError("xFormers is required for using nested tensors")
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError


###
### Factory HATP DinoV2 Generator
###
def make_hatp_class(transformer_class: torch.nn.Module):
    class HATPDinoVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwargs) -> torch.Tensor:
            ### Update self._hatp_info
            ### Check what self.r is update accordingly
            if isinstance(self.r, list):
                #print(len(self.backbone.blocks))
                #print(len(self.r))
                #print(f"r = {self.r}")
                assert(len(self.r) == len(self.blocks))
                self._hatp_info.r = copy.deepcopy(self.r)
            elif isinstance(self.r, int):
                self._hatp_info.r = [self.r] * len(self.blocks)
            else:
                raise AssertionError(f"Improper r type {type(self.r)}")

            #print(f"self._hatp_info.r: {self._hatp_info.r}")

            return super().forward(*args, **kwargs)

    return HATPDinoVisionTransformer


###
### "Master" function to apply ToMe to a DinoV2 model
###
def dinov2_apply_patch(
    model: torch.nn.Module
    ):
    assert isinstance(model, torch.nn.Module)

    HATPDinoVisionTransformer = make_hatp_class(model.__class__)

    model.__class__ = HATPDinoVisionTransformer
    model.r = 0
    model._hatp_info = SimpleNamespace(
        r = model.r,
        ### NOTE: Assumes there is only 1 (CLS token) Update as needed
        prefix_tokens = 1,
    )

    ### Iterate over backbone modules
    for module in model.modules():
        ### Note: order matters
        if isinstance(module, NestedTensorBlock):
            module.__class__ = TopKDinoV2NestedTensorBlock
            module._hatp_info = model._hatp_info
        elif isinstance(module, Block):
            module.__class__ =  TopKDinoV2Block
            module._hatp_info = model._hatp_info

        ### Note: order matters
        if isinstance(module, MemEffAttention):
            module.__class__ = TopKDinoV2MemEffAttention
        elif isinstance(module, Attention):
            module.__class__ = TopKDinoV2Attention

    return model

