from functools import partial
from typing import Callable, Literal, Optional, Tuple, Type, Union

import torch.nn as nn
from timm.layers import Mlp, PatchEmbed, get_act_layer, get_norm_layer
from timm.layers.typing import LayerType
from timm.models._builder import build_model_with_cfg
from timm.models.vision_transformer import (
    Block,
    VisionTransformer,
    checkpoint_filter_fn,
)
import torch
from torch.nn.modules import Module
from typing_extensions import Literal

from compvit.layers.compressor import Compressor


# Make the forward similar to DINO
class DeiT(VisionTransformer):
    def forward(self, x: torch.Tensor, is_training=False) -> torch.Tensor:
        x = self.forward_features(x)
        if is_training:
            return {
                "x_norm_clstoken": x[:, 0],
                "x_norm_patchtokens": x[:, 1:],
            }
        x = self.forward_head(x)
        return x

class CompDeiT(DeiT):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: Literal["", "avg", "token", "map"] = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        class_token: bool = True,
        no_embed_class: bool = False,
        reg_tokens: int = 0,
        pre_norm: bool = False,
        fc_norm: Optional[bool] = None,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        weight_init: Literal["skip", "jax", "jax_nlhb", "moco", ""] = "",
        fix_init: bool = False,
        embed_layer: Callable = PatchEmbed,
        norm_layer: Optional[LayerType] = None,
        act_layer: Optional[LayerType] = None,
        block_fn: Type[nn.Module] = Block,
        mlp_layer: Type[nn.Module] = Mlp,
        num_compressed_tokens=0,
        bottleneck_loc=5,
    ) -> None:
        super().__init__(
            img_size,
            patch_size,
            in_chans,
            num_classes,
            global_pool,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            qkv_bias,
            qk_norm,
            init_values,
            class_token,
            no_embed_class,
            reg_tokens,
            pre_norm,
            fc_norm,
            dynamic_img_size,
            dynamic_img_pad,
            drop_rate,
            pos_drop_rate,
            patch_drop_rate,
            proj_drop_rate,
            attn_drop_rate,
            drop_path_rate,
            weight_init,
            fix_init,
            embed_layer,
            norm_layer,
            act_layer,
            block_fn,
            mlp_layer,
        )
        norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        act_layer = get_act_layer(act_layer) or nn.GELU

        self.total_tokens = self.patch_embed.num_patches
        self.num_compressed_tokens = num_compressed_tokens  # Add CLS Token
        self.bottleneck_loc = bottleneck_loc

        self.compressor = Compressor(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
            norm_layer=norm_layer,
            act_layer=act_layer,
            ffn_layer=mlp_layer,
            init_values=init_values,
            num_compressed_tokens=self.num_compressed_tokens,
            num_tokens=self.total_tokens,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == self.bottleneck_loc:
                x = self.compressor(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if self.attn_pool is not None:
            x = self.attn_pool(x)
        elif self.global_pool == "avg":
            x = x[:, self.num_prefix_tokens :].mean(dim=1)
        elif self.global_pool:
            x = x[:, 0]  # class token
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)


def _create_deit(variant, pretrained=False, compvit=False, **kwargs):
    if kwargs.get("features_only", None):
        raise RuntimeError(
            "features_only not implemented for Vision Transformer models."
        )
    model_cls = CompDeiT if compvit else DeiT
    model = build_model_with_cfg(
        model_cls,
        variant,
        pretrained,
        pretrained_filter_fn=partial(checkpoint_filter_fn, adapt_layer_scale=True),
        **kwargs,
    )
    return model


def deit_tiny_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """DeiT-tiny model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    model = _create_deit(
        "deit_tiny_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model, {**model_args, **kwargs}


def deit3_small_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """DeiT-3 small model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        no_embed_class=True,
        init_values=1e-6,
    )
    model = _create_deit(
        "deit3_small_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model, {**model_args, **kwargs}


def deit3_base_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """DeiT-3 base model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        no_embed_class=True,
        init_values=1e-6,
    )
    model = _create_deit(
        "deit3_base_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model, {**model_args, **kwargs}


def deit3_large_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """DeiT-3 large model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        no_embed_class=True,
        init_values=1e-6,
    )
    model = _create_deit(
        "deit3_large_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model, {**model_args, **kwargs}


if __name__ == "__main__":
    # model = deit_tiny_patch16_224(pretrained=False)
    # print(model(torch.randn(1, 3, 224, 224)).cuda().shape)
    model = deit_tiny_patch16_224(pretrained=False, compvit=True)
    print(model(torch.randn(1, 3, 224, 224)).shape)
