import torch
import torch.nn as nn
import torch.nn.functional as F


def make_exit_class(transformer_class: torch.nn.Module):
    class DinoExited(transformer_class):
        def forward_features(self, x, masks=None, **kwargs):
            if isinstance(x, list):
                return self.forward_features_list(x, masks)

            x = self.prepare_tokens_with_masks(x, masks)

            exit_at = kwargs.get("exit_at", None)
            for i, blk in enumerate(self.blocks):
                x = blk(x)
                if i == exit_at and exit_at is not None:
                    break

            x_norm = self.norm(x)
            return {
                "x_norm_clstoken": x_norm[:, 0],
                "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                "x_prenorm": x,
                "x_norm": x_norm,
                "masks": masks,
            }
    return DinoExited


def exit_patch(model: nn.Module):
    exit_class = make_exit_class(model.__class__)
    model.__class__ = exit_class
    return model