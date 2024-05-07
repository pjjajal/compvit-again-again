import torch
import torch.nn
import torch.utils.data
from types import SimpleNamespace
from typing import Any, List, Tuple, Iterator
from argparse import Namespace
import copy

### TIMM
from timm.models.vision_transformer import Block, Attention

###
### Helper Function
###
def HATPMetric(softmax_attn : torch.Tensor, v : torch.Tensor, hatp_info : SimpleNamespace) -> torch.Tensor:
        ###
        ### metric_attn Measures how much each token is attended to by all other tokens - this is a metric to gauge token 'importance'
        ###
        metric_attn = (
            torch.amax(softmax_attn[..., hatp_info.prefix_tokens :, hatp_info.prefix_tokens :], dim=1)
            #attn[..., info.num_prefix_tokens :, info.num_prefix_tokens :]
            #.mean(dim=1) # Average over attention heads
            .sum(dim=1, keepdim=True) # Sum along the 'rows' - this essentially counts how much a particular token is attended to by all other tokens
            .transpose(-2,-1) # Need to reshape into (B, ..., 1) instead of (B, 1, ...) shape
        )
        # ### Maximum possible value for any score = 1
        metric_attn = metric_attn / torch.max(metric_attn)

        ###
        ### metric_magnitude estimates information content (magnitude of information) by summing features of k - this is a metric to gauge token 'importance'
        ###
        metric_magnitude = torch.softmax(
            torch.amax(v, dim=1)
            #v # The K matrix!
            #.mean(dim=1) # Average over attention heads
            .sum(dim=-1, keepdim=True), # Sum along the # feature per head dimension
            dim=1 # Apply softmax across the sums of features for each token
        )[:, hatp_info.prefix_tokens :] # Shave off special tokens, attenuate
        ### Maximum possible value for any score = 1

        ###
        ### Combine Importance Scores
        ###
        metric = (metric_attn + metric_magnitude)

        return metric


###
### Top-K Metric
###
def TopKMetric(softmax_attn : torch.Tensor, hatp_info : SimpleNamespace) -> torch.Tensor:
    ### NOTE: Assumes CLS token exists
    ### softmax attn has B, # heads, N, N
    ### Collapse head dimension via average
    metric = softmax_attn.mean(dim=1)[:, 0, 1:]

    ### Metric has shape B, N-1, 1
    return metric.unsqueeze(-1)


###
### Top-K Forward
###
def TopKForward(x : torch.Tensor, metric : torch.Tensor, hatp_info : SimpleNamespace) -> torch.Tensor:
    ### Apply metric
    B, N, C = x.size()
    r = hatp_info.r.pop(0)
    T = N - r - 1

    ### Return early if we don't have to do anything
    if r == 0 or T <= hatp_info.prefix_tokens:
        return x
    
    ### Add offset - since we shaved off the prefix tokens we need to account for that with our indices
    offset = torch.tensor(
        data=[1], dtype=torch.long, device=x.device
    )

    indices = metric.argsort(dim=1, descending=True) + offset
    kept_indices = indices[:, :T]

    x = torch.cat(
        (
            x[:, 0 : hatp_info.prefix_tokens, :],
            torch.gather(x, dim=1, index=kept_indices.expand(B, T, C)),
        ),
        dim=1,
    )

    #print(f"topk post prune x.shape {x.shape}")

    return x


def HATPForward(x : torch.Tensor, metric : torch.Tensor, hatp_info : SimpleNamespace) -> torch.Tensor:
    ### Apply metric
    B, N, C = x.size()
    r = hatp_info.r.pop(0)
    T = N - r - 2

    ### Return early if we don't have to do anything
    if r == 0 or T <= hatp_info.prefix_tokens:
        return x
    
    #print(f"popped r={r}")
    #print(f"prefix tokens={hatp_info.prefix_tokens}")
    #print(f"remaining blocks={len(hatp_info.r)}")

    ### Add offset - since we shaved off the prefix tokens we need to account for that with our indices
    offset = torch.tensor(
        data=[hatp_info.prefix_tokens], dtype=torch.long, device=x.device
    )
    similarity_indices = metric.argsort(dim=1, descending=True) + offset

    ### Keep highest scores
    kept_indices = similarity_indices[:, :T]
    discard_indices = similarity_indices[:, T:]

    ### Congregate the discarded tokens then take the mean of them
    x_discarded = torch.gather(x, dim=1, index=discard_indices.expand(B, discard_indices.shape[1], C)).mean(dim=1, keepdim=True)

    ### Create pruned x'
    x = torch.cat(
        (
            x[:, 0 : hatp_info.prefix_tokens, :],
            torch.gather(x, dim=1, index=kept_indices.expand(B, T, C)),
            x_discarded,
        ),
        dim=1,
    )

    #print(f"hatp post prune x.shape {x.shape}")

    return x


###
### Intended to be used with image classification tasks
### Function that computes masked attention according to the Dynamic ViT (2021 NEURIPS) paper in addition to a novel technique
### Intended to overwrite an Attention layer during training, copying all of its attributes but now with additional functionality
###
class CustomTIMMAttention(Attention):
    ### Functions for computing Attention
    def forward(self, x: torch.Tensor, hatp_info: SimpleNamespace) -> torch.Tensor:
        B, N, C = x.size()
        ### Emulate qkv matrix from TIMM VisionTransformer
        ### Code is taken from forward(...) of TIMM VisonTransformer Attention Block
        qkv_vectors = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv_vectors.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        metric = HATPMetric(attn, v, hatp_info)

        ### Dropout Layer
        attn = self.attn_drop(attn)

        ### Now finish the attention computation
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, metric


###
### Intended to be used with image classification tasks
### Function that computes masked attention according to the Dynamic ViT (2021 NEURIPS) paper in addition to a novel technique
### Intended to overwrite an Attention layer during training, copying all of its attributes but now with additional functionality
###
class TopKTIMMAttention(Attention):
    ### Functions for computing Attention
    def forward(self, x: torch.Tensor, hatp_info: SimpleNamespace) -> torch.Tensor:
        B, N, C = x.size()
        ### Emulate qkv matrix from TIMM VisionTransformer
        ### Code is taken from forward(...) of TIMM VisonTransformer Attention Block
        qkv_vectors = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv_vectors.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        metric = TopKMetric(attn, hatp_info)

        ### Dropout Layer
        attn = self.attn_drop(attn)

        ### Now finish the attention computation
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, metric


###
### TIMM Compatible Block for Transformer Models
###
class CustomTIMMAttentionBlock(Block):
    ###
    ### x will have shape (batch size, # tokens, features)
    ### token decision mask will have shape (batch size, # tokens, 1)
    ###
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn, metric = self.attn(self.norm1(x), self._hatp_info)
        x = x + self.drop_path1(self.ls1(attn))

        x = HATPForward(x, metric, self._hatp_info)

        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


###
### TIMM Compatible Block for Transformer Models
###
class TopKTIMMAttentionBlock(Block):
    ###
    ### x will have shape (batch size, # tokens, features)
    ### token decision mask will have shape (batch size, # tokens, 1)
    ###
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn, metric = self.attn(self.norm1(x), self._hatp_info)
        x = x + self.drop_path1(self.ls1(attn))

        x = TopKForward(x, metric, self._hatp_info)

        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


###
### Wrapper Function
### Heavily inspired by Token Merging apply_patch(...) and model wrapping technique
###
def get_timm_wrapped_class(vit : torch.nn.Module):
    ### Define our own little function that steals the class from the input 'vit'
    class HAVisionTransformer(vit.__class__):
        ###
        ### Overloaded from TIMM VisionTransformer - slightly modified to remove the checkpoint_seq(...) call
        ###
        def forward(self, x: torch.Tensor) -> Tuple:
            ### Update self._hatp_info
            ### Check what self.r is update accordingly
            if isinstance(self.r, list):
                assert(len(self.r) == len(self.blocks))
                self._hatp_info.r = copy.deepcopy(self.r)
            elif isinstance(self.r, int):
                self._hatp_info.r = [self.r] * len(self.blocks)
            else:
                raise AssertionError(f"Improper r type {type(self.r)}")
            
            return super().forward(x)

    ### Return
    return HAVisionTransformer


###
### Patch a specific model
###
def timm_apply_patch(
    args : Namespace, 
    vit : torch.nn.Module
    ):

    ### Generate class
    HAVisionTransformerClass = get_timm_wrapped_class(vit)
    vit.__class__ = HAVisionTransformerClass

    ### Store metadata for our data
    vit._hatp_info = SimpleNamespace(
        r = 0,
        prefix_tokens=vit.num_prefix_tokens,
    )

    ###
    ### Replace modules as necessary
    ###
    for module in vit.modules():
        if isinstance(module, Block):
            module.__class__ = CustomTIMMAttentionBlock if args.wrapper == "hatp" else TopKTIMMAttentionBlock
            module._hatp_info = vit._hatp_info
        if isinstance(module, Attention):
            module.__class__ = CustomTIMMAttention if args.wrapper == "hatp" else TopKTIMMAttention

    return vit