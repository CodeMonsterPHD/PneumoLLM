# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Embedding, Linear
from timm.models.layers import  DropPath
import clip
from torch.cuda.amp import autocast
from collections import OrderedDict


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    hidden_proj: int=128

    max_batch_size: int = 32
    max_seq_len: int = 2048
    drop_path: float = 0.


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        #modified bias for reparameterizing
        self.wq = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wk = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wv = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wo = Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False
        )

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):

        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        keys = xk
        values = xv

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = Linear(
            dim, hidden_dim, bias=False
        )
        self.w2 = Linear(
            hidden_dim, dim, bias=False
        )
        self.w3 = Linear(
            dim, hidden_dim, bias=False
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x),inplace=False) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.drop_path = DropPath(args.drop_path) if args.drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):

        h = x + self.drop_path(self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, adapter))
        out = h + self.drop_path(self.feed_forward.forward(self.ffn_norm(h)))
        return out


class AdapterMLP(nn.Module):
    """ Pytorch Implemention of RepAdapter for 1d tensor"""

    def __init__(
            self,
            in_features=32,
            hidden_dim=128,
            out_features=32
    ):
        super().__init__()
        self.conv_A=nn.Linear(in_features,hidden_dim)
        self.conv_B = nn.Linear(hidden_dim, out_features)

        nn.init.xavier_uniform_( self.conv_A.weight)
        nn.init.zeros_(self.conv_A.bias)
        nn.init.xavier_uniform_(self.conv_B.weight)
        nn.init.zeros_(self.conv_B.bias)

    def forward(self, x):
        with autocast():
            x=self.conv_B(F.silu(self.conv_A(x)))
        return x

class conv_backbone(nn.Module):
    def __init__(self, patch_size=32, width=1024):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]  ([16, 1024, 16, 16])
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2] [16, 1024, 256]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]  [16, 256, 1024]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)
        return x
class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = Embedding(
            params.vocab_size, params.dim
        )  # vocab_size: 32000, dim: 4096

        # NOTICE: do not set ignore_index
        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        # self.output = Linear(params.dim, params.vocab_size, bias=False)
        self.adapter_output_cls = Linear(params.dim, 2, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )
        # self.backbone = conv_backbone().float()
        self.backbone = clip.load('ViT-L/14')[0]
        # self.backbone = clip.load('ViT-B/32')[0]
        # self.backbone = clip.load('RN50')[0]
        # initialization token
        self.adapter_proj = AdapterMLP(1024, params.hidden_proj, params.dim).float()
        self.dtype = self.backbone.visual.conv1.weight.dtype
        self.define_mask()
        self.prompt_meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(1024, 12)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(12, self.params.RPO_K))
        ]))

    def define_mask(self):
        # image encoder mask
        att_size = 24//self.params.multiscale + self.params.RPO_K
        visual_mask = torch.zeros((att_size, att_size), dtype=self.dtype, requires_grad=False)
        visual_mask[:, -1 * self.params.RPO_K:] = float("-inf")
        #####
        self.visual_mask = visual_mask
    def initialization_token(self):
        #### visual token initialization ####
        visual_token = self.backbone.visual.class_embedding
        self.K = self.params.RPO_K  # the number of prompt pair
        visual_token = visual_token.repeat(self.K, 1)
        visual_noise = torch.randn(self.K, 1024)
        visual_noise = visual_noise / visual_noise.norm(dim=-1, keepdim=True)
        visual_token += 0.1 * visual_noise
        visual_token = visual_token.type(self.dtype)
        self.img_prompt = nn.Parameter(visual_token)

    def insert_image_embeds(self, examples, labels, images_embeds, prefix_img):
        _bsz, seqlen, _ = examples.shape
        new_examples=[]
        new_labels=[]
        for i, (example, label) in enumerate(zip(examples, labels)):
            new_example=torch.cat([example[:1], prefix_img, images_embeds[i], example[1:]], 0)  # [49, 4096]
            new_label=torch.cat([label[:1], torch.zeros(prefix_img.shape[0]+images_embeds.shape[1]).to(examples.device).type_as(labels), label[1:]])
            new_example = new_example[:seqlen]
            new_label = new_label[:seqlen]
            new_examples.append(new_example.unsqueeze(0))
            new_labels.append(new_label.unsqueeze(0))
        new_examples = torch.cat(new_examples, 0)
        new_labels = torch.cat(new_labels, 0)
        return new_examples, new_labels

    def forward(self, images, labels, return_preds=False, return_feas=False):

        images_embeds, patch_embeds = self.backbone.encode_image(images,
                                                                 multiscale=self.params.multiscale)  # [B, 3, 224, 224] --> [B, 6, 1024]

        img_prompt = self.prompt_meta_net(images_embeds.float()).permute(0, 2, 1)
        img_prompt = F.softmax(img_prompt, dim=-1) @ images_embeds.float()

        images_embeds = torch.cat([images_embeds, img_prompt], dim=1)
        images_embeds = self.adapter_proj(images_embeds)  # [B, 6, 1024] --> [B, 6, 4096]
        h = images_embeds
        seqlen = images_embeds.shape[1]  # [6]



        # todo: is this position embedding?
        freqs_cis = self.freqs_cis.to(h.device)  # [2L, 12]
        freqs_cis = freqs_cis[:seqlen]  # [L, 12]
        mask = torch.full((1, 1, seqlen, seqlen), 0., device=h.device)
        # todo: check here
        start_pos = 0
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, self.visual_mask)

        h = self.norm(h)  # [B, 6, 4096]
        feature = torch.mean(h, dim=1)  # [B, 4096]
        pred = self.adapter_output_cls(feature.float()) # [B, 2]
        loss = self.criterion(pred, labels)
        if return_preds:
            if return_feas:
                return pred, loss, feature.squeeze(0).detach().cpu().numpy()
            else:
                return pred, loss
        else:
            # loss = self.criterion(pred, labels)
            if return_feas:
                return loss, feature.squeeze(0).detach().cpu().numpy()
            else:
                return loss
