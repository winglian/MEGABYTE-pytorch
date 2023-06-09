import math
import functools
from itertools import zip_longest

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, reduce, repeat, pack, unpack
from einops.layers.torch import Rearrange

from beartype import beartype
from beartype.typing import Tuple, Union

from MEGABYTE_pytorch.attend import Attend

from tqdm import tqdm

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def remainder_to_mult(num, mult):
    return (mult - num % mult) % mult

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

def reduce_mult(nums):
    return functools.reduce(lambda x, y: x * y, nums, 1)

# tensor helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# token shift, from Peng et al of RWKV

def token_shift(t):
    t, t_shift = t.chunk(2, dim = -1)
    t_shift = F.pad(t_shift, (0, 0, 1, -1))
    return torch.cat((t, t_shift), dim = -1)

# positional bias

class Alibi(nn.Module):
    def __init__(self, heads, **kwargs):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.register_buffer('slopes', slopes, persistent = False)
        self.register_buffer('bias', None, persistent = False)

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    def forward(self, i, j, device):
        if exists(self.bias) and self.bias.shape[-1] >= j:
            return self.bias[..., :j]

        bias = torch.arange(j, device = device)
        bias = rearrange(bias, 'j -> 1 1 j')
        bias = bias * self.slopes

        self.register_buffer('bias', bias, persistent = False)
        return self.bias

# norm

class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g

# helper classes

def FeedForward(*, dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        RMSNorm(dim),
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        flash = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.attend = Attend(
            causal = True,
            flash = flash,
            dropout = dropout
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, attn_bias = None):
        h, device = self.heads, x.device

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        out = self.attend(q, k, v, attn_bias = attn_bias)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        layers,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_mult = 4,
        rel_pos_bias = True,
        flash_attn = False
    ):
        super().__init__()
        self.alibi = Alibi(heads = heads) if rel_pos_bias else None
        self.layers = nn.ModuleList([])

        for _ in range(layers):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, flash = flash_attn),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))

        self.norm = RMSNorm(dim)

    def forward(self, x):
        n = x.shape[-2]
        attn_bias = self.alibi(n, n, device = x.device) if exists(self.alibi) else None

        for attn, ff in self.layers:
            x = attn(token_shift(x), attn_bias = attn_bias) + x
            x = ff(token_shift(x)) + x

        return self.norm(x)

# main class


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MEGABYTE(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        global_hidden_size,  # D_G
        global_num_hidden_layers,
        global_num_attention_heads,
        local_hidden_size,  # D_L
        local_num_hidden_layers,
        local_num_attention_heads,
        vocab_size=256,  # V
        max_position_embedding=8192,  # T
        patch_size=8,  # P
        attn_dropout=0.0,
        ff_mult=4,
        ff_dropout=0.0,
        pad_id=0,
        rel_pos_bias=True,
        flash_attn=False,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.pad = pad_id
        self.max_position_embedding = max_position_embedding

        self.pos_emb = nn.Embedding(max_position_embedding, global_hidden_size)
        self.global_emb = nn.Embedding(vocab_size, global_hidden_size)
        self.local_emb = nn.Embedding(vocab_size, local_hidden_size)

        self.global_to_local = nn.Linear(global_hidden_size, local_hidden_size)
        self.local_to_logits = nn.Linear(local_hidden_size, vocab_size)

        self.global_transformer = Transformer(
            dim=patch_size * global_hidden_size,
            layers=global_num_hidden_layers,
            dim_head=(patch_size * global_hidden_size) // global_num_attention_heads,
            heads=global_num_attention_heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            ff_mult=ff_mult,
            rel_pos_bias=rel_pos_bias,
            flash_attn=flash_attn,
        )
        self.local_transformer = Transformer(
            dim=local_hidden_size,
            layers=local_num_hidden_layers,
            dim_head=local_hidden_size // local_num_attention_heads,
            heads=local_num_attention_heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            ff_mult=ff_mult,
            rel_pos_bias=rel_pos_bias,
            flash_attn=flash_attn,
        )

        print(f"Params global: {count_trainable_params(self.global_transformer):,}")
        print(f"Params local : {count_trainable_params(self.local_transformer):,}")

    def generate(
        self, prime=None, filter_thres=0.9, temperature=1.0, default_batch_size=1
    ):
        total_seq_len = self.max_position_embedding
        device = next(self.parameters()).device

        if not exists(prime):
            prime = torch.empty(
                (default_batch_size, 0), dtype=torch.long, device=device
            )
        seq = prime

        for _ in tqdm(range(total_seq_len - seq.shape[-1])):
            seq_in = F.pad(seq, (0, 1), value=self.pad)
            logits = self.forward(seq_in)[:, -1]
            logits = top_k(logits, thres=filter_thres)
            sampled = gumbel_sample(logits, dim=-1, temperature=temperature)
            seq = torch.cat((seq, rearrange(sampled, "b -> b 1")), dim=-1)

        return seq

    def prepare_input(self, bytes):
        padding_global = bytes.new(bytes.shape[0], self.patch_size).fill_(self.pad)
        bytes_global = torch.cat((padding_global, bytes[:, : -self.patch_size]), -1)
        bytes_input = rearrange(bytes, "b (t p) -> (b t) p", p=self.patch_size)
        padding_local = bytes_input.new(bytes_input.shape[0], 1).fill_(self.pad)
        bytes_local = torch.cat((padding_local, bytes_input[:, :-1]), -1)
        return bytes_global, bytes_local

    def forward(self, bytes, return_loss=False):
        device = bytes.device
        batch_size, seq_len = bytes.shape

        padlen = remainder_to_mult(seq_len, self.patch_size)
        bytes_padded = F.pad(bytes, (0, padlen), value=self.pad)
        bytes_global, bytes_local = self.prepare_input(bytes_padded)

        global_bytes_embedded = self.global_emb(bytes_global)
        global_bytes_embedded += self.pos_emb(
            torch.arange(global_bytes_embedded.shape[1], device=device)
        )
        global_in = rearrange(
            global_bytes_embedded,
            "b (t p) e -> b t (p e)",
            p=self.patch_size,
        )
        global_out = self.global_transformer(global_in)
        global_out_reshaped = rearrange(
            global_out,
            "b t (p e) -> (b t) p e",
            p=self.patch_size,
        )

        local_in = self.global_to_local(global_out_reshaped)
        local_in += self.local_emb(bytes_local)
        local_out = self.local_transformer(local_in)
        local_out = rearrange(local_out, "(b t) l v -> b (t l) v", b=batch_size)

        logits = self.local_to_logits(local_out)
        logits = logits[:, :seq_len]
        if not return_loss:
            return logits

        logits = rearrange(logits, "b n c -> b c n")
        loss = F.cross_entropy(logits, bytes, ignore_index=self.pad)
        return loss
