# based on https://raw.githubusercontent.com/lucidrains/CoLT5-attention/main/colt5_attention/coor_descent.py


from collections import namedtuple
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, pack, unpack
from torch.amp import autocast
from enum import Enum


# ===== NORMAL HELPER =====
def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val

    return d() if callable(d) else d


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def divisible_by(num, den):
    return (num % den) == 0


def chunk_num(num, chunks):
    num_per_chunk, remainder = divmod(num, chunks)

    out = []
    for i in range(chunks):
        n = num_per_chunk
        out.append(n + int(i < remainder))

    return out


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def cast_tuple(el, length: int = 1):
    return el if isinstance(el, tuple) else ((el,) * length)


def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))


# ===== TENSOR HELPER =====
def cumsum_exclusive(t, dim=-3):
    assert dim < 0
    num_pad_dims = -dim - 1
    pre_padding = (0, 0) * num_pad_dims
    return F.pad(t, (*pre_padding, 1, -1)).cumsum(dim=dim)


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def l2norm(t):
    return F.normalize(t, dim=- 1)


class GEGLU(nn.Module):
    def __init__(
            self,
            dim,
            hidden_dim=None,  # UNUSED, but added for compatibility
            mult_bias=True
    ):
        super().__init__()
        self.mult_bias = nn.Parameter(torch.ones(dim)) if mult_bias else torch.ones(1)

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        x_last_dim = x.shape[-1]
        return F.gelu(gate) * x * self.mult_bias[..., :x_last_dim]

    @classmethod
    def is_geglu(cls, obj) -> bool:
        """
        Check if the object is GEGLU or partial of GEGLU
        """
        return (isinstance(obj, cls) or
                obj == cls or
                hasattr(obj, 'func') and obj.func == cls or
                hasattr(obj, 'func') and obj.func == cls.func)


class SlimStrategy(Enum):
    FIRST = "first"
    LAST = "last"
    RANDOM = "random"


def compute_slim_indices(max_dim: int, target_dim: int, strategy: SlimStrategy, seed: int = 0) -> torch.Tensor:
    """Return a sorted 1D index tensor of length target_dim."""
    assert 0 < target_dim <= max_dim
    if strategy == SlimStrategy.FIRST:
        return torch.arange(target_dim)
    elif strategy == SlimStrategy.LAST:
        return torch.arange(max_dim - target_dim, max_dim)
    elif strategy == SlimStrategy.RANDOM:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(max_dim, generator=g)
        selected = perm[:target_dim]
        return selected.sort().values  # sort so heads stay ordered
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# expert class
class FFNExpert(nn.Module):
    def __init__(
            self,
            dim: int,
            dim_hidden: Optional[int] = None,
            hidden_mult: float = 4.,
            mult_bias: bool = True,
            prenorm: bool = False,
            dropout: float = 0.,
            act_layer: Callable = GEGLU,
            mixtral_like: bool = False,
            fc_bias: bool = True,
    ):
        super().__init__()
        if dim_hidden is None:
            dim_hidden = int(dim * hidden_mult * 2 / 3)
        if GEGLU.is_geglu(act_layer):
            dim_factor = 2
        else:
            dim_factor = 1

        self.pre_norm = RMSNorm(dim) if prenorm else nn.Identity()
        self.fc1 = nn.Linear(dim, dim_hidden * dim_factor, bias=fc_bias)
        self.act = act_layer(dim_hidden, mult_bias=mult_bias) if act_layer == GEGLU else act_layer()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim_hidden, dim, bias=fc_bias)
        if mixtral_like:
            self.fc_noact = nn.Linear(dim, dim_hidden, bias=fc_bias)

        self.max_hidden_dim = dim_hidden
        # default: use all dims (equivalent to scale=1.0)
        self.register_buffer('_slim_indices', torch.arange(dim_hidden), persistent=False)

        self.apply(self.init_)

    def init_(self, module):
        if isinstance(module, nn.Linear):
            dim = module.weight.shape[0]
            std = dim ** -0.5

            module.weight.data.uniform_(-std, std)
            if module.bias is not None:
                module.bias.data.uniform_(-std, std)

    @property
    def slimmable_hidden_dim(self):
        return len(self._slim_indices)

    @slimmable_hidden_dim.setter
    def slimmable_hidden_dim(self, value):
        # Backward compat: setting an int uses FIRST strategy
        self._slim_indices = torch.arange(value, device=self._slim_indices.device)

    def set_slim_indices(self, indices: torch.Tensor):
        """Set arbitrary indices directly."""
        self._slim_indices = indices.to(self._slim_indices.device)

    def forward(self, x):
        x = self.pre_norm(x)
        indices = self._slim_indices

        # For GEGLU, we need paired indices (each hidden dim maps to 2 entries in fc1)
        if isinstance(self.act, GEGLU):
            fc1_indices = torch.cat([indices * 2, indices * 2 + 1]).sort().values
        else:
            fc1_indices = indices

        x1 = self.act(F.linear(
            x,
            self.fc1.weight[fc1_indices, :],
            self.fc1.bias[fc1_indices] if self.fc1.bias is not None else None
        ))

        if hasattr(self, 'fc_noact'):
            x2 = F.linear(
                x,
                self.fc_noact.weight[indices, :],
                self.fc_noact.bias[indices] if self.fc_noact.bias is not None else None
            )
            x = x1 * x2
        else:
            x = x1

        x = self.dropout(x)
        x = F.linear(
            x,
            self.fc2.weight[:, indices],
            self.fc2.bias if self.fc2.bias is not None else None
        )
        return x


# pytorch one hot throws an error if there are out of bound indices.
# tensorflow, in contrast, does not throw an error
def safe_one_hot(indexes, max_length):
    max_index = indexes.max() + 1
    one_hot_classes = max(max_index + 1, max_length)
    return F.one_hot(indexes, one_hot_classes)[..., :max_length]


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return l2norm(x) * self.scale * self.gamma


@autocast(enabled=False, device_type='cuda')
def coor_descent(
        s,
        *,
        n_iters,
        k,
        eps=1e-1,
        eps_init=None,
        eps_decay=1.,
        mask=None
):
    """
    coordinate descent  - https://arxiv.org/abs/1502.04759, utilized in https://arxiv.org/abs/2303.09752
    ε-scaling           - https://arxiv.org/abs/1610.06519, utilized in https://arxiv.org/abs/2304.04947

    in a follow up paper applying coordinate descent routing to efficient fine tuning
    they were able to cut n_iters from 50 -> 20 by setting eps_init = 4 and eps_decay = 0.7
    eps was dependent on the task, and ranged from 0.02 to 1
    """

    assert n_iters > 0

    mask_value = -torch.finfo(s.dtype).max

    if not isinstance(k, torch.Tensor):
        k = torch.Tensor([k]).to(s)
    else:
        k = rearrange(k, '... -> ... 1')

    logk = log(k)

    if exists(mask):
        s = s.masked_fill(~mask, mask_value)

    a = 0
    b = -s

    current_eps = max(default(eps_init, eps), eps)

    for _ in range(n_iters):
        sb = ((s + b) / current_eps)

        if exists(mask):
            sb = sb.masked_fill(~mask, mask_value)

        a = current_eps * (logk - sb.logsumexp(dim=-1, keepdim=True))
        b = -F.relu(s + a)

        current_eps = max(current_eps * eps_decay, eps)

    scores = ((s + a + b) / current_eps).exp()

    if exists(mask):
        scores = scores.masked_fill(~mask, 0.)

    return scores


TopkReturn = namedtuple('TopkReturn', ['values', 'indices', 'coor_descent_values', 'gates'])


@autocast(enabled=False, device_type='cuda')
def topk(
        x,
        k,
        coor_descent_k_ratio=9 / 8,
        n_iters=20,
        eps=1e-1,
        eps_init=None,
        eps_decay=1.,
        mask=None,
        fused=False,
        non_differentiable=False
):
    """
    differentiable top-k on last dimension
    """

    if non_differentiable:
        values, indices = torch.topk(x, k=k, dim=-1)
        return TopkReturn(values, indices, None, None)

    assert coor_descent_k_ratio >= 1.
    assert k > 0

    # whether to used fused kernel or not

    fn = coor_descent

    if fused and x.is_cuda:
        from colt5_attention.triton_coor_descent import triton_coor_descent
        fn = triton_coor_descent

    # do coordinate descent for gradients

    coor_descent_out = fn(
        x,
        k=min(k * coor_descent_k_ratio, x.shape[-1]),
        # fetch a bit more for better learning, as in CoLT5 paper (they fetched 9 / 8 times more)
        mask=mask,
        n_iters=n_iters,
        eps=eps,
        eps_init=eps_init,
        eps_decay=eps_decay
    )

    # do straight through

    gates = coor_descent_out + (1 - coor_descent_out).detach()

    x = x * gates

    # hard topk

    values, indices = torch.topk(x, k, dim=-1)

    # return something that looks like a usual topk, but now differentiable

    coor_descent_values = coor_descent_out.gather(-1, indices)
    gates = gates.gather(-1, indices)

    return TopkReturn(values, indices, coor_descent_values, gates)


if __name__ == "__main__":
    # test slimmable FFNExpert
    model = FFNExpert(dim=3, dim_hidden=6, act_layer=nn.ReLU)
    x = torch.randn(2, 4, 3)
    # forward with full hidden dim
    model.zero_grad()
    # print all gradients
    print(">>> After zero_grad")
    for n, p in model.named_parameters():
        print(n, "\n", p.grad)
    print()
    _ = model(x)

    # fake a loss
    loss = _.sum()
    loss.backward()
    print(">>> After first backward")
    for n, p in model.named_parameters():
        print(n, "\n", p.grad)
    print()

    # set slimmable dim to 2
    model.slimmable_hidden_dim = 2
    _ = model(x)
    loss = _.sum()
    loss.backward()
    print(">>> After second backward with slimmable dim = 2")
    for n, p in model.named_parameters():
        print(n, "\n", p.grad)
    print()

    # set slimmable dim to 4
    model.slimmable_hidden_dim = 4
    _ = model(x)
    loss = _.sum()
    loss.backward()
    print(">>> After third backward with slimmable dim = 4")
    for n, p in model.named_parameters():
        print(n, "\n", p.grad)
    print()

    # now with slimmable dim = 2 but with zero grad
    model.slimmable_hidden_dim = 2
    model.zero_grad()
    _ = model(x)
    loss = _.sum()
    loss.backward()
    print(">>> After backward with slimmable dim = 2 and zero_grad")
    for n, p in model.named_parameters():
        print(n, "\n", p.grad)
    print()
