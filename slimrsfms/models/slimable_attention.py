from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import use_fused_attn


class Attention(nn.Module):
    """ Multi-Head Self Attention module with support for slimmable attention heads. Adopted from timm."""

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            fused_attn: bool = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.fused_attn = use_fused_attn() if fused_attn is None else fused_attn

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.max_hidden_dim = self.head_dim
        self.register_buffer('_slim_indices', torch.arange(self.head_dim), persistent=False)

    def set_to_slimmable_dim(self, hidden_dim: int) -> None:
        assert 0 < hidden_dim <= self.max_hidden_dim, \
            f"Invalid hidden dim: {hidden_dim}, should be in range (0, {self.max_hidden_dim}]"
        self.slimmable_hidden_dim = hidden_dim

    @property
    def slimmable_hidden_dim(self):
        return len(self._slim_indices)

    @slimmable_hidden_dim.setter
    def slimmable_hidden_dim(self, value):
        self._slim_indices = torch.arange(value, device=self._slim_indices.device)

    def set_slim_indices(self, indices: torch.Tensor):
        self._slim_indices = indices.to(self._slim_indices.device)

    def forward(self, x, mask=None):
        if mask is not None:
            raise NotImplementedError("Masking not implemented for Slimable Attention")
        B, N, C = x.shape

        # Build full-dim indices: for each head, offset by head_start + per-head indices
        head_indices = []
        for h in range(self.num_heads):
            start = h * self.max_hidden_dim
            head_indices.append(start + self._slim_indices)
        head_indices = torch.cat(head_indices, dim=0)

        q_weight = self.q.weight[head_indices, :]
        k_weight = self.k.weight[head_indices, :]
        v_weight = self.v.weight[head_indices, :]

        q = torch.einsum('...nd,hd->...nh', x, q_weight)
        k = torch.einsum('...nd,hd->...nh', x, k_weight)
        v = torch.einsum('...nd,hd->...nh', x, v_weight)

        if self.q.bias is not None:
            q += self.q.bias[head_indices]
            k += self.k.bias[head_indices]
            v += self.v.bias[head_indices]

        q = q.reshape(B, N, self.num_heads, self.slimmable_hidden_dim).permute(0, 2, 1, 3)  # B, num_heads, N, head_dim
        k = k.reshape(B, N, self.num_heads, self.slimmable_hidden_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.slimmable_hidden_dim).permute(0, 2, 1, 3)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * (self.slimmable_hidden_dim ** -0.5)
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, self.num_heads * self.slimmable_hidden_dim)
        x = torch.einsum('...nd,hd->...nh', x, self.proj.weight[:, head_indices])
        if self.proj.bias is not None:
            x += self.proj.bias
        x = self.proj_drop(x)
        return x


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis
    from timm.models.vision_transformer import Attention as StandardAttention

    dim = 128
    n_tokens = 3 * 8 * 8 + 1
    heads = 8
    attn = Attention(dim=dim, num_heads=heads)
    reference_attn = StandardAttention(dim=dim, num_heads=heads)
    x = torch.randn(1, n_tokens, dim)
    print(" in shape:", x.shape)
    out = attn(x)
    print("out shape:", out.shape)

    for with_fused_attn in [True, False]:
        print(f"With{'' if with_fused_attn else 'out'} fused attention:")
        attn.fused_attn = with_fused_attn
        reference_attn.fused_attn = with_fused_attn

        fca = FlopCountAnalysis(reference_attn, (x,))
        fca.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        print(f"   ref FLOPs: {fca.total():,}")

        attn.set_to_slimmable_dim(int(attn.max_hidden_dim * 1.0))
        fca = FlopCountAnalysis(attn, (x,))
        fca.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        print(f"       FLOPs: {fca.total():,}")

        attn.set_to_slimmable_dim(int(attn.max_hidden_dim * 0.5))
        fca = FlopCountAnalysis(attn, (x,))
        fca.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        print(f"  0.5x FLOPs: {fca.total():,}")

        attn.set_to_slimmable_dim(int(attn.max_hidden_dim * 0.1))
        fca = FlopCountAnalysis(attn, (x,))
        fca.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        print(f"  0.1x FLOPs: {fca.total():,}")
        print()
