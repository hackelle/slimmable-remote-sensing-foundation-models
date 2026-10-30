# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import collections.abc
import math
from functools import partial
from itertools import repeat
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange
from functorch import vmap
from timm.models.vision_transformer import Block


class FourierExpansion(nn.Module):
    """A Fourier series-style expansion into a high-dimensional space.

    Attributes:
        lower (float): Lower wavelength.
        upper (float): Upper wavelength.
        assert_range (bool): Assert that the encoded tensor is within the specified wavelength
            range.
    """

    def __init__(self, lower: float, upper: float, assert_range: bool = True) -> None:
        """Initialise.

        Args:
            lower (float): Lower wavelength.
            upper (float): Upper wavelength.
            assert_range (bool, optional): Assert that the encoded tensor is within the specified
                wavelength range. Defaults to `True`.
        """
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.assert_range = assert_range

    def forward(self, x: torch.Tensor, d: int) -> torch.Tensor:
        """Perform the expansion.

        Adds a dimension of length `d` to the end of the shape of `x`.

        Args:
            x (:class:`torch.Tensor`): Input to expand of shape `(..., n)`. All elements of `x` must
                lie within `[self.lower, self.upper]` if `self.assert_range` is `True`.
            d (int): Dimensionality. Must be a multiple of two.

        Raises:
            AssertionError: If `self.assert_range` is `True` and not all elements of `x` are not
                within `[self.lower, self.upper]`.
            ValueError: If `d` is not a multiple of two.

        Returns:
            torch.Tensor: Fourier series-style expansion of `x` of shape `(..., n, d)`.
        """
        # If the input is not within the configured range, the embedding might be ambiguous!
        in_range = torch.logical_and(self.lower <= x.abs(), torch.all(x.abs() <= self.upper))
        in_range_or_zero = torch.all(
            torch.logical_or(in_range, x == 0)
        )  # Allow zeros to pass through.
        if self.assert_range and not in_range_or_zero:
            raise AssertionError(
                f"The input tensor is not within the configured range"
                f" `[{self.lower}, {self.upper}]`."
            )

        # We will use half of the dimensionality for `sin` and the other half for `cos`.
        if not (d % 2 == 0):
            raise ValueError("The dimensionality must be a multiple of two.")

        # Always perform the expansion with `float64`s to avoid numerical accuracy shenanigans.
        x = x.double()

        wavelengths = torch.logspace(
            math.log10(self.lower),
            math.log10(self.upper),
            d // 2,
            base=10,
            device=x.device,
            dtype=x.dtype,
        )
        prod = torch.einsum("...i,j->...ij", x, 2 * np.pi / wavelengths)
        encoding = torch.cat((torch.sin(prod), torch.cos(prod)), dim=-1)

        return encoding.float()  # Cast to `float32` to avoid incompatibilities.


def to_2tuple(x: Any) -> Tuple:
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    return tuple(repeat(x, 2))


def resize_abs_pos_embed(
        pos_embed: torch.Tensor,
        new_size: Tuple[int, int],
        old_size: Optional[Union[int, Tuple[int, int]]] = None,
        num_prefix_tokens: int = 1,
        interpolation: str = "bicubic",
        antialias: bool = True,
) -> torch.Tensor:
    """Resize absolute position embeddings to a target resolution via interpolation

    Modified from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/pos_embed.py

    Args:
        pos_embed: Position embeddings tensor of size [b, n, d]
        new_size: Target [height, width] of embedding
        old_size: Original [height, width] of embedding
        num_prefix_tokens: Number of non-spatial prefix tokens (eg. cls)
        interpolation: Resize interpolation type
        antialias: Whether to apply antialiasing resizing
    Returns:
        Resized pos_embed of size [b, n', d]
    """

    new_size = to_2tuple(new_size)
    new_ntok = new_size[0] * new_size[1]

    if not old_size:
        old_size = int(math.sqrt(pos_embed.shape[1] - num_prefix_tokens))  # type:ignore
    old_size = to_2tuple(old_size)

    # Return if no resize necessary
    if new_size == old_size:
        return pos_embed

    if num_prefix_tokens:
        posemb_prefix, pos_embed = (
            pos_embed[:, :num_prefix_tokens],
            pos_embed[:, num_prefix_tokens:],
        )
    else:
        posemb_prefix, pos_embed = None, pos_embed

    # Interpolate position embedding
    pos_embed = pos_embed.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    pos_embed = F.interpolate(
        pos_embed, size=new_size, mode=interpolation, antialias=antialias
    )
    pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, new_ntok, -1)

    # Add back extra prefix tokens
    if posemb_prefix is not None:
        pos_embed = torch.cat([posemb_prefix, pos_embed], dim=1)

    return pos_embed


def pi_resize_patch_embed(
        patch_embed: torch.Tensor,
        new_patch_size: Tuple[int, int],
        interpolation: str = "bicubic",
        antialias: bool = True,
):
    """Resample patch embedding weights to a target resolution via pseudo-inverse
    resizing.

    Based on:
        https://github.com/google-research/big_vision/blob/b00544b81f8694488d5f36295aeb7972f3755ffe/big_vision/models/proj/flexi/vit.py
        https://arxiv.org/abs/2212.08013

    Args:
        patch_embed: Patch embedding parameters of size [d, c, h, w]
        new_patch_size: Target [height, width] of embedding
        interpolation: Resize interpolation type
        antialias: Whether to apply antialiasing resizing
    Returns:
        Resized pos_embed of size [d, c h', w']
    """
    assert len(patch_embed.shape) == 4, "Patch embed kernel should be a 4D tensor"
    assert len(new_patch_size) == 2, "New patch size should only be (height, width)"

    old_patch_size = tuple(patch_embed.shape[2:])

    # Return original kernel if no resize is necessary
    if old_patch_size == new_patch_size:
        return patch_embed

    def resize(x: torch.Tensor, shape: Tuple[int, int]):
        x_resized = F.interpolate(
            x[None, None, ...],
            shape,
            mode=interpolation,
            antialias=antialias,
        )
        return x_resized[0, 0, ...]

    def calculate_pinv(old_shape: Tuple[int, int], new_shape: Tuple[int, int]):
        mat = []
        for i in range(np.prod(old_shape)):
            basis_vec = torch.zeros(old_shape)
            basis_vec[np.unravel_index(i, old_shape)] = 1.0
            mat.append(resize(basis_vec, new_shape).reshape(-1))
        resize_matrix = torch.stack(mat)
        return torch.linalg.pinv(resize_matrix)

    # Calculate pseudo-inverse of resize matrix
    resize_matrix_pinv = calculate_pinv(old_patch_size, new_patch_size)
    resize_matrix_pinv = resize_matrix_pinv.to(patch_embed.device)

    def resample_patch_embed(patch_embed: torch.Tensor):
        h, w = new_patch_size
        resampled_kernel = resize_matrix_pinv @ patch_embed.reshape(-1)
        return rearrange(resampled_kernel, "(h w) -> h w", h=h, w=w)

    v_resample_patch_embed = vmap(vmap(resample_patch_embed, 0, 0), 1, 1)

    return v_resample_patch_embed(patch_embed)


class TransformerWeightGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, num_heads=4, num_layers=1):
        super(TransformerWeightGenerator, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            activation="gelu",
            norm_first=False,
            batch_first=False,
            dropout=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Linear layer to map transformer output to desired weight shape
        self.fc_weight = nn.Linear(input_dim, output_dim)
        self.fc_bias = nn.Linear(input_dim, embed_dim)
        self.wt_num = 128
        self.weight_tokens = nn.Parameter(torch.empty([self.wt_num, input_dim]))
        self.bias_token = nn.Parameter(torch.empty([1, input_dim]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.weight_tokens, std=0.02)
        torch.nn.init.normal_(self.bias_token, std=0.02)

    def forward(self, x):
        # x should have shape [seq_len, batch, input_dim]
        pos_wave = x
        x = torch.cat([self.weight_tokens, pos_wave], dim=0)
        x = torch.cat([x, self.bias_token], dim=0)
        transformer_output = self.transformer_encoder(x)
        weights = self.fc_weight(transformer_output[self.wt_num: -1] + pos_wave)
        bias = self.fc_bias(
            transformer_output[-1]
        )  # Using the last output to generate bias
        return weights, bias


class FCResLayer(nn.Module):
    def __init__(self, linear_size=128):
        super(FCResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        # self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        # y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y
        return out


class Dynamic_MLP_OFA_spectral(nn.Module):
    """
    Input: channels of wavelength and bandwidth (normalized): List -> List
           kernel size of the depth-wise convolution: kernel_size, default 3x3
           wv_planes
           inplanes
    """

    def __init__(self, wv_planes, inter_dim=128, kernel_size=3, embed_dim=1024):
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self._num_kernel = self.kernel_size * self.kernel_size * self.embed_dim
        self.inter_dim = inter_dim
        self.patch_size = (kernel_size, kernel_size)
        self.num_patches = -1

        ## CopernicusFM: fourier embedding for wavelength and bandwidth
        # min wavelength: ultraviolet light (100 nm)
        # max wavelength: radio waves (1 m)
        self.spectrum_central_expansion = FourierExpansion(100, 1e9)
        # min bandwidth: s2 ~ 10nm
        # max bandwidth: s1 ~ 1m
        self.spectrum_bandwidth_expansion = FourierExpansion(1, 1e9)

        self.weight_generator = TransformerWeightGenerator(
            wv_planes, self._num_kernel, embed_dim
        )
        self.scaler = 0.01

        self.fclayer = FCResLayer(wv_planes)

        self._init_weights()

    def _get_weights(self, waves):
        dynamic_weights = self.weight_generator(waves)

        return dynamic_weights

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _init_weights(self):
        """
        initialize the base weights and dynamic mlp weights
        """
        self.weight_generator.apply(self.weight_init)
        self.fclayer.apply(self.weight_init)

    def forward(self, img_feat, wvs, bandwidths, kernel_size=None):
        """
        wvs: nm
        bandwidths: nm
        """
        inplanes = wvs.size(0)
        # wv_feats: 9,128 -> 9, 3x3x3
        # waves = get_1d_sincos_pos_embed_from_grid_torch(self.wv_planes, wvs * 1000) # dofa: fixed sincos pos embedding
        # waves = get_1d_fourier_pos_embed_from_grid_torch(self.wv_planes, wvs * 1000) # new: fourier pos embedding
        emb_central = self.spectrum_central_expansion(wvs, self.wv_planes)
        emb_bandwidth = self.spectrum_bandwidth_expansion(bandwidths, self.wv_planes)
        waves = emb_central + emb_bandwidth  # simply add two embeddings, can be more complex later

        waves = self.fclayer(waves)
        weight, bias = self._get_weights(waves)  # 3x3x3

        # Fix bug
        dynamic_weight = weight.view(inplanes, self.kernel_size, self.kernel_size, self.embed_dim)  # 9, 3, 3, 1024
        dynamic_weight = dynamic_weight.permute([3, 0, 1, 2])  # 1024, 9, 3, 3

        # resize the weight to match different preferred kernel sizes
        if kernel_size != None and self.kernel_size != kernel_size:
            dynamic_weight = pi_resize_patch_embed(dynamic_weight, (kernel_size, kernel_size))
        else:
            kernel_size = self.kernel_size

        if bias is not None:
            bias = bias.view([self.embed_dim]) * self.scaler

        weights = dynamic_weight * self.scaler

        dynamic_out = F.conv2d(img_feat, weights, bias=bias, stride=kernel_size, padding=1, dilation=1)

        x = dynamic_out
        x = x.flatten(2).transpose(1, 2)

        return x, waves


class Dynamic_MLP_OFA_variable(nn.Module):
    """
    Input: language embedding of variable name: Pytorch tensor
           kernel size of the depth-wise convolution: kernel_size, default 3x3
           wv_planes
           inplanes
    """

    def __init__(self, wv_planes, inter_dim=128, kernel_size=3, embed_dim=1024):
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self._num_kernel = self.kernel_size * self.kernel_size * self.embed_dim
        self.inter_dim = inter_dim
        self.patch_size = (kernel_size, kernel_size)
        self.num_patches = -1

        self.language_proj = nn.Linear(2048, self.wv_planes)  # project to the same dimension as wv_planes

        self.weight_generator = TransformerWeightGenerator(
            wv_planes, self._num_kernel, embed_dim
        )
        self.scaler = 0.01

        self.fclayer = FCResLayer(wv_planes)

        self._init_weights()

    def _get_weights(self, waves):
        dynamic_weights = self.weight_generator(waves)

        return dynamic_weights

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _init_weights(self):
        """
        initialize the base weights and dynamic mlp weights
        """
        self.weight_generator.apply(self.weight_init)
        self.fclayer.apply(self.weight_init)

    def forward(self, img_feat, language_embed, kernel_size=None):
        """
        wvs: nm
        bandwidths: nm
        """
        # wv_feats: 9,128 -> 9, 3x3x3
        emb_language = language_embed.unsqueeze(0)
        waves = self.language_proj(emb_language)
        # print(waves.size())

        waves = self.fclayer(waves)
        # print(waves.size())
        weight, bias = self._get_weights(waves)  # 3x3x3

        # inplanes = wvs.size(0)
        inplanes = waves.size(0)
        # print(inplanes)
        # Fix bug
        dynamic_weight = weight.view(inplanes, self.kernel_size, self.kernel_size, self.embed_dim)  # 9, 3, 3, 1024
        dynamic_weight = dynamic_weight.permute([3, 0, 1, 2])  # 1024, 9, 3, 3

        # resize the weight to match different preferred kernel sizes
        if kernel_size != None and self.kernel_size != kernel_size:
            dynamic_weight = pi_resize_patch_embed(dynamic_weight, (kernel_size, kernel_size))
        else:
            kernel_size = self.kernel_size

        if bias is not None:
            bias = bias.view([self.embed_dim]) * self.scaler

        weights = dynamic_weight * self.scaler

        dynamic_out = F.conv2d(img_feat, weights, bias=bias, stride=kernel_size, padding=1, dilation=1)

        x = dynamic_out
        x = x.flatten(2).transpose(1, 2)

        return x, waves


class CopernicusFMViT(nn.Module):
    """CopernicusFM: VisionTransformer backbone"""

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            drop_rate=0.0,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            wv_planes=128,
            num_classes=0,
            global_pool=True,
            mlp_ratio=4.0,
            norm_layer=nn.LayerNorm,
            loc_option='lonlat',
            return_intermediate=False,
            intermediate_indices=None,
    ):
        super().__init__()

        self.wv_planes = wv_planes
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = norm_layer
            embed_dim = embed_dim
            self.fc_norm = norm_layer(embed_dim)
        else:
            self.norm = norm_layer(embed_dim)

        self.patch_embed_spectral = Dynamic_MLP_OFA_spectral(wv_planes=128, inter_dim=128, kernel_size=16, embed_dim=embed_dim)
        self.patch_embed_variable = Dynamic_MLP_OFA_variable(wv_planes=128, inter_dim=128, kernel_size=16, embed_dim=embed_dim)

        self.num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # ---------------------------------------------------------------------------

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.loc_option = loc_option
        if loc_option == 'cartesian':
            self.coord_expansion = FourierExpansion(1e-7, 2)
        elif loc_option == 'lonlat':
            self.coord_expansion = FourierExpansion(0.0001, 720)

        self.scale_expansion = FourierExpansion(0.001, 5.1e8)  # 1m2 to 5.1e8 km2
        self.time_expansion = FourierExpansion(1, 365.25, assert_range=False)  # 1 to 365.25 days, enable more than 1 year
        self.coord_fc = nn.Linear(embed_dim, embed_dim)
        self.scale_fc = nn.Linear(embed_dim, embed_dim)
        self.time_fc = nn.Linear(embed_dim, embed_dim)
        # if meta info is not available, set to a learned parameter
        self.coord_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.scale_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.time_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.head_drop = nn.Dropout(drop_rate)
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        self.return_intermediate = return_intermediate
        self.intermediate_indices = intermediate_indices

    def get_coord_pos_embed(self, lons, lats, embed_dim):
        if self.loc_option == 'cartesian':
            # convert to spherical coordinates
            spherical_x = torch.cos(lons * math.pi / 180) * torch.cos(lats * math.pi / 180) + 1 + 1e-7
            spherical_y = torch.sin(lons * math.pi / 180) * torch.cos(lats * math.pi / 180) + 1 + 1e-7
            spherical_z = torch.sin(lats * math.pi / 180) + 1 + 1e-7
            coord_embed_spherical_x = self.coord_expansion(spherical_x, embed_dim // 3)
            coord_embed_spherical_y = self.coord_expansion(spherical_y, embed_dim // 3)
            coord_embed_spherical_z = self.coord_expansion(spherical_z, embed_dim // 3)
            coord_embed = torch.cat([coord_embed_spherical_x, coord_embed_spherical_y, coord_embed_spherical_z], dim=-1)  # [B,D]
        elif self.loc_option == 'lonlat':
            coord_embed_lon = self.coord_expansion(lons + 180, embed_dim // 2)
            coord_embed_lat = self.coord_expansion(lats + 90, embed_dim // 2)
            coord_embed = torch.cat([coord_embed_lon, coord_embed_lat], dim=-1)

        if coord_embed.shape[-1] < embed_dim:
            # pad zeros
            coord_embed = torch.cat((coord_embed, torch.zeros(coord_embed.shape[0], embed_dim - coord_embed.shape[-1], device=coord_embed.device)), dim=-1)

        return coord_embed.unsqueeze(1)  # [B,1,D]

    def get_area_pos_embed(self, areas, embed_dim):
        scale_embed = self.scale_expansion(areas, embed_dim)  # B, D
        return scale_embed.unsqueeze(1)  # [B,1,D]

    def get_time_pos_embed(self, times, embed_dim):
        time_embed = self.time_expansion(times, embed_dim)  # B, D
        return time_embed.unsqueeze(1)  # [B,1,D]

    def forward_features(self, x, meta_info, wave_list, bandwidth, language_embed, input_mode, kernel_size=None):
        # embed patches
        if input_mode == 'spectral':
            wavelist = torch.tensor(wave_list, device=x.device).float()
            bandwidths = torch.tensor(bandwidth, device=x.device).float()
            self.waves = wavelist
            x, _ = self.patch_embed_spectral(x, self.waves, bandwidths, kernel_size)
        elif input_mode == 'variable':
            x, _ = self.patch_embed_variable(x, language_embed, kernel_size)

        # resize pos embed
        num_patches = x.size(1)
        num_patches_sqrt = int(math.sqrt(num_patches))
        num_patches_sqrt_origin = int(math.sqrt(self.num_patches))
        pos_embed = resize_abs_pos_embed(self.pos_embed, num_patches_sqrt, (num_patches_sqrt_origin, num_patches_sqrt_origin), num_prefix_tokens=1)

        # coord, scale and time pos embed
        lons, lats, times, areas = meta_info[:, 0], meta_info[:, 1], meta_info[:, 2], meta_info[:, 3]
        embed_dim = pos_embed.shape[-1]
        if torch.isnan(lons).any() or torch.isnan(lats).any():
            coord_embed = self.coord_token
        else:
            coord_embed = self.get_coord_pos_embed(lons, lats, embed_dim)
        coord_embed = self.coord_fc(coord_embed)
        if torch.isnan(areas).any():
            area_embed = self.scale_token
        else:
            area_embed = self.get_area_pos_embed(areas, embed_dim)
        area_embed = self.scale_fc(area_embed)
        if torch.isnan(times).any():
            time_embed = self.time_token
        else:
            time_embed = self.get_time_pos_embed(times, embed_dim)
        time_embed = self.time_fc(time_embed)
        pos_embed = pos_embed + coord_embed + area_embed + time_embed

        # add pos embed w/o cls token
        x = x + pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        intermediate_features = []
        hw = num_patches_sqrt
        hw_shape = (hw, hw)

        # apply Transformer blocks
        for i, block in enumerate(self.blocks):
            x = block(x)
            if self.return_intermediate and (i in self.intermediate_indices):
                out = x[:, 1:]
                B, _, C = out.shape
                out = (
                    out.reshape(B, hw_shape[0], hw_shape[1], C)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                intermediate_features.append(out)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        if self.return_intermediate:
            return outcome, intermediate_features

        return outcome

    def forward_head(self, x, pre_logits=False):
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x, meta_info, wave_list, bandwidth, language_embed, input_mode, kernel_size=None):
        if self.return_intermediate:
            x, intermediate_features = self.forward_features(x, meta_info, wave_list, bandwidth, language_embed, input_mode, kernel_size)
            return x, intermediate_features
        else:
            fx = self.forward_features(x, meta_info, wave_list, bandwidth, language_embed, input_mode, kernel_size)
            x = self.forward_head(fx)
            return x, fx


def vit_small_patch16(**kwargs):
    model = CopernicusFMViT(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_base_patch16(**kwargs):
    model = CopernicusFMViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_large_patch16(**kwargs):
    model = CopernicusFMViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_huge_patch14(**kwargs):
    model = CopernicusFMViT(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model