import torch
from functools import partial
from terratorch_slimable import make_terratorch_prithvi_dofa_slimable
from slimrsfms.helper import SlimStrategy

# Sentinel-2A central wavelengths in nm (matches the README example: 664/559/493 for RGB)
PANOPTICON_S2_NM = {
    "B01": 442.7, "B02": 492.4, "B03": 559.8, "B04": 664.6, "B05": 704.1,
    "B06": 740.5, "B07": 782.8, "B08": 832.8, "B8A": 864.7, "B09": 945.1,
    "B10": 1373.5, "B11": 1613.7, "B12": 2202.4,
}
# chn_ids = gaussian.mu from dinov2/configs/data/satellites/sentinel1.yaml (orbit BOTH)
PANOPTICON_SAR_ID = {"VV": -1.0, "VH": -2.0, "HH": -3.0, "HV": -4.0}


def panopticon_chn_id(band_info, include_sar: bool = False):
    """Map a geobench band to a Panopticon chn_id. Returns None if the band should be dropped."""
    names = [band_info.name] + list(getattr(band_info, "alt_names", ()) or ())
    for n in names:
        key = n.upper().replace("-", "").replace("_", "").replace(" ", "")
        if key in PANOPTICON_S2_NM:
            return PANOPTICON_S2_NM[key]
        if key == "B8A":
            return PANOPTICON_S2_NM["B8A"]
        for pol, cid in PANOPTICON_SAR_ID.items():
            if key.startswith(pol):
                return cid if include_sar else None
    wl = getattr(band_info, "wavelength", None)   # geobench SpectralBand, in micrometres
    if wl is None:
        return None
    return float(wl) * 1000.0 if wl < 100 else float(wl)


class PanopticonBackbone(torch.nn.Module):
    """Adapts Panopticon's dict-input API to the plain (B, C, H, W) tensor interface."""

    def __init__(self, vit, chn_ids, feature_mode: str = "blocks", n_blocks: int = 12):
        super().__init__()
        self.vit = vit
        self.feature_mode = feature_mode
        self.n_blocks = n_blocks
        self.register_buffer(
            "chn_ids", torch.tensor(chn_ids, dtype=torch.float32), persistent=False
        )

    def _to_dict(self, x):
        assert x.shape[1] == self.chn_ids.numel(), \
            f"got {x.shape[1]} channels but {self.chn_ids.numel()} chn_ids"
        return dict(imgs=x, chn_ids=self.chn_ids.unsqueeze(0).repeat(x.shape[0], 1))

    def forward(self, x):
        d = self._to_dict(x)
        if self.feature_mode == "cls":
            # KnnBaseline asserts a list/tuple, so wrap it
            return [self.vit(d).unsqueeze(1)]           # (B, 1, 768)

        v = self.vit
        h = v.patch_embed(d)        # PanopticonPE consumes the dict -> (B, N, C)
        h = v._pos_embed(h)         # prepends cls token + adds pos embed -> (B, 1+N, C)
        h = v.patch_drop(h)
        h = v.norm_pre(h)

        outs = []
        for blk in v.blocks:
            h = blk(h)
            outs.append(v.norm(h))  # matches timm's norm=True on intermediates
        return outs[-self.n_blocks:]


def make_panopticon_slimable(wrapper: PanopticonBackbone):
    """Slims the DINOv2 trunk. The PanopticonPE channel-attention stays at full width."""
    make_terratorch_prithvi_dofa_slimable(wrapper.vit)   # hits the `hasattr(model, 'blocks')` branch
    wrapper.slimmable_module_list = wrapper.vit.slimmable_module_list
    wrapper.current_scale = 1.0

    def set_to_scale(self, scale, strategy=SlimStrategy.FIRST, seed: int = 0):
        self.vit.set_to_scale(scale, strategy=strategy, seed=seed)
        self.current_scale = self.vit.current_scale

    wrapper.set_to_scale = partial(set_to_scale, wrapper)
    return wrapper