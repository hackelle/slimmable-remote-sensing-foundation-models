import torch

# Sentinel-2A central wavelengths (nm) — same values as panopticon_helpers.PANOPTICON_S2_NM
CF_S2_NM = {
    "01": 442.7, "02": 492.4, "03": 559.8, "04": 664.6, "05": 704.1,
    "06": 740.5, "07": 782.8, "08": 832.8, "8A": 864.7, "09": 945.1,
    "10": 1373.5, "11": 1613.7, "12": 2202.4,
}
# Standard Sentinel-2 bandwidths (nm) — matches your demo's B,G,R,NIR = 65,35,30,115
CF_S2_BANDWIDTH_NM = {
    "01": 20, "02": 65, "03": 35, "04": 30, "05": 15,
    "06": 15, "07": 20, "08": 115, "8A": 20, "09": 20,
    "10": 30, "11": 90, "12": 180,
}


def copernicusFM_band_params(band_info, include_sar: bool = False):
    """Map a geobench band to (wavelength_nm, bandwidth_nm) for CopernicusFM.
    Returns None if the band should be dropped (SAR, or unknown band with
    no bandwidth available)."""
    names = [band_info.name] + list(getattr(band_info, "alt_names", ()) or ())
    for n in names:
        key = n.upper().replace("-", "").replace("_", "").replace(" ", "")
        if key in CF_S2_NM:
            return CF_S2_NM[key], CF_S2_BANDWIDTH_NM[key]
        if key.startswith(("VV", "VH", "HH", "HV")):
            return None  # SAR — skipped

    wl = getattr(band_info, "wavelength", None)   # geobench SpectralBand, micrometres
    bw = getattr(band_info, "bandwidth", None)    # UNVERIFIED: may not exist
    if wl is None or bw is None:
        return None
    wv_nm = float(wl) * 1000.0 if wl < 100 else float(wl)
    bw_nm = float(bw) * 1000.0 if bw < 100 else float(bw)
    return wv_nm, bw_nm


class CopernicusFMBackbone(torch.nn.Module):
    """Adapts CopernicusFM's (img, meta, wvs, bws, lang, mode, kernel) forward
    to the plain (B, C, H, W) -> features interface KnnBaseline expects."""

    def __init__(self, model, wvs, bws, kernel_size: int = 16, input_mode: str = "spectral"):
        super().__init__()
        self.model = model
        self.wvs = wvs
        self.bws = bws
        self.kernel_size = kernel_size
        self.input_mode = input_mode

    def get_all_features(self, val):
        if val:
            self.model.return_intermediate = True
            self.model.intermediate_indices = list(range(len(self.model.blocks)))  # all layers
        else:
            self.model.return_intermediate = False
            self.model.intermediate_indices = []
            
    def forward(self, x):
        b = x.shape[0]
        meta = torch.full((b, 4), float("nan"), device=x.device, dtype=x.dtype)
        logit, embed = self.model(x, meta, self.wvs, self.bws, None, self.input_mode, self.kernel_size)
        # ASSUMPTION: mirrors PanopticonBackbone "cls" mode.
        if isinstance(embed, list):
            return embed
        return [embed.unsqueeze(1) if embed.dim() == 2 else embed]