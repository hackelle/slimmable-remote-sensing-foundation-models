from functools import partial

import torch
from terratorch import BACKBONE_REGISTRY

from slimrsfms.helper import FFNExpert as SlimableFFNExpert
from slimrsfms.helper import SlimStrategy, compute_slim_indices
from slimrsfms.models.slimable_attention import Attention as SlimableAttention

SLIM_ATTN = True
SLIM_FFN = True

def make_terratorch_prithvi_dofa_slimable(model):
    """
    Replaces FFNs and ATTNs in transformer blocks in the prithvi or dofa trained model with Slimable FFN experts and Slimable ATTN.
    Takes the weights and shape of the previously existing layers at the same space.

    !!! Modifies the original model, does not return a copy !!!
    """
    slimmable_module_list = []
    blocks = model.vit_model.blocks if hasattr(model, 'vit_model') else model.blocks if hasattr(model, 'blocks') else model.dofa_model.blocks

    if SLIM_FFN:
        for i, old_mlp in [(i, b.mlp) for i, b in enumerate(blocks)]:
            new_mlp = SlimableFFNExpert(
                dim=old_mlp.fc1.in_features,
                dim_hidden=old_mlp.fc2.in_features,
                hidden_mult=None,
                mult_bias=False,
                prenorm=False,
                dropout=0.,
                act_layer=type(old_mlp.act),
                mixtral_like=False,
            )
            new_mlp.fc1.weight.data.copy_(old_mlp.fc1.weight.data)
            new_mlp.fc1.bias.data.copy_(old_mlp.fc1.bias.data)
            new_mlp.fc2.weight.data.copy_(old_mlp.fc2.weight.data)
            new_mlp.fc2.bias.data.copy_(old_mlp.fc2.bias.data)
            blocks[i].mlp = new_mlp
            slimmable_module_list.append(new_mlp)

    if SLIM_ATTN:
        for i, old_attn in [(i, b.attn) for i, b in enumerate(blocks)]:
            dim = old_attn.qkv.in_features
            new_attn = SlimableAttention(
                dim=dim,
                num_heads=old_attn.num_heads,
                qkv_bias=True,
                attn_drop=old_attn.attn_drop.p,
                proj_drop=old_attn.proj_drop.p,
                norm_layer=torch.nn.Identity,
            )
            new_attn.proj.weight.data.copy_(old_attn.proj.weight.data)
            new_attn.proj.bias.data.copy_(old_attn.proj.bias.data)
            new_attn.q.weight.data.copy_(old_attn.qkv.weight.data[:dim])
            new_attn.q.bias.data.copy_(old_attn.qkv.bias.data[:dim])
            new_attn.k.weight.data.copy_(old_attn.qkv.weight.data[dim:2 * dim])
            new_attn.k.bias.data.copy_(old_attn.qkv.bias.data[dim:2 * dim])
            new_attn.v.weight.data.copy_(old_attn.qkv.weight.data[-dim:])
            new_attn.v.bias.data.copy_(old_attn.qkv.bias.data[-dim:])
            blocks[i].attn = new_attn
            slimmable_module_list.append(new_attn)

    assert len(slimmable_module_list) > 0, "No modules were made slimable. Did you configure something wrong?"

    model.slimmable_module_list = slimmable_module_list
    model.current_scale = 1.0

    def set_to_scale(self, scale: float, strategy: SlimStrategy = SlimStrategy.FIRST, seed: int = 0):
        assert 0 < scale <= 1.0, f"Invalid scale {scale}, should be in (0, 1]"
        self.current_scale = scale
        for mod in self.slimmable_module_list:
            width = max(1, int(mod.max_hidden_dim * scale))
            indices = compute_slim_indices(mod.max_hidden_dim, width, strategy, seed=seed)
            mod.set_slim_indices(indices)

    model.set_to_scale = partial(set_to_scale, model)

    return model



def make_terratorch_terramind_slimable(model):
    """
    Replaces FFNs in transformer blocks in the terramind trained model with Slimable FFN experts.
    Takes the weights and shape of the previously existing FFNs at the same space.

    !!! Modifies the original model, does not return a copy !!!
    """
    slimmable_module_list = []

    if SLIM_FFN:
        for i, old_mlp in [(i, b.mlp) for i, b in enumerate(model.encoder)]:
            new_mlp = SlimableFFNExpert(
                dim=old_mlp.fc1.in_features,
                dim_hidden=old_mlp.fc2.in_features,
                hidden_mult=None,
                mult_bias=False,
                prenorm=False,
                dropout=0.,
                act_layer=type(old_mlp.act),
                mixtral_like=True,
                fc_bias=False
            )
            new_mlp.fc1.weight.data.copy_(old_mlp.fc1.weight.data)
            new_mlp.fc2.weight.data.copy_(old_mlp.fc2.weight.data)
            new_mlp.fc_noact.weight.data.copy_(old_mlp.fc3.weight.data)
            model.encoder[i].mlp = new_mlp
            slimmable_module_list.append(new_mlp)

    if SLIM_ATTN:
        for i, old_attn in [(i, b.attn) for i, b in enumerate(model.encoder)]:
            dim = old_attn.qkv.in_features
            new_attn = SlimableAttention(
                dim=dim,
                num_heads=old_attn.num_heads,
                qkv_bias=False,
                qk_norm=False,
                proj_bias=False,
                attn_drop=old_attn.attn_drop.p,
                proj_drop=old_attn.proj_drop.p,
                norm_layer=ValueError,  # should not be used
                fused_attn=False
            )
            new_attn.proj.weight.data.copy_(old_attn.proj.weight.data)
            new_attn.q.weight.data.copy_(old_attn.qkv.weight.data[:dim])
            new_attn.k.weight.data.copy_(old_attn.qkv.weight.data[dim:2 * dim])
            new_attn.v.weight.data.copy_(old_attn.qkv.weight.data[-dim:])
            model.encoder[i].attn = new_attn
            slimmable_module_list.append(new_attn)

    assert len(slimmable_module_list) > 0, "No modules were made slimable. Did you configure something wrong?"

    model.slimmable_module_list = slimmable_module_list
    model.current_scale = 1.0

    def set_to_scale(self, scale: float, strategy: SlimStrategy = SlimStrategy.FIRST, seed: int = 0):
        assert 0 < scale <= 1.0, f"Invalid scale {scale}, should be in (0, 1]"
        self.current_scale = scale
        for mod in self.slimmable_module_list:
            width = max(1, int(mod.max_hidden_dim * scale))
            indices = compute_slim_indices(mod.max_hidden_dim, width, strategy, seed=seed)
            mod.set_slim_indices(indices)

    model.set_to_scale = partial(set_to_scale, model)

    return model


if __name__ == "__main__":
    from terratorch_baselines import evaluated_models
    from fvcore.nn import FlopCountAnalysis


    def compare_model_outputs(m1, m2, input_shape, is_same=True):
        t = torch.ones(*input_shape)
        m1.eval()
        m2.eval()
        with torch.no_grad():
            res_1 = m1(t)
            res_2 = m2(t)
        assert len(res_1) == len(res_2)
        if len(res_1) > 12:
            # more layer accumulate more numerical drift due to kernel operation shifting
            atol = 1e-3
        else:
            # even smaller number of layers have numerical drift. Operational this makes no difference
            atol = 1e-4
        # for i, (a, b) in enumerate(zip(res_1, res_2)):
        #     max_abs = (a - b).abs().max()
        #     max_rel = max_abs / (a.abs().max() + 1e-6)
        #     if max_abs < atol:
        #         print(f"+ Layer {i} within tolerance: max_abs={max_abs:.2e}, max_rel={max_rel:.2e}")
        #     else:
        #         print(f"- Layer {i} exceeds tolerance: max_abs={max_abs:.2e}, max_rel={max_rel:.2e}")

        if is_same:
            res = [torch.allclose(res_1[i], res_2[i], atol=atol) for i in range(len(res_1))]
            assert all(res), res
        else:
            res = [not torch.allclose(res_1[i], res_2[i], atol=atol) for i in range(len(res_1))]
            assert any(res), res


    def compare_model_flops(m1, m2, input_shape, is_same=True):
        t = torch.ones(*input_shape)
        model_flops_forward = FlopCountAnalysis(m1, (t,))
        model_flops_forward.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        res_1 = model_flops_forward.total()
        model_flops_forward = FlopCountAnalysis(m2, (t,))
        model_flops_forward.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        res_2 = model_flops_forward.total()
        rel_flop_difference = 1 - res_1 / res_2
        if is_same:
            assert rel_flop_difference < 0.05, f"Flops are supposed to be equal (±5%) but are not: {res_1}, {res_2}, {rel_flop_difference:.4f}"
        else:
            assert res_1 < res_2, f"Flops of m1 are supposed to be smaller than m2 but are not: {res_1}, {res_2}"
        return res_1, res_2


    tested_models = [
        ('terramind', 12),
        ('terramind_large', 12),
        ('prithvi2_300', 6),
        ('prithvi2_600', 6),
        ('dofa', 10),
        ('dofa_large', 10),
    ]
    tested_scales = [0.001, 0.01, 0.1, 0.5, 1.0]

    for model_name, c in tested_models:
        # load the model
        model = BACKBONE_REGISTRY.build(**evaluated_models[model_name]["kwargs"])
        slim_model = BACKBONE_REGISTRY.build(**evaluated_models[model_name]["kwargs"])
        if 'terramind' in model_name:
            slim_model = make_terratorch_terramind_slimable(slim_model)
        elif 'prithvi' in model_name or 'dofa' in model_name:
            slim_model = make_terratorch_prithvi_dofa_slimable(slim_model)
        else:
            raise Exception(f"Unknown model {model_name}")

        print(f"Comparing {model_name} model with slimable model")
        print("  Test 1 (full) ... ", end='', flush=True)
        compare_model_outputs(slim_model, model, (1, c, 224, 224), is_same=True)
        compare_model_flops(slim_model, model, (1, c, 224, 224), is_same=True)
        print("OK\n  Test 2 (full) ... ", end='', flush=True)
        compare_model_outputs(slim_model, model, (1, c, 224, 224), is_same=True)
        compare_model_flops(slim_model, model, (1, c, 224, 224), is_same=True)
        print("OK\n  Test 3 (slim) ... ", end='', flush=True)
        messages = []
        for tested_scale in tested_scales:
            slim_model.set_to_scale(tested_scale)
            compare_model_outputs(slim_model, model, (1, c, 224, 224), is_same=tested_scale == 1)
            flops_1, flops_2 = compare_model_flops(slim_model, model, (1, c, 224, 224), is_same=tested_scale == 1)
            messages.append(f"    Flops of slim model (Scale {tested_scale}: {flops_1 / 1e9:.1f}G / {flops_2 / 1e9:.1f}G = {flops_1 / flops_2 * 100:.1f}%)")
        print("OK")
        for m in messages:
            print(m)
        slim_model.set_to_scale(1.0)
        print("  Test 4 (full) ... ", end='', flush=True)
        compare_model_outputs(slim_model, model, (1, c, 224, 224), is_same=True)
        compare_model_flops(slim_model, model, (1, c, 224, 224), is_same=True)
        print("OK\n")
