"""ShuffleUp3d: exact replacement for nnU-Net's ConvTranspose3d(kernel_size == stride, padding 0).

A transposed conv whose kernel equals its stride maps every input voxel to one s_z*s_y*s_x
block of output voxels with no overlap, so it is exactly a 1x1x1 Conv3d to
out_ch * prod(stride) channels followed by depth-to-space. Same MACs, exact in arithmetic,
bit-exact in fp32 where the backends sum alike (measured: max |diff| 0.0 on TS models) and not
everywhere: on the CPU the 1x1 conv sums its input channels in another order, 2.9e-6 apart in
fp32 on a tiny ResEnc (2026-10-18, tests/test_multifold_surgery.py). -11 % even in fp32 on MPS,
and - the point - runs in fp16 / bf16 on MPS, which refuses ConvTranspose3d in half. Applied as
load-time network surgery by :class:`haversack.TorchModel`.
"""
from __future__ import annotations

import math

import torch


def as_shuffled(weight: torch.Tensor, bias: torch.Tensor | None = None):
    """A ConvTranspose3d(kernel == stride)'s ``weight`` and ``bias`` as :class:`ShuffleUp3d`'s
    1x1x1 conv holds them. The ONE statement of the mapping: the surgery itself and
    :func:`surgered_state` (every fold after the first, 2026-10-18) both call it, so the
    network built at load and the folds loaded into it later cannot disagree."""
    ic, oc = weight.shape[:2]                                                   # (ic, oc, kz, ky, kx)
    n_taps = math.prod(weight.shape[2:])
    w = weight.detach().permute(1, 2, 3, 4, 0).reshape(oc * n_taps, ic, 1, 1, 1)
    return w, (None if bias is None else bias.detach().repeat_interleave(n_taps))


class ShuffleUp3d(torch.nn.Module):
    def __init__(self, t: torch.nn.ConvTranspose3d):
        super().__init__()
        k, s = tuple(t.kernel_size), tuple(t.stride)
        if not (k == s and tuple(t.padding) == (0, 0, 0) and tuple(t.output_padding) == (0, 0, 0)
                and t.groups == 1 and tuple(t.dilation) == (1, 1, 1)):
            raise ValueError(f"ShuffleUp3d needs kernel == stride, no padding, groups 1: got k={k} s={s} "
                             f"p={t.padding} op={t.output_padding} g={t.groups} d={t.dilation}")
        ic, oc = t.in_channels, t.out_channels
        self.s, self.oc = s, oc
        n_taps = s[0] * s[1] * s[2]
        self.conv = torch.nn.Conv3d(ic, oc * n_taps, kernel_size=1, bias=t.bias is not None,
                                    device=t.weight.device, dtype=t.weight.dtype)
        w, b = as_shuffled(t.weight, t.bias)
        with torch.no_grad():
            self.conv.weight.copy_(w)
            if b is not None:
                self.conv.bias.copy_(b)

    def forward(self, x):
        n, _, Z, Y, X = x.shape
        sz, sy, sx = self.s
        u = self.conv(x).view(n, self.oc, sz, sy, sx, Z, Y, X)
        u = u.permute(0, 1, 5, 2, 6, 3, 7, 4)                                   # (n, oc, Z, sz, Y, sy, X, sx)
        return u.reshape(n, self.oc, Z * sz, Y * sy, X * sx)


def swap_transposed(module: torch.nn.Module) -> int:
    """Replace every ConvTranspose3d under `module` in place. Returns the count."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.ConvTranspose3d):
            setattr(module, name, ShuffleUp3d(child)); n += 1
        else:
            n += swap_transposed(child)
    return n


def surgered_state(state: dict, net: torch.nn.Module) -> dict:
    """A checkpoint's state dict - written for the architecture BEFORE :func:`swap_transposed` -
    renamed and reshaped for ``net`` after it: each swapped module's ``weight``/``bias`` become
    its ``conv.weight``/``conv.bias`` through :func:`as_shuffled`; every other entry is passed
    through as the same tensor (no copy).

    Why it exists: the surgery is built from the weights the network holds at load, which are
    fold 0's, and the other folds' checkpoints still hold ConvTranspose3d tensors under the old
    names. Until 2026-10-18 TorchModel refused every fold after the first with surgery on (the
    default), so no multi-fold model ran at all; no shipped catalog had one until LNQ.

    Pure apart from reading ``net``'s module names and strides; the result is checked by
    ``load_state_dict``'s strict key match when a fold is loaded. A checkpoint whose kernel is not
    the module's stride is refused here rather than reshaped into a wrong network.
    """
    out = dict(state)
    for name, m in net.named_modules():
        if not isinstance(m, ShuffleUp3d):
            continue
        w = out.pop(f"{name}.weight", None)
        if w is None:
            raise ValueError(f"checkpoint has no {name}.weight for the swapped ConvTranspose3d {name}")
        if tuple(w.shape[2:]) != tuple(m.s) or w.shape[1] != m.oc:
            raise ValueError(f"checkpoint's {name}.weight is {tuple(w.shape)}; the network's "
                             f"ConvTranspose3d had {m.oc} outputs and kernel == stride {m.s}")
        b = out.pop(f"{name}.bias", None)
        out[f"{name}.conv.weight"], cb = as_shuffled(w, b)
        if cb is not None:
            out[f"{name}.conv.bias"] = cb
    return out
