"""
Spherical harmonics utilities for NEP3 angular descriptors.

This module is written to mirror GPUMD NEP3's workflow:
  1) accumulate_s : accumulate real spherical harmonics components into s (NUM_OF_ABC=80)
  2) find_q       : convert s into invariant angular descriptors q via pre-tabulated coefficients

Implementation notes (PyTorch):
- All math is done in torch Tensors (no .item() on geometry-dependent values),
  so gradients can flow from q back to atomic positions.
- The control-flow loops match GPUMD's integer loops (L, n1, n2, ...).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple, Union


# Constants from GPUMD nep_utilities.cuh
NUM_OF_ABC = 80  # 3 + 5 + 7 + 9 + 11 + 13 + 15 + 17 for L_max = 8

# C3B coefficients for find_q (GPUMD constants)
C3B = torch.tensor([
    0.238732414637843, 0.119366207318922, 0.119366207318922, 0.099471839432435,
    0.596831036594608, 0.596831036594608, 0.149207759148652, 0.149207759148652,
    0.139260575205408, 0.104445431404056, 0.104445431404056, 1.044454314040563,
    1.044454314040563, 0.174075719006761, 0.174075719006761, 0.011190581936149,
    0.223811638722978, 0.223811638722978, 0.111905819361489, 0.111905819361489,
    1.566681471060845, 1.566681471060845, 0.195835183882606, 0.195835183882606,
    0.013677377921960, 0.102580334414698, 0.102580334414698, 2.872249363611549,
    2.872249363611549, 0.119677056817148, 0.119677056817148, 2.154187022708661,
    2.154187022708661, 0.215418702270866, 0.215418702270866, 0.004041043476943,
    0.169723826031592, 0.169723826031592, 0.106077391269745, 0.106077391269745,
    0.424309565078979, 0.424309565078979, 0.127292869523694, 0.127292869523694,
    2.800443129521260, 2.800443129521260, 0.233370260793438, 0.233370260793438,
    0.004662742473395, 0.004079899664221, 0.004079899664221, 0.024479397985326,
    0.024479397985326, 0.012239698992663, 0.012239698992663, 0.538546755677165,
    0.538546755677165, 0.134636688919291, 0.134636688919291, 3.500553911901575,
    3.500553911901575, 0.250039565135827, 0.250039565135827, 0.000082569397966,
    0.005944996653579, 0.005944996653579, 0.104037441437634, 0.104037441437634,
    0.762941237209318, 0.762941237209318, 0.114441185581398, 0.114441185581398,
    5.950941650232678, 5.950941650232678, 0.141689086910302, 0.141689086910302,
    4.250672607309055, 4.250672607309055, 0.265667037956816, 0.265667037956816
], dtype=torch.float32)

# C4B and C5B for 4-body and 5-body terms (GPUMD constants)
C4B = torch.tensor([
    -0.007499480826664,
    -0.134990654879954,
    0.067495327439977,
    0.404971964639861,
    -0.809943929279723
], dtype=torch.float32)

C5B = torch.tensor([
    0.026596810706114,
    0.053193621412227,
    0.026596810706114
], dtype=torch.float32)

# Z_COEFFICIENT arrays for spherical harmonics (GPUMD constants)
Z_COEFFICIENTS = [
    None,  # L=0 not used
    torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32),  # L=1
    torch.tensor([[-1.0, 0.0, 3.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32),  # L=2
    torch.tensor([
        [0.0, -3.0, 0.0, 5.0],
        [-1.0, 0.0, 5.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0]
    ], dtype=torch.float32),  # L=3
    torch.tensor([
        [3.0, 0.0, -30.0, 0.0, 35.0],
        [0.0, -3.0, 0.0, 7.0, 0.0],
        [-1.0, 0.0, 7.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0]
    ], dtype=torch.float32),  # L=4
    torch.tensor([
        [0.0, 15.0, 0.0, -70.0, 0.0, 63.0],
        [1.0, 0.0, -14.0, 0.0, 21.0, 0.0],
        [0.0, -1.0, 0.0, 3.0, 0.0, 0.0],
        [-1.0, 0.0, 9.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ], dtype=torch.float32),  # L=5
    torch.tensor([
        [-5.0, 0.0, 105.0, 0.0, -315.0, 0.0, 231.0],
        [0.0, 5.0, 0.0, -30.0, 0.0, 33.0, 0.0],
        [1.0, 0.0, -18.0, 0.0, 33.0, 0.0, 0.0],
        [0.0, -3.0, 0.0, 11.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 11.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ], dtype=torch.float32),  # L=6
    torch.tensor([
        [0.0, -35.0, 0.0, 315.0, 0.0, -693.0, 0.0, 429.0],
        [-5.0, 0.0, 135.0, 0.0, -495.0, 0.0, 429.0, 0.0],
        [0.0, 15.0, 0.0, -110.0, 0.0, 143.0, 0.0, 0.0],
        [3.0, 0.0, -66.0, 0.0, 143.0, 0.0, 0.0, 0.0],
        [0.0, -3.0, 0.0, 13.0, 0.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 13.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ], dtype=torch.float32),  # L=7
    torch.tensor([
        [35.0, 0.0, -1260.0, 0.0, 6930.0, 0.0, -12012.0, 0.0, 6435.0],
        [0.0, -35.0, 0.0, 385.0, 0.0, -1001.0, 0.0, 715.0, 0.0],
        [-1.0, 0.0, 33.0, 0.0, -143.0, 0.0, 143.0, 0.0, 0.0],
        [0.0, 3.0, 0.0, -26.0, 0.0, 39.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, -26.0, 0.0, 65.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ], dtype=torch.float32),  # L=8
]


TensorLike = Union[torch.Tensor, float, int]


def _to_scalar_tensor(x: TensorLike, ref: torch.Tensor) -> torch.Tensor:
    """Convert ``x`` to a 0-dim tensor on ``ref``'s device/dtype (cheap fast-path)."""
    if isinstance(x, torch.Tensor):
        # Avoid creating new tensors when already aligned.
        if x.device == ref.device and x.dtype == ref.dtype and x.ndim == 0:
            return x
        return x.to(device=ref.device, dtype=ref.dtype)
    return torch.as_tensor(x, device=ref.device, dtype=ref.dtype)


# ---- small caches to avoid repeated .to() on constant tables ----
_Z_CACHE: dict[tuple[int, str, str], torch.Tensor] = {}
_MASK_CACHE: dict[tuple[int, str, str], torch.Tensor] = {}

# Precompute parity/limit masks on CPU (float32), then move/cast once per (device, dtype).
_Z_MASKS_CPU: list[torch.Tensor | None] = [None]
for _L in range(1, 9):
    _n1 = torch.arange(0, _L + 1, dtype=torch.int64).view(-1, 1)
    _n2 = torch.arange(0, _L + 1, dtype=torch.int64).view(1, -1)
    # allowed n2: n2 <= L-n1 and parity(n2) == parity(L+n1)
    _mask = (_n2 <= (_L - _n1)) & ((_n2 & 1) == ((_L + _n1) & 1))
    _Z_MASKS_CPU.append(_mask.to(torch.float32))

def _get_Z_coeff(L: int, ref: torch.Tensor) -> torch.Tensor:
    key = (int(L), str(ref.device), str(ref.dtype))
    z = _Z_CACHE.get(key)
    if z is None:
        z = Z_COEFFICIENTS[int(L)].to(device=ref.device, dtype=ref.dtype)
        _Z_CACHE[key] = z
    return z

def _get_Z_mask(L: int, ref: torch.Tensor) -> torch.Tensor:
    key = (int(L), str(ref.device), str(ref.dtype))
    m = _MASK_CACHE.get(key)
    if m is None:
        m = _Z_MASKS_CPU[int(L)].to(device=ref.device, dtype=ref.dtype)
        _MASK_CACHE[key] = m
    return m


def complex_product(a: TensorLike, b: TensorLike, real_part: torch.Tensor, imag_part: torch.Tensor, ref: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Complex multiplication: (real + i*imag) * (a + i*b) with torch tensors."""
    a_t = _to_scalar_tensor(a, ref)
    b_t = _to_scalar_tensor(b, ref)
    new_real = a_t * real_part - b_t * imag_part
    new_imag = a_t * imag_part + b_t * real_part
    return new_real, new_imag


def accumulate_s_one(L: int, x12: TensorLike, y12: TensorLike, z12: TensorLike, fn: TensorLike, s: torch.Tensor) -> None:
    """
    Accumulate spherical harmonics for angular momentum L (GPUMD accumulate_s_one).

    This implementation is vectorized over the internal (n1,n2) summations to reduce
    Python-level overhead and repeated tensor conversions.
    """
    ref = s
    x12_t = _to_scalar_tensor(x12, ref)
    y12_t = _to_scalar_tensor(y12, ref)
    z12_t = _to_scalar_tensor(z12, ref)
    fn_t = _to_scalar_tensor(fn, ref)

    s_index = L * L - 1

    # z^n, n=0..L
    n = torch.arange(0, L + 1, device=ref.device, dtype=ref.dtype)
    z_pow = z12_t ** n  # (L+1,)

    # z_factor[n1] = fn * sum_{n2} Z[n1,n2] * z^n2, with GPUMD parity/limit rule
    Z = _get_Z_coeff(L, ref)          # (L+1, L+1)
    M = _get_Z_mask(L, ref)           # (L+1, L+1)
    z_factor = (Z * M) @ z_pow        # (L+1,)
    z_factor = z_factor * fn_t        # (L+1,)

    # (x + i y)^m, m=1..L (use complex autograd, then take real/imag)
    if L >= 1:
        c = torch.complex(x12_t, y12_t)
        k = torch.arange(1, L + 1, device=ref.device, dtype=torch.int64)
        c_pow = c ** k  # (L,)
        real_part = c_pow.real.to(dtype=ref.dtype)
        imag_part = c_pow.imag.to(dtype=ref.dtype)

    # fill (2L+1) components into s: [m=0, m=1(cos), m=1(sin), ...]
    vals = torch.empty((2 * L + 1,), device=ref.device, dtype=ref.dtype)
    vals[0] = z_factor[0]
    if L >= 1:
        vals[1::2] = z_factor[1:] * real_part
        vals[2::2] = z_factor[1:] * imag_part

    s[s_index : s_index + 2 * L + 1] = s[s_index : s_index + 2 * L + 1] + vals


def accumulate_s(L_max: int, d12: TensorLike, x12: TensorLike, y12: TensorLike, z12: TensorLike, fn: TensorLike, s: torch.Tensor) -> None:
    """
    Accumulate spherical harmonics for all L = 1..L_max (GPUMD accumulate_s).
    """
    ref = s
    d12_t = _to_scalar_tensor(d12, ref)
    x12_t = _to_scalar_tensor(x12, ref)
    y12_t = _to_scalar_tensor(y12, ref)
    z12_t = _to_scalar_tensor(z12, ref)
    fn_t = _to_scalar_tensor(fn, ref)

    d12inv = 1.0 / d12_t
    x12_n = x12_t * d12inv
    y12_n = y12_t * d12inv
    z12_n = z12_t * d12inv

    for L in range(1, L_max + 1):
        accumulate_s_one(L, x12_n, y12_n, z12_n, fn_t, s)


def find_q_one(L: int, s: torch.Tensor, c3b: torch.Tensor) -> torch.Tensor:
    """
    Compute invariant q for a given L (GPUMD find_q_one).

    Vectorized to avoid Python loops and repeated dtype/device casts.
    """
    start_index = L * L - 1
    num_terms = 2 * L + 1

    # Ensure coefficient table is on the same device/dtype (cheap if already aligned)
    c = c3b.to(device=s.device, dtype=s.dtype)

    ss = s[start_index : start_index + num_terms]
    cc = c[start_index : start_index + num_terms]

    # q = c0*s0^2 + 2*sum_{k=1}^{2L} ck*sk^2
    q = cc[0] * ss[0] * ss[0]
    if num_terms > 1:
        q = q + 2.0 * torch.sum(cc[1:] * (ss[1:] * ss[1:]))
    return q


def find_q(
    L_max: int,
    num_L: int,
    n_max_angular_plus_1: int,
    n: int,
    s: torch.Tensor,
    q: torch.Tensor,
    c3b: torch.Tensor,
    c4b: torch.Tensor,
    c5b: torch.Tensor
) -> None:
    """
    Convert accumulated s into angular descriptors q (GPUMD find_q).
    """
    # 3-body channels, L=1..L_max (capped to 8 in GPUMD tables)
    for L in range(1, min(L_max + 1, 9)):
        q_val = find_q_one(L, s, c3b)
        q[(L - 1) * n_max_angular_plus_1 + n] = q_val

    # 4-body term (enabled when l_max_4body==2 in GPUMD; reflected by num_L)
    if num_L >= L_max + 1:
        q_4b = (
            c4b[0] * s[3] * s[3] * s[3] +
            c4b[1] * s[3] * (s[4] * s[4] + s[5] * s[5]) +
            c4b[2] * s[3] * (s[6] * s[6] + s[7] * s[7]) +
            c4b[3] * s[6] * (s[5] * s[5] - s[4] * s[4]) +
            c4b[4] * s[4] * s[5] * s[7]
        )
        q[L_max * n_max_angular_plus_1 + n] = q_4b

    # 5-body term (enabled when l_max_5body==1 in GPUMD; reflected by num_L)
    if num_L >= L_max + 2:
        s0_sq = s[0] * s[0]
        s1_sq_plus_s2_sq = s[1] * s[1] + s[2] * s[2]
        q_5b = (
            c5b[0] * s0_sq * s0_sq +
            c5b[1] * s0_sq * s1_sq_plus_s2_sq +
            c5b[2] * s1_sq_plus_s2_sq * s1_sq_plus_s2_sq
        )
        q[(L_max + 1) * n_max_angular_plus_1 + n] = q_5b


def accumulate_s_edges_batched(
    L_max: int,
    rij: torch.Tensor,
    fn: torch.Tensor,
    edge_i: torch.Tensor,
    n_atoms: int,
) -> torch.Tensor:
    """
    Batched accumulate_s over an *edge list*.

    Parameters
    ----------
    L_max : int
        Maximum angular momentum (same as in GPUMD; typical <= 8).
    rij : torch.Tensor
        Edge vectors in Cartesian coordinates, shape (E, 3).
        Must be built from torch positions to preserve gradients w.r.t. positions.
    fn : torch.Tensor
        Radial prefactors per edge and per angular n-channel, shape (E, n_max_angular_plus_1).
        This is typically gn12 for each edge (i,j) and channel n.
    edge_i : torch.Tensor
        Center indices for each edge, shape (E,) long. Contributions are accumulated to these atoms.
    n_atoms : int
        Number of atoms N.

    Returns
    -------
    s : torch.Tensor
        Accumulated real spherical-harmonics components, shape (N, n_max_angular_plus_1, NUM_OF_ABC).
        Layout of the last dimension follows the GPUMD packing (start=L^2-1, length=2L+1).
    """
    if rij.numel() == 0:
        return torch.zeros((n_atoms, int(fn.shape[1]), NUM_OF_ABC), device=rij.device, dtype=rij.dtype)

    ref = rij
    E = int(rij.shape[0])
    n_chan = int(fn.shape[1])

    # distances and normalized directions
    d = torch.linalg.norm(rij, dim=1)  # (E,)
    d_inv = 1.0 / d
    x = rij[:, 0] * d_inv
    y = rij[:, 1] * d_inv
    z = rij[:, 2] * d_inv

    s_out = torch.zeros((n_atoms, n_chan, NUM_OF_ABC), device=ref.device, dtype=ref.dtype)

    # Loop over L only (small); vectorize over edges and n-channels.
    for L in range(1, L_max + 1):
        start = L * L - 1
        width = 2 * L + 1

        # z^n, n=0..L, for all edges
        n = torch.arange(0, L + 1, device=ref.device, dtype=ref.dtype)  # (L+1,)
        z_pow = z[:, None] ** n[None, :]  # (E, L+1)

        Z = _get_Z_coeff(L, ref)  # (L+1, L+1)
        M = _get_Z_mask(L, ref)   # (L+1, L+1)
        ZM = (Z * M)              # (L+1, L+1)

        # z_factor_base[e, n1] = sum_{n2} ZM[n1,n2] * z_pow[e,n2]
        z_factor_base = z_pow @ ZM.T  # (E, L+1)

        # (x + i y)^m, m=1..L
        if L >= 1:
            c = torch.complex(x, y)  # (E,)
            k = torch.arange(1, L + 1, device=ref.device, dtype=torch.int64)  # (L,)
            c_pow = c[:, None] ** k[None, :]  # (E, L)
            real_part = c_pow.real.to(dtype=ref.dtype)
            imag_part = c_pow.imag.to(dtype=ref.dtype)

        # base_block is the (2L+1) components for fn=1.0
        base_block = torch.empty((E, width), device=ref.device, dtype=ref.dtype)
        base_block[:, 0] = z_factor_base[:, 0]
        if L >= 1:
            base_block[:, 1::2] = z_factor_base[:, 1:] * real_part
            base_block[:, 2::2] = z_factor_base[:, 1:] * imag_part

        # Multiply by fn for all channels and scatter-add to centers
        vals = fn[:, :, None] * base_block[:, None, :]  # (E, n_chan, width)
        vals2d = vals.reshape(E, n_chan * width)

        tmp = torch.zeros((n_atoms, n_chan * width), device=ref.device, dtype=ref.dtype)
        tmp.index_add_(0, edge_i, vals2d)

        s_out[:, :, start : start + width] += tmp.reshape(n_atoms, n_chan, width)

    return s_out


def find_q_batched(
    L_max: int,
    num_L: int,
    s: torch.Tensor,
    c3b: torch.Tensor,
    c4b: torch.Tensor,
    c5b: torch.Tensor,
) -> torch.Tensor:
    """
    Batched GPUMD find_q.

    Parameters
    ----------
    s : torch.Tensor
        Accumulated s, shape (N, n_chan, NUM_OF_ABC).

    Returns
    -------
    q : torch.Tensor
        Angular invariants, shape (N, num_L, n_chan).
        Flattening with q.reshape(N, -1) matches GPUMD indexing:
            q[(L-1)*n_chan + n] for 3-body channels.
    """
    N = int(s.shape[0])
    n_chan = int(s.shape[1])
    device = s.device
    dtype = s.dtype

    q = torch.zeros((N, num_L, n_chan), device=device, dtype=dtype)

    c3 = c3b.to(device=device, dtype=dtype)

    # 3-body: L=1..min(L_max, 8)
    for L in range(1, min(L_max + 1, 9)):
        start = L * L - 1
        width = 2 * L + 1

        ss = s[:, :, start : start + width]  # (N, n_chan, width)
        w = c3[start : start + width]        # (width,)
        if width > 1:
            w = w.clone()
            w[1:] = w[1:] * 2.0

        q[:, L - 1, :] = torch.sum((ss * ss) * w.view(1, 1, -1), dim=-1)

    # 4-body term (when num_L >= L_max+1)
    if num_L >= L_max + 1:
        c4 = c4b.to(device=device, dtype=dtype)
        s3 = s[:, :, 3]
        s4 = s[:, :, 4]
        s5 = s[:, :, 5]
        s6 = s[:, :, 6]
        s7 = s[:, :, 7]
        q4 = (
            c4[0] * s3 * s3 * s3 +
            c4[1] * s3 * (s4 * s4 + s5 * s5) +
            c4[2] * s3 * (s6 * s6 + s7 * s7) +
            c4[3] * s6 * (s5 * s5 - s4 * s4) +
            c4[4] * s4 * s5 * s7
        )
        q[:, L_max, :] = q4

    # 5-body term (when num_L >= L_max+2)
    if num_L >= L_max + 2:
        c5 = c5b.to(device=device, dtype=dtype)
        s0 = s[:, :, 0]
        s1 = s[:, :, 1]
        s2 = s[:, :, 2]
        s0_sq = s0 * s0
        s1_sq_plus_s2_sq = s1 * s1 + s2 * s2
        q5 = (
            c5[0] * s0_sq * s0_sq +
            c5[1] * s0_sq * s1_sq_plus_s2_sq +
            c5[2] * s1_sq_plus_s2_sq * s1_sq_plus_s2_sq
        )
        q[:, L_max + 1, :] = q5

    return q

class SphericalHarmonicsNEP(nn.Module):
    """
    Small wrapper storing GPUMD coefficient tables as buffers.
    """
    def __init__(self, l_max: int):
        super().__init__()
        self.l_max = int(l_max)
        self.register_buffer('c3b', C3B)
        self.register_buffer('c4b', C4B)
        self.register_buffer('c5b', C5B)

    def to(self, *args, **kwargs):
        # Ensure buffers move correctly (default nn.Module.to already does, but keep explicit)
        return super().to(*args, **kwargs)