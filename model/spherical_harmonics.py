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
    """Convert x to a 0-dim tensor on ref's device/dtype."""
    if isinstance(x, torch.Tensor):
        return x.to(device=ref.device, dtype=ref.dtype)
    return torch.tensor(x, device=ref.device, dtype=ref.dtype)


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

    Parameters
    ----------
    L : int
        Angular momentum.
    x12, y12, z12 : TensorLike
        Normalized direction components.
    fn : TensorLike
        Radial prefactor (e.g., g_n(r_ij)).
    s : torch.Tensor
        Output array (length NUM_OF_ABC), updated in place.
    """
    ref = s
    x12_t = _to_scalar_tensor(x12, ref)
    y12_t = _to_scalar_tensor(y12, ref)
    z12_t = _to_scalar_tensor(z12, ref)
    fn_t = _to_scalar_tensor(fn, ref)

    s_index = L * L - 1

    # z^n
    z_pow = [torch.ones((), device=ref.device, dtype=ref.dtype)]
    for _ in range(1, L + 1):
        z_pow.append(z12_t * z_pow[-1])

    real_part = x12_t
    imag_part = y12_t

    Z_COEFF = Z_COEFFICIENTS[L].to(device=ref.device, dtype=ref.dtype)

    for n1 in range(L + 1):
        n2_start = 0 if (L + n1) % 2 == 0 else 1
        z_factor = torch.zeros((), device=ref.device, dtype=ref.dtype)
        for n2 in range(n2_start, L - n1 + 1, 2):
            z_factor = z_factor + Z_COEFF[n1, n2] * z_pow[n2]
        z_factor = z_factor * fn_t

        if n1 == 0:
            s[s_index] = s[s_index] + z_factor
            s_index += 1
        else:
            s[s_index] = s[s_index] + z_factor * real_part
            s_index += 1
            s[s_index] = s[s_index] + z_factor * imag_part
            s_index += 1
            real_part, imag_part = complex_product(x12_t, y12_t, real_part, imag_part, ref)


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
    """
    start_index = L * L - 1
    num_terms = 2 * L + 1

    q = torch.zeros((), device=s.device, dtype=s.dtype)
    for k in range(1, num_terms):
        q = q + c3b[start_index + k].to(dtype=s.dtype, device=s.device) * s[start_index + k] * s[start_index + k]
    q = q * 2.0
    q = q + c3b[start_index].to(dtype=s.dtype, device=s.device) * s[start_index] * s[start_index]
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
