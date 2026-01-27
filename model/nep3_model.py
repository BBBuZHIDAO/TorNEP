"""
NEP3 model (PyTorch) aligned to GPUMD NEP3 descriptor definitions.

Key alignment points vs. your previous draft:
1) Bias handling ("方式一"): use bias-free Linear layers and subtract explicit trainable biases.
2) Neighbor vectors: ASE is used only to obtain (i, j, S) indices; rij vectors are built from torch positions
   so gradients (forces) are preserved. Torch MIC fallback is provided when ASE is unavailable.
3) Angular pipeline: uses accumulate_s + find_q from spherical_harmonics with torch tensors end-to-end
   (no geometry-dependent .item()).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional

# --- robust import for spherical_harmonics ---
try:
    # package usage: import model.nep3_model
    from .spherical_harmonics import (
        SphericalHarmonicsNEP,
        accumulate_s,
        find_q,
        NUM_OF_ABC,
    )
except Exception:
    try:
        # standalone: python model/nep3_model.py (cwd=model)
        from spherical_harmonics import SphericalHarmonicsNEP, accumulate_s, find_q, NUM_OF_ABC
    except Exception:
        # repo-root: python -c "import model.nep3_model"
        from model.spherical_harmonics import SphericalHarmonicsNEP, accumulate_s, find_q, NUM_OF_ABC

class ChebyshevBasis(nn.Module):
    """
    Chebyshev polynomial basis functions for NEP3 (GPUMD form).
    Used for both radial and angular basis functions.
    """

    def __init__(self, basis_size: int, cutoff: float):
        super().__init__()
        self.basis_size = int(basis_size)
        self.cutoff = float(cutoff)
        self.rcinv = 1.0 / float(cutoff)

    def cutoff_function(self, r: torch.Tensor) -> torch.Tensor:
        """
        Smooth cosine cutoff (GPUMD):
            f_c(r) = 0.5 * [cos(pi*r/rc) + 1] for r <= rc, else 0
        """
        x = r * self.rcinv
        return torch.where(
            r <= self.cutoff,
            0.5 * (torch.cos(torch.pi * x) + 1.0),
            torch.zeros_like(r),
        )

    def basis_functions(self, r: torch.Tensor) -> torch.Tensor:
        """
        GPUMD transformation:
            x  = 2 * (r/rc - 1)^2 - 1
            f0 = fc
            fn = (T_n(x) + 1) * 0.5 * fc   for n>=1
        Returns shape (..., basis_size+1).
        """
        fc = self.cutoff_function(r)
        r_scaled = r * self.rcinv
        x = 2.0 * (r_scaled - 1.0) ** 2 - 1.0

        half_fc = 0.5 * fc
        basis: List[torch.Tensor] = [fc]  # n=0

        if self.basis_size >= 1:
            basis.append((x + 1.0) * half_fc)  # n=1

        if self.basis_size >= 2:
            T_m_minus_2 = torch.ones_like(x)
            T_m_minus_1 = x
            for _m in range(2, self.basis_size + 1):
                T_m = 2.0 * x * T_m_minus_1 - T_m_minus_2
                basis.append((T_m + 1.0) * half_fc)
                T_m_minus_2 = T_m_minus_1
                T_m_minus_1 = T_m

        return torch.stack(basis, dim=-1)


def _mic_rij(rij: torch.Tensor, cell: torch.Tensor, inv_cell: torch.Tensor) -> torch.Tensor:
    """
    Minimum-image convention for a general 3x3 cell.
    rij: (..., 3)
    """
    frac = rij @ inv_cell.T
    frac = frac - torch.round(frac)
    return frac @ cell.T


class NEP3Model(nn.Module):
    """
    NEP3 (Neuroevolution Potential v3) implementation.

    Descriptors:
      - Radial (2-body): q_n = sum_j g_n(r_ij)
      - Angular (3-body / optional 4-/5-body invariants): via accumulate_s + find_q

    Neural network (GPUMD-like):
      y = W2 * tanh(W1 * x - b1) - b2
      (i.e., linear layers without bias and explicit trainable biases subtracted)
    """

    def __init__(
        self,
        n_max_radial: int = 4,
        n_max_angular: int = 4,
        basis_size_radial: int = 8,
        basis_size_angular: int = 8,
        l_max: int = 4,
        l_max_4body: int = 0,
        l_max_5body: int = 0,
        rc_radial: float = 8.0,
        rc_angular: float = 6.0,
        num_neurons: int = 100,
        num_types: int = 1,
        neighbor_backend: str = "ase",  # "ase" (recommended) or "torch"/"gpumd"
    ):
        super().__init__()

        self.n_max_radial = int(n_max_radial)
        self.n_max_angular = int(n_max_angular)
        self.basis_size_radial = int(basis_size_radial)
        self.basis_size_angular = int(basis_size_angular)
        self.l_max = int(l_max)
        self.l_max_4body = int(l_max_4body)
        self.l_max_5body = int(l_max_5body)
        self.rc_radial = float(rc_radial)
        self.rc_angular = float(rc_angular)
        self.num_types = int(num_types)
        self.num_neurons = int(num_neurons)

        # Basis generators
        self.radial_basis = ChebyshevBasis(self.basis_size_radial, self.rc_radial)
        self.angular_basis = ChebyshevBasis(self.basis_size_angular, self.rc_angular)

        # Spherical harmonics coefficient tables
        self.spherical_harmonics = SphericalHarmonicsNEP(self.l_max)

        # Trainable descriptor coefficients (GPUMD layout): [t_pair, n, k]
        self.c_radial = nn.Parameter(
            0.1 * torch.randn(self.num_types * self.num_types, self.n_max_radial + 1, self.basis_size_radial + 1)
        )
        self.c_angular_base = nn.Parameter(
            0.1 * torch.randn(self.num_types * self.num_types, self.n_max_angular + 1, self.basis_size_angular + 1)
        )

        # Descriptor dimensions
        self.dim_radial = self.n_max_radial + 1

        # num_L (GPUMD logic): 3-body channels = l_max; + optional 4b,5b
        self.num_L = self.l_max
        if self.l_max_4body == 2:
            self.num_L += 1
        if self.l_max_5body == 1:
            self.num_L += 1

        self.dim_angular = (self.n_max_angular + 1) * self.num_L
        self.descriptor_dim = self.dim_radial + self.dim_angular

        # --- Neural network (方式一：减去 b) ---
        self.lin1 = nn.Linear(self.descriptor_dim, self.num_neurons, bias=False)
        self.b1 = nn.Parameter(torch.zeros(self.num_neurons))
        self.lin2 = nn.Linear(self.num_neurons, 1, bias=False)
        self.b2 = nn.Parameter(torch.zeros(1))

        # Descriptor scaler (GPUMD q_scaler). Default=1 so it is a no-op unless loaded from nep.txt
        self.register_buffer("q_scaler", torch.ones(self.descriptor_dim))

        # Atomic reference energies (per type)
        self.atomic_energies = nn.Parameter(torch.zeros(self.num_types))

        self.neighbor_backend = neighbor_backend

    # --------------------------- neighbor building ---------------------------

    def build_neighbor_vectors_ase(
        self,
        positions: torch.Tensor,
        cell: Optional[torch.Tensor],
        cutoff: float,
        pbc: bool = True,
    ) -> Dict[int, List[Tuple[int, torch.Tensor]]]:
        """
        Use ASE to obtain (i, j, S) within cutoff, then build rij with torch positions:
            rij = r_j + S0*a + S1*b + S2*c - r_i
        This preserves gradients wrt positions.
        """
        n_atoms = int(positions.shape[0])
        neigh: Dict[int, List[Tuple[int, torch.Tensor]]] = {i: [] for i in range(n_atoms)}

        if cell is None:
            return self.build_neighbor_vectors_torch(positions, None, cutoff)

        try:
            from ase import Atoms
            from ase.neighborlist import neighbor_list
            import numpy as _np

            pos_np = positions.detach().cpu().numpy()
            cell_np = cell.detach().cpu().numpy()
            numbers = _np.ones(n_atoms, dtype=int)

            atoms = Atoms(numbers=numbers, positions=pos_np, cell=cell_np, pbc=pbc)
            i_idx, j_idx, S = neighbor_list("ijS", atoms, cutoff)

            # Build rij using torch tensors (no pos_np in geometry expression)
            for i, j, s in zip(i_idx, j_idx, S):
                i = int(i)
                j = int(j)
                if i == j:
                    continue
                shift = torch.as_tensor(s, device=positions.device, dtype=positions.dtype)
                delta = shift[0] * cell[0] + shift[1] * cell[1] + shift[2] * cell[2]
                rij = positions[j] + delta - positions[i]
                neigh[i].append((j, rij))

            return neigh
        except Exception:
            # Fallback to torch MIC neighbor build
            return self.build_neighbor_vectors_torch(positions, cell, cutoff)

    def build_neighbor_vectors_torch(
        self,
        positions: torch.Tensor,
        cell: Optional[torch.Tensor],
        cutoff: float,
    ) -> Dict[int, List[Tuple[int, torch.Tensor]]]:
        """
        Pure torch neighbor build (O(N^2)). For PBC, uses MIC with the given cell.
        """
        n_atoms = int(positions.shape[0])
        neigh: Dict[int, List[Tuple[int, torch.Tensor]]] = {i: [] for i in range(n_atoms)}

        inv_cell = None
        if cell is not None:
            inv_cell = torch.inverse(cell)

        for i in range(n_atoms):
            for j in range(n_atoms):
                if i == j:
                    continue
                rij = positions[j] - positions[i]
                if cell is not None and inv_cell is not None:
                    rij = _mic_rij(rij, cell, inv_cell)
                dist = torch.linalg.norm(rij)
                if dist.item() < cutoff and dist.item() > 1e-12:
                    neigh[i].append((j, rij))
        return neigh

    def _neighbors(self, positions: torch.Tensor, cell: Optional[torch.Tensor], cutoff: float) -> Dict[int, List[Tuple[int, torch.Tensor]]]:
        if self.neighbor_backend.lower() in ("gpumd", "torch"):
            return self.build_neighbor_vectors_torch(positions, cell, cutoff)
        return self.build_neighbor_vectors_ase(positions, cell, cutoff)

    # --------------------------- descriptors ---------------------------

    def compute_radial_descriptor(
        self,
        positions: torch.Tensor,
        atom_types: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Radial descriptor q_radial[i, n] = sum_j g_n(r_ij).
        """
        n_atoms = int(positions.shape[0])
        dtype = positions.dtype
        device = positions.device

        q_radial = torch.zeros((n_atoms, self.n_max_radial + 1), device=device, dtype=dtype)

        neighbors_map = self._neighbors(positions, cell, self.rc_radial)

        for i in range(n_atoms):
            t_i = int(atom_types[i].item())
            for j, rij in neighbors_map.get(i, []):
                dist = torch.linalg.norm(rij)
                if dist.item() >= self.rc_radial:
                    continue

                t_j = int(atom_types[j].item())
                t_pair = t_i * self.num_types + t_j

                basis = self.radial_basis.basis_functions(dist)  # (K,)
                # gn(n) = sum_k fn_k * c[t_pair, n, k]
                # vectorize over k
                for n in range(self.n_max_radial + 1):
                    gn = torch.sum(basis * self.c_radial[t_pair, n, :].to(dtype=dtype))
                    q_radial[i, n] = q_radial[i, n] + gn

        return q_radial

    def compute_angular_descriptor(
        self,
        positions: torch.Tensor,
        atom_types: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Angular descriptor via GPUMD pipeline:
          for each i, for each n:
            s[:] = 0
            for each neighbor j:
              gn12 = sum_k fn_k(r_ij) * c[t_pair, n, k]
              accumulate_s(l_max, r_ij, gn12, s)
            find_q(l_max, num_L, n_max+1, n, s, q_i, c3b,c4b,c5b)
        """
        n_atoms = int(positions.shape[0])
        dtype = positions.dtype
        device = positions.device

        n_max_angular_plus_1 = self.n_max_angular + 1
        q_angular = torch.zeros((n_atoms, n_max_angular_plus_1 * self.num_L), device=device, dtype=dtype)

        neighbors_map = self._neighbors(positions, cell, self.rc_angular)

        for i in range(n_atoms):
            t_i = int(atom_types[i].item())
            neighbors_i = neighbors_map.get(i, [])
            if len(neighbors_i) == 0:
                continue

            for n in range(n_max_angular_plus_1):
                s = torch.zeros((NUM_OF_ABC,), device=device, dtype=dtype)

                for j, rij in neighbors_i:
                    dist_j = torch.linalg.norm(rij)
                    if dist_j.item() >= self.rc_angular:
                        continue

                    t_j = int(atom_types[j].item())
                    t_pair = t_i * self.num_types + t_j

                    basis_j = self.angular_basis.basis_functions(dist_j)  # (K,)
                    gn12 = torch.sum(basis_j * self.c_angular_base[t_pair, n, :].to(dtype=dtype))

                    accumulate_s(self.l_max, dist_j, rij[0], rij[1], rij[2], gn12, s)

                find_q(
                    self.l_max,
                    self.num_L,
                    n_max_angular_plus_1,
                    n,
                    s,
                    q_angular[i],
                    self.spherical_harmonics.c3b.to(dtype=dtype),
                    self.spherical_harmonics.c4b.to(dtype=dtype),
                    self.spherical_harmonics.c5b.to(dtype=dtype),
                )

        return q_angular

    def compute_descriptors(
        self,
        positions: torch.Tensor,
        atom_types: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_radial = self.compute_radial_descriptor(positions, atom_types, cell)
        q_angular = self.compute_angular_descriptor(positions, atom_types, cell)
        descriptors = torch.cat([q_radial, q_angular], dim=1)
        return descriptors, q_radial, q_angular

    # --------------------------- forward ---------------------------

    def forward(
        self,
        positions: torch.Tensor,
        atom_types: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Total energy = sum_i [ NN(descriptor_i) + E0(type_i) ].
        """
        descriptors, _, _ = self.compute_descriptors(positions, atom_types, cell)

        # Apply GPUMD q_scaler (element-wise) before ANN
        if getattr(self, "q_scaler", None) is not None:
            descriptors = descriptors * self.q_scaler

        # 方式一：显式减去 b
        h = torch.tanh(self.lin1(descriptors) - self.b1)
        e_i = (self.lin2(h) - self.b2).squeeze(-1)

        e_i = e_i + self.atomic_energies[atom_types]
        return torch.sum(e_i)
