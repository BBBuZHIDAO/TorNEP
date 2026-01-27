# ase_nep3_calculator.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import torch

from ase.calculators.calculator import Calculator, all_changes


@dataclass
class NEP3Meta:
    type_names: list[str]


class NEP3Calculator(Calculator):
    """
    ASE Calculator wrapper for your PyTorch NEP3 model loaded from GPUMD-format nep.txt.

    Implements:
      - energy (total potential energy, eV)
      - forces (eV/Å)

    Notes:
      - stress is NOT implemented (you can add later if needed).
      - PBC: if atoms.pbc has any True, we pass cell into the model; otherwise cell=None.
    """
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        nep_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        neighbor_backend: str = "ase",
        type_map: Optional[Dict[str, int]] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.nep_path = nep_path
        self.device = torch.device(device if (device != "auto") else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype
        self.neighbor_backend = neighbor_backend

        # ---- load model (expects your nep_io.py to provide load_gpumd_nep3_model) ----
        # Make sure nep_io.py is importable from your project.
        try:
            from io_utils.nep_io import load_gpumd_nep3_model  # repo-root execution
        except ImportError:
            from .io_utils.nep_io import load_gpumd_nep3_model  # package execution

        model, meta = load_gpumd_nep3_model(nep_path, device=self.device, neighbor_backend=neighbor_backend)
        model.to(self.device)
        model.eval()

        self.model = model

        # ---- type mapping: symbol -> type index ----
        # Prefer: user-provided type_map. Otherwise infer from nep.txt meta.
        if type_map is not None:
            self.type_map = dict(type_map)
            self.meta = NEP3Meta(type_names=list(type_map.keys()))
        else:
            type_names = getattr(meta, "type_names", None)
            if type_names is None:
                raise ValueError("Cannot infer type_names from NEP meta. Please pass type_map={'C':0,...}.")
            self.type_map = {sym: i for i, sym in enumerate(type_names)}
            self.meta = NEP3Meta(type_names=list(type_names))

    def _symbols_to_types(self, symbols: list[str]) -> torch.Tensor:
        try:
            t = [self.type_map[s] for s in symbols]
        except KeyError as e:
            raise KeyError(
                f"Element {e} not found in NEP type list. "
                f"Available types: {list(self.type_map.keys())}"
            ) from e
        return torch.tensor(t, dtype=torch.long, device=self.device)

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        # positions: [N,3], requires_grad for forces
        pos = torch.tensor(
            self.atoms.get_positions(),
            dtype=self.dtype,
            device=self.device,
            requires_grad=True,
        )

        # cell: pass only if periodic
        pbc = np.asarray(self.atoms.get_pbc(), dtype=bool)
        if np.any(pbc):
            cell = torch.tensor(self.atoms.get_cell().array, dtype=self.dtype, device=self.device)
        else:
            cell = None

        atype = self._symbols_to_types(self.atoms.get_chemical_symbols())

        # forward energy (scalar)
        E = self.model(pos, atype, cell)

        # forces: F = -dE/dR
        (dE_dR,) = torch.autograd.grad(E, pos, create_graph=False, retain_graph=False)
        F = -dE_dR

        # write ASE results (numpy on CPU)
        self.results["energy"] = float(E.detach().cpu().item())
        self.results["forces"] = F.detach().cpu().numpy().astype(np.float64)

