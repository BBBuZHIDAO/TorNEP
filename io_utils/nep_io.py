"""
Utility functions for NEP3 I/O.

This module supports:
1) Official GPUMD `nep.txt` parsing/writing (NEP3 potential, i.e., model_type=0, version=3).
2) Loading GPUMD NEP3 parameters into the PyTorch `NEP3Model` implementation used in this repo.
3) Saving a PyTorch `NEP3Model` back to a GPUMD-compatible `nep.txt`.

Notes on GPUMD alignment:
- GPUMD NEP3 uses a single shared ANN for all atom types, and type dependence enters via pair-dependent
  descriptor coefficients (c_radial and c_angular).
- GPUMD uses the form `tanh(Wx - b)` and `(w·h - b)`; in the PyTorch model this is implemented as
  bias-free Linear layers and explicit trainable biases subtracted in forward.
- GPUMD applies an element-wise `q_scaler` to the descriptor vector before the ANN. The PyTorch model
  should expose a buffer `q_scaler` (default ones). If absent, this module will attach it, but you must
  ensure your forward applies it.

If you only need interoperability with GPUMD, prefer:
- `load_gpumd_nep3_model(...)`
- `save_gpumd_nep3_model(...)`
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import torch


# ----------------------------- model import ---------------------------------

def _import_nep3_model():
    """
    Robustly import NEP3Model.

    Supported layouts:
      1) As an installed package: <rootpkg>.model.nep3_model
      2) Repo-root on PYTHONPATH: model.nep3_model
      3) Load by file path: ../model/nep3_model.py relative to this file
    """
    import importlib
    import importlib.util as _ilu
    import sys as _sys
    from pathlib import Path

    # 1) If inside a package (e.g., tornep.io_utils), try <rootpkg>.model.nep3_model
    if __package__ and "." in __package__:
        root_pkg = __package__.split(".", 1)[0]
        for modname in (
            f"{root_pkg}.model.nep3_model",
            f"{root_pkg}.model.nep3_model_fixed_v4",
            f"{root_pkg}.model.nep3_model_fixed_v3",
            f"{root_pkg}.model.nep3_model_fixed_v2",
            f"{root_pkg}.model.nep3_model_fixed",
        ):
            try:
                return importlib.import_module(modname).NEP3Model  # type: ignore[attr-defined]
            except Exception:
                pass

    # 2) If running from repo root (root on sys.path), try model.nep3_model
    for modname in (
        "model.nep3_model",
        "model.nep3_model_fixed_v4",
        "model.nep3_model_fixed_v3",
        "model.nep3_model_fixed_v2",
        "model.nep3_model_fixed",
    ):
        try:
            return importlib.import_module(modname).NEP3Model  # type: ignore[attr-defined]
        except Exception:
            pass

    # 3) Load by file path: <repo_root>/model/nep3_model*.py
    here = Path(__file__).resolve()
    repo_root = here.parent.parent
    model_dir = repo_root / "model"

    for fname in (
        "nep3_model.py",
        "nep3_model_fixed_v4.py",
        "nep3_model_fixed_v3.py",
        "nep3_model_fixed_v2.py",
        "nep3_model_fixed.py",
    ):
        fpath = model_dir / fname
        if fpath.exists():
            mod_name = f"_nep3_dynamic_{fpath.stem}"
            spec = _ilu.spec_from_file_location(mod_name, fpath)
            if spec and spec.loader:
                mod = _ilu.module_from_spec(spec)
                _sys.modules[mod_name] = mod
                spec.loader.exec_module(mod)  # type: ignore[misc]
                return getattr(mod, "NEP3Model")

    raise ImportError(
        "Cannot import NEP3Model. Checked package import, model.nep3_model, and ../model/*.py"
    )


NEP3Model = _import_nep3_model()



# ------------------------------ GPUMD format --------------------------------

@dataclass
class GPUMDNEP:
    """Container for the official GPUMD nep.txt layout."""
    version: int
    model_type: int  # 0: potential, 1: dipole, 2: polarizability, 3: temperature
    num_types: int
    type_names: List[str]

    rc_radial: float
    rc_angular: float
    max_nn_radial: int
    max_nn_angular: int
    typewise_cutoff_factors: Optional[Tuple[float, float, float]] = None

    n_max_radial: int = 0
    n_max_angular: int = 0
    basis_size_radial: int = 0
    basis_size_angular: int = 0
    l_max_3body: int = 0
    l_max_4body: int = 0
    l_max_5body: int = 0
    num_neurons: int = 0

    zbl_enabled: bool = False
    zbl_flexible: bool = False
    zbl_rc_inner: float = 0.0
    zbl_rc_outer: float = 0.0

    parameters: Optional[torch.Tensor] = None   # shape [num_para]
    q_scaler: Optional[torch.Tensor] = None     # shape [dim]
    zbl_parameters: Optional[torch.Tensor] = None


_MODEL_TAGS: Dict[str, Tuple[int, int, bool]] = {
    "nep3": (3, 0, False),
    "nep3_zbl": (3, 0, True),
    "nep4": (4, 0, False),
    "nep4_zbl": (4, 0, True),
    "nep5": (5, 0, False),
    "nep5_zbl": (5, 0, True),

    "nep3_temperature": (3, 3, False),
    "nep3_zbl_temperature": (3, 3, True),
    "nep4_temperature": (4, 3, False),
    "nep4_zbl_temperature": (4, 3, True),

    "nep3_dipole": (3, 1, False),
    "nep4_dipole": (4, 1, False),

    "nep3_polarizability": (3, 2, False),
    "nep4_polarizability": (4, 2, False),
}


def _parse_model_tag(tag: str) -> Tuple[int, int, bool]:
    if tag not in _MODEL_TAGS:
        raise ValueError(f"Unsupported NEP model tag: {tag}")
    return _MODEL_TAGS[tag]


def _model_tag_from_meta(nep: GPUMDNEP) -> str:
    if nep.model_type == 1:
        return f"nep{nep.version}_dipole"
    if nep.model_type == 2:
        return f"nep{nep.version}_polarizability"
    if nep.model_type == 3:
        return f"nep{nep.version}{'_zbl' if nep.zbl_enabled else ''}_temperature"
    return f"nep{nep.version}{'_zbl' if nep.zbl_enabled else ''}"


def _compute_counts(nep: GPUMDNEP) -> Dict[str, int]:
    # number of L-channels (GPUMD logic)
    num_L = nep.l_max_3body
    if nep.l_max_4body == 2:
        num_L += 1
    if nep.l_max_5body == 1:
        num_L += 1

    dim = (nep.n_max_radial + 1) + (nep.n_max_angular + 1) * num_L
    if nep.model_type == 3:  # temperature adds an extra scalar input
        dim += 1

    if nep.version == 3:
        num_para_ann = (dim + 2) * nep.num_neurons + 1
    elif nep.version == 4:
        num_para_ann = (dim + 2) * nep.num_neurons * nep.num_types + 1
    else:
        num_para_ann = ((dim + 2) * nep.num_neurons + 1) * nep.num_types + 1

    if nep.model_type == 2:  # polarizability is doubled
        num_para_ann *= 2

    num_types_sq = nep.num_types * nep.num_types
    num_para_descriptor = num_types_sq * (
        (nep.n_max_radial + 1) * (nep.basis_size_radial + 1)
        + (nep.n_max_angular + 1) * (nep.basis_size_angular + 1)
    )

    num_para = num_para_ann + num_para_descriptor
    num_type_zbl = (nep.num_types * (nep.num_types + 1)) // 2

    return {
        "dim": dim,
        "num_L": num_L,
        "num_para_ann": num_para_ann,
        "num_para_descriptor": num_para_descriptor,
        "num_para": num_para,
        "num_type_zbl": num_type_zbl,
    }


def read_gpumd_nep_txt(path: str | Path, device: Optional[torch.device] = None) -> GPUMDNEP:
    """Parse an official GPUMD nep.txt file into a structured container."""
    lines = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError("nep.txt is empty")

    idx = 0
    header_tokens = lines[idx].split()
    idx += 1

    version, model_type, zbl_enabled = _parse_model_tag(header_tokens[0])
    num_types = int(header_tokens[1])
    type_names = header_tokens[2:]
    if len(type_names) != num_types:
        raise ValueError("The first line must list all element symbols.")

    zbl_flexible = False
    zbl_rc_inner = 0.0
    zbl_rc_outer = 0.0
    if zbl_enabled:
        tokens = lines[idx].split()
        idx += 1
        if tokens[0] != "zbl" or len(tokens) != 3:
            raise ValueError("Invalid zbl line")
        zbl_rc_inner = float(tokens[1])
        zbl_rc_outer = float(tokens[2])
        zbl_flexible = zbl_rc_inner == 0.0 and zbl_rc_outer == 0.0

    tokens = lines[idx].split()
    idx += 1
    if tokens[0] != "cutoff" or len(tokens) not in (5, 8):
        raise ValueError("Invalid cutoff line")
    rc_radial = float(tokens[1])
    rc_angular = float(tokens[2])
    max_nn_radial = int(tokens[3])
    max_nn_angular = int(tokens[4])
    typewise_cutoff = None
    if len(tokens) == 8:
        typewise_cutoff = (float(tokens[5]), float(tokens[6]), float(tokens[7]))

    tokens = lines[idx].split()
    idx += 1
    if tokens[0] != "n_max" or len(tokens) != 3:
        raise ValueError("Invalid n_max line")
    n_max_radial = int(tokens[1])
    n_max_angular = int(tokens[2])

    tokens = lines[idx].split()
    idx += 1
    if tokens[0] != "basis_size" or len(tokens) != 3:
        raise ValueError("Invalid basis_size line")
    basis_size_radial = int(tokens[1])
    basis_size_angular = int(tokens[2])

    tokens = lines[idx].split()
    idx += 1
    if tokens[0] != "l_max" or len(tokens) != 4:
        raise ValueError("Invalid l_max line")
    l_max_3body = int(tokens[1])
    l_max_4body = int(tokens[2])
    l_max_5body = int(tokens[3])

    tokens = lines[idx].split()
    idx += 1
    if tokens[0] != "ANN" or len(tokens) != 3:
        raise ValueError("Invalid ANN line")
    num_neurons = int(tokens[1])

    nep = GPUMDNEP(
        version=version,
        model_type=model_type,
        num_types=num_types,
        type_names=type_names,
        rc_radial=rc_radial,
        rc_angular=rc_angular,
        max_nn_radial=max_nn_radial,
        max_nn_angular=max_nn_angular,
        typewise_cutoff_factors=typewise_cutoff,
        n_max_radial=n_max_radial,
        n_max_angular=n_max_angular,
        basis_size_radial=basis_size_radial,
        basis_size_angular=basis_size_angular,
        l_max_3body=l_max_3body,
        l_max_4body=l_max_4body,
        l_max_5body=l_max_5body,
        num_neurons=num_neurons,
        zbl_enabled=zbl_enabled,
        zbl_flexible=zbl_flexible,
        zbl_rc_inner=zbl_rc_inner,
        zbl_rc_outer=zbl_rc_outer,
    )

    counts = _compute_counts(nep)

    numeric_tokens: List[str] = []
    for ln in lines[idx:]:
        numeric_tokens.extend(ln.split())
    numeric_values = [float(x) for x in numeric_tokens]

    pos = 0
    need_params = counts["num_para"]
    if len(numeric_values) < need_params + counts["dim"]:
        raise ValueError("nep.txt does not contain enough numeric entries")

    parameters = torch.tensor(numeric_values[pos: pos + need_params], device=device, dtype=torch.float32)
    pos += need_params
    q_scaler = torch.tensor(numeric_values[pos: pos + counts["dim"]], device=device, dtype=torch.float32)
    pos += counts["dim"]

    zbl_parameters = None
    if nep.zbl_flexible:
        need_zbl = 10 * counts["num_type_zbl"]
        if len(numeric_values) < pos + need_zbl:
            raise ValueError("nep.txt lacks flexible ZBL parameters")
        zbl_parameters = torch.tensor(numeric_values[pos: pos + need_zbl], device=device, dtype=torch.float32)

    nep.parameters = parameters
    nep.q_scaler = q_scaler
    nep.zbl_parameters = zbl_parameters
    return nep


def _write_per_line(f, values: torch.Tensor):
    flat = values.reshape(-1).cpu().tolist()
    for x in flat:
        f.write(f"{float(x):15.7e}\n")


def write_gpumd_nep_txt(nep: GPUMDNEP, path: str | Path) -> str:
    """Write a GPUMD-compatible nep.txt file."""
    counts = _compute_counts(nep)

    if nep.parameters is None or nep.parameters.numel() != counts["num_para"]:
        raise ValueError("parameters size does not match model shape")
    if nep.q_scaler is None or nep.q_scaler.numel() != counts["dim"]:
        raise ValueError("q_scaler size does not match descriptor dimension")
    if nep.zbl_flexible:
        need_zbl = 10 * counts["num_type_zbl"]
        if nep.zbl_parameters is None or nep.zbl_parameters.numel() != need_zbl:
            raise ValueError("flexible ZBL parameters size mismatch")

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    tag = _model_tag_from_meta(nep)

    with p.open("w", encoding="ascii") as f:
        f.write(f"{tag} {nep.num_types} {' '.join(nep.type_names)}\n")

        if nep.zbl_enabled:
            if nep.zbl_flexible:
                f.write("zbl 0 0\n")
            else:
                f.write(f"zbl {nep.zbl_rc_inner:g} {nep.zbl_rc_outer:g}\n")

        if nep.typewise_cutoff_factors:
            cr, ca, cz = nep.typewise_cutoff_factors
            f.write(
                f"cutoff {nep.rc_radial:g} {nep.rc_angular:g} {nep.max_nn_radial} {nep.max_nn_angular} "
                f"{cr:g} {ca:g} {cz:g}\n"
            )
        else:
            f.write(f"cutoff {nep.rc_radial:g} {nep.rc_angular:g} {nep.max_nn_radial} {nep.max_nn_angular}\n")

        f.write(f"n_max {nep.n_max_radial} {nep.n_max_angular}\n")
        f.write(f"basis_size {nep.basis_size_radial} {nep.basis_size_angular}\n")
        f.write(f"l_max {nep.l_max_3body} {nep.l_max_4body} {nep.l_max_5body}\n")
        f.write(f"ANN {nep.num_neurons} 0\n")

        _write_per_line(f, nep.parameters)
        _write_per_line(f, nep.q_scaler)

        if nep.zbl_flexible and nep.zbl_parameters is not None:
            _write_per_line(f, nep.zbl_parameters)

    return str(p)


# -------------------- GPUMD NEP3 <-> PyTorch model --------------------------

def _ensure_model_q_scaler(model: NEP3Model):
    """Attach a q_scaler buffer (ones) if the model does not provide one."""
    if getattr(model, "q_scaler", None) is None:
        model.register_buffer("q_scaler", torch.ones(model.descriptor_dim, device=next(model.parameters()).device))
    else:
        # shape guard
        if model.q_scaler.numel() != model.descriptor_dim:
            raise ValueError("model.q_scaler size mismatch")


def gpumd_nep3_to_torch_model(
    nep: GPUMDNEP,
    device: Optional[torch.device] = None,
    neighbor_backend: str = "ase",
) -> NEP3Model:
    """
    Convert a GPUMD NEP3 potential (version=3, model_type=0) into a PyTorch NEP3Model.
    """
    if nep.version != 3 or nep.model_type != 0:
        raise ValueError("Only GPUMD NEP3 potential (version=3, model_type=0) is supported.")
    if nep.zbl_enabled:
        raise ValueError("ZBL is not supported by this PyTorch NEP3Model converter.")
    if nep.parameters is None or nep.q_scaler is None:
        raise ValueError("nep.parameters/q_scaler must be populated (use read_gpumd_nep_txt first).")

    # Instantiate model
    model = NEP3Model(
        n_max_radial=nep.n_max_radial,
        n_max_angular=nep.n_max_angular,
        basis_size_radial=nep.basis_size_radial,
        basis_size_angular=nep.basis_size_angular,
        l_max=nep.l_max_3body,
        l_max_4body=nep.l_max_4body,
        l_max_5body=nep.l_max_5body,
        rc_radial=nep.rc_radial,
        rc_angular=nep.rc_angular,
        num_neurons=nep.num_neurons,
        num_types=nep.num_types,
        neighbor_backend=neighbor_backend,
    )
    if device is not None:
        model = model.to(device)

    counts = _compute_counts(nep)
    dim = counts["dim"]
    num_para_ann = counts["num_para_ann"]
    num_para_descriptor = counts["num_para_descriptor"]

    params = nep.parameters.to(dtype=torch.float32, device=next(model.parameters()).device)
    ann = params[:num_para_ann]
    desc = params[num_para_ann:num_para_ann + num_para_descriptor]

    # ---- ANN unpack (GPUMD order) ----
    # w1 (num_neurons x dim), b1 (num_neurons), w2 (num_neurons), b2 (1)
    pos = 0
    w1 = ann[pos: pos + nep.num_neurons * dim].reshape(nep.num_neurons, dim); pos += nep.num_neurons * dim
    b1 = ann[pos: pos + nep.num_neurons].reshape(nep.num_neurons); pos += nep.num_neurons
    w2 = ann[pos: pos + nep.num_neurons].reshape(1, nep.num_neurons); pos += nep.num_neurons
    b2 = ann[pos: pos + 1].reshape(1)

    # Load into model (bias-free linears + explicit -b)
    if model.lin1.weight.shape != w1.shape:
        raise ValueError(f"lin1.weight shape mismatch: {model.lin1.weight.shape} vs {w1.shape}")
    model.lin1.weight.data.copy_(w1)
    model.b1.data.copy_(b1)
    model.lin2.weight.data.copy_(w2)
    model.b2.data.copy_(b2)

    # ---- Descriptor coefficients ----
    # GPUMD: [types^2, (n_r+1)*(basis_r+1)] then [types^2, (n_a+1)*(basis_a+1)]
    n_types_sq = nep.num_types * nep.num_types
    n_r = nep.n_max_radial + 1
    k_r = nep.basis_size_radial + 1
    n_a = nep.n_max_angular + 1
    k_a = nep.basis_size_angular + 1

    pos = 0
    cr_size = n_types_sq * n_r * k_r
    ca_size = n_types_sq * n_a * k_a

    c_radial = desc[pos: pos + cr_size].reshape(n_types_sq, n_r, k_r); pos += cr_size
    c_angular_base = desc[pos: pos + ca_size].reshape(n_types_sq, n_a, k_a); pos += ca_size

    if model.c_radial.shape != c_radial.shape:
        raise ValueError(f"c_radial shape mismatch: {model.c_radial.shape} vs {c_radial.shape}")
    if model.c_angular_base.shape != c_angular_base.shape:
        raise ValueError(f"c_angular_base shape mismatch: {model.c_angular_base.shape} vs {c_angular_base.shape}")

    model.c_radial.data.copy_(c_radial)
    model.c_angular_base.data.copy_(c_angular_base)

    # Atomic reference energies are not stored in GPUMD NEP3; keep zeros unless you manage offsets externally.
    model.atomic_energies.data.zero_()

    # q_scaler
    _ensure_model_q_scaler(model)
    if model.q_scaler.numel() != nep.q_scaler.numel():
        raise ValueError("q_scaler length mismatch")
    model.q_scaler.data.copy_(nep.q_scaler.to(device=model.q_scaler.device, dtype=torch.float32))

    return model


def torch_model_to_gpumd_nep3(
    model: NEP3Model,
    type_names: List[str],
    max_nn_radial: int = 200,
    max_nn_angular: int = 200,
    typewise_cutoff_factors: Optional[Tuple[float, float, float]] = None,
) -> GPUMDNEP:
    """
    Pack a PyTorch NEP3Model into an official GPUMD NEP3 `nep.txt` container.
    """
    if len(type_names) != int(model.num_types):
        raise ValueError("type_names length must match model.num_types")

    nep = GPUMDNEP(
        version=3,
        model_type=0,
        num_types=int(model.num_types),
        type_names=type_names,
        rc_radial=float(model.rc_radial),
        rc_angular=float(model.rc_angular),
        max_nn_radial=int(max_nn_radial),
        max_nn_angular=int(max_nn_angular),
        typewise_cutoff_factors=typewise_cutoff_factors,
        n_max_radial=int(model.n_max_radial),
        n_max_angular=int(model.n_max_angular),
        basis_size_radial=int(model.basis_size_radial),
        basis_size_angular=int(model.basis_size_angular),
        l_max_3body=int(model.l_max),
        l_max_4body=int(getattr(model, "l_max_4body", 0)),
        l_max_5body=int(getattr(model, "l_max_5body", 0)),
        num_neurons=int(model.num_neurons),
        zbl_enabled=False,
        zbl_flexible=False,
    )

    counts = _compute_counts(nep)
    dim = counts["dim"]

    # Ensure q_scaler exists
    _ensure_model_q_scaler(model)

    # ---- ANN pack ----
    w1 = model.lin1.weight.reshape(-1)
    b1 = model.b1.reshape(-1)
    w2 = model.lin2.weight.reshape(-1)
    b2 = model.b2.reshape(-1)
    ann = torch.cat([w1, b1, w2, b2], dim=0).to(dtype=torch.float32)

    if ann.numel() != counts["num_para_ann"]:
        raise ValueError("Packed ANN size mismatch; check model/metadata.")

    # ---- Descriptor coefficients ----
    desc = torch.cat([model.c_radial.reshape(-1), model.c_angular_base.reshape(-1)], dim=0).to(dtype=torch.float32)
    if desc.numel() != counts["num_para_descriptor"]:
        raise ValueError("Packed descriptor size mismatch; check model/metadata.")

    nep.parameters = torch.cat([ann, desc], dim=0).to(dtype=torch.float32)
    nep.q_scaler = model.q_scaler.reshape(-1).to(dtype=torch.float32)

    if nep.parameters.numel() != counts["num_para"]:
        raise ValueError("Total parameter count mismatch after packing.")
    if nep.q_scaler.numel() != dim:
        raise ValueError("q_scaler length mismatch after packing.")

    return nep


def load_gpumd_nep3_model(
    path: str | Path,
    device: Optional[torch.device] = None,
    neighbor_backend: str = "ase",
) -> Tuple[NEP3Model, GPUMDNEP]:
    """Load an official GPUMD NEP3 nep.txt into a PyTorch model."""
    nep = read_gpumd_nep_txt(path, device=device)
    model = gpumd_nep3_to_torch_model(nep, device=device, neighbor_backend=neighbor_backend)
    return model, nep


def save_gpumd_nep3_model(
    model: NEP3Model,
    path: str | Path,
    type_names: List[str],
    max_nn_radial: int = 200,
    max_nn_angular: int = 200,
    typewise_cutoff_factors: Optional[Tuple[float, float, float]] = None,
) -> str:
    """Save a PyTorch NEP3Model to an official GPUMD nep.txt."""
    nep = torch_model_to_gpumd_nep3(
        model=model,
        type_names=type_names,
        max_nn_radial=max_nn_radial,
        max_nn_angular=max_nn_angular,
        typewise_cutoff_factors=typewise_cutoff_factors,
    )
    return write_gpumd_nep_txt(nep, path)


# ---------------------------- Torch-friendly I/O ----------------------------

def _write_flat(f, values: torch.Tensor, cols: int = 5):
    flat = values.reshape(-1).cpu().numpy()
    for i in range(0, flat.size, cols):
        chunk = flat[i: i + cols]
        f.write(" ".join(f"{x:.8e}" for x in chunk) + "\n")


def _read_block(lines: List[str], idx: int):
    vals: List[float] = []
    while idx < len(lines):
        line = lines[idx]
        if line and line[0].isalpha():
            break
        vals.extend(float(x) for x in line.split())
        idx += 1
    return vals, idx


def save_nep_torch_txt(
    model: NEP3Model,
    path: str | Path,
    type_names: Optional[List[str]] = None,
) -> str:
    """
    Save parameters to a simple text format (NOT GPUMD native).
    Intended for debugging / unit tests.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if type_names is None:
        type_names = [f"T{i}" for i in range(int(model.num_types))]

    _ensure_model_q_scaler(model)

    with p.open("w", encoding="ascii") as f:
        f.write(f"nep3_torch {int(model.num_types)} {' '.join(type_names)}\n")
        f.write(f"cutoff {float(model.rc_radial):g} {float(model.rc_angular):g}\n")
        f.write(f"n_max {int(model.n_max_radial)} {int(model.n_max_angular)}\n")
        f.write(f"basis_size {int(model.basis_size_radial)} {int(model.basis_size_angular)}\n")
        f.write(f"l_max {int(model.l_max)} {int(getattr(model,'l_max_4body',0))} {int(getattr(model,'l_max_5body',0))}\n")
        f.write(f"ANN {int(model.num_neurons)}\n")
        f.write(f"descriptor_dim {int(model.descriptor_dim)}\n")

        f.write("q_scaler\n")
        _write_flat(f, model.q_scaler)

        f.write("atomic_energies\n")
        _write_flat(f, model.atomic_energies)

        f.write("c_radial\n")
        _write_flat(f, model.c_radial)

        f.write("c_angular_base\n")
        _write_flat(f, model.c_angular_base)

        f.write("ann_w1\n")
        _write_flat(f, model.lin1.weight)

        f.write("ann_b1\n")
        _write_flat(f, model.b1)

        f.write("ann_w2\n")
        _write_flat(f, model.lin2.weight)

        f.write("ann_b2\n")
        _write_flat(f, model.b2)

    return str(p)


def load_nep_torch_txt(path: str | Path, device: Optional[torch.device] = None) -> Tuple[NEP3Model, List[str]]:
    """
    Load a file produced by `save_nep_torch_txt`.
    Returns (model, type_names).
    """
    lines = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    idx = 0

    tokens = lines[idx].split()
    if tokens[0] != "nep3_torch":
        raise ValueError("Invalid header (expected nep3_torch)")
    num_types = int(tokens[1])
    type_names = tokens[2:]
    idx += 1

    rc_line = lines[idx].split(); idx += 1
    rc_radial, rc_angular = float(rc_line[1]), float(rc_line[2])

    n_line = lines[idx].split(); idx += 1
    n_max_r, n_max_a = int(n_line[1]), int(n_line[2])

    b_line = lines[idx].split(); idx += 1
    basis_r, basis_a = int(b_line[1]), int(b_line[2])

    l_line = lines[idx].split(); idx += 1
    l_max_3, l_max_4, l_max_5 = int(l_line[1]), int(l_line[2]), int(l_line[3])

    ann_line = lines[idx].split(); idx += 1
    num_neurons = int(ann_line[1])

    _ = int(lines[idx].split()[1]); idx += 1  # descriptor_dim

    model = NEP3Model(
        n_max_radial=n_max_r,
        n_max_angular=n_max_a,
        basis_size_radial=basis_r,
        basis_size_angular=basis_a,
        l_max=l_max_3,
        l_max_4body=l_max_4,
        l_max_5body=l_max_5,
        rc_radial=rc_radial,
        rc_angular=rc_angular,
        num_neurons=num_neurons,
        num_types=num_types,
    )
    if device is not None:
        model = model.to(device)

    # q_scaler
    if lines[idx] != "q_scaler":
        raise ValueError("Missing q_scaler block")
    idx += 1
    vals, idx = _read_block(lines, idx)
    _ensure_model_q_scaler(model)
    model.q_scaler.data.copy_(torch.tensor(vals, device=model.q_scaler.device, dtype=torch.float32).reshape(model.q_scaler.shape))

    # atomic energies
    if lines[idx] != "atomic_energies":
        raise ValueError("Missing atomic_energies block")
    idx += 1
    vals, idx = _read_block(lines, idx)
    model.atomic_energies.data.copy_(torch.tensor(vals, device=model.atomic_energies.device, dtype=torch.float32).reshape(model.atomic_energies.shape))

    # c_radial
    if lines[idx] != "c_radial":
        raise ValueError("Missing c_radial block")
    idx += 1
    vals, idx = _read_block(lines, idx)
    model.c_radial.data.copy_(torch.tensor(vals, device=model.c_radial.device, dtype=torch.float32).reshape(model.c_radial.shape))

    # c_angular_base
    if lines[idx] != "c_angular_base":
        raise ValueError("Missing c_angular_base block")
    idx += 1
    vals, idx = _read_block(lines, idx)
    model.c_angular_base.data.copy_(torch.tensor(vals, device=model.c_angular_base.device, dtype=torch.float32).reshape(model.c_angular_base.shape))

    # ANN params
    if lines[idx] != "ann_w1":
        raise ValueError("Missing ann_w1 block")
    idx += 1
    vals, idx = _read_block(lines, idx)
    model.lin1.weight.data.copy_(torch.tensor(vals, device=model.lin1.weight.device, dtype=torch.float32).reshape(model.lin1.weight.shape))

    if lines[idx] != "ann_b1":
        raise ValueError("Missing ann_b1 block")
    idx += 1
    vals, idx = _read_block(lines, idx)
    model.b1.data.copy_(torch.tensor(vals, device=model.b1.device, dtype=torch.float32).reshape(model.b1.shape))

    if lines[idx] != "ann_w2":
        raise ValueError("Missing ann_w2 block")
    idx += 1
    vals, idx = _read_block(lines, idx)
    model.lin2.weight.data.copy_(torch.tensor(vals, device=model.lin2.weight.device, dtype=torch.float32).reshape(model.lin2.weight.shape))

    if lines[idx] != "ann_b2":
        raise ValueError("Missing ann_b2 block")
    idx += 1
    vals, idx = _read_block(lines, idx)
    model.b2.data.copy_(torch.tensor(vals, device=model.b2.device, dtype=torch.float32).reshape(model.b2.shape))

    return model, type_names