from pathlib import Path
from ase.io import read
from ase_tornep import NEP3Calculator
import numpy as np


# ----------- inputs -----------
# --- paths (robust) ---
HERE = Path(__file__).resolve().parent          # .../clean/test
ROOT = HERE.parent                               # .../clean

# 如果 model.xyz / C_2022_NEP3.txt 放在 clean/ 根目录：
xyz_path = HERE / "model.xyz"
nep_path = HERE / "C_2022_NEP3.txt"

# ----------- read structure -----------
atoms = read(xyz_path)

# Case A: 你的 xyz 是 extxyz，里面自带 Lattice/pbc（推荐）
# Case B: 普通 xyz 没有 cell/pbc -> 需要你手动设置，否则会走 O(N^2) 邻居，且周期性不对
if (atoms.cell.volume == 0) or (atoms.pbc is None) or (not atoms.pbc.any()):
    # 下面给一个“合理默认”：石墨烯常用 2D 周期 + z 方向真空
    # !!! 如果你有真实晶胞参数，请把 cell 改成你的 !!!
    # 例：a=2.46 Å 的六角晶格，真空 20 Å
    a = 2.46
    vacuum = 20.0
    atoms.set_cell([[a, 0, 0],
                    [a/2, a*(3**0.5)/2, 0],
                    [0, 0, vacuum]])
    atoms.set_pbc([True, True, False])

# ----------- attach calculator -----------
# 若你的势文件里没有 type_names（只支持 C），就显式给 type_map={"C":0}
calc = NEP3Calculator(
    nep_path=nep_path,
    device="cuda",             # 没有 GPU 就改成 "cpu"
    neighbor_backend="ase",    # 周期体系强烈推荐
    type_map={"C": 0},
)

atoms.calc = calc

# ----------- compute -----------
E = atoms.get_potential_energy()     # eV
F = atoms.get_forces()               # eV/Å

print("Natoms =", len(atoms))
print("PBC    =", atoms.pbc)
print("Cell(Å) =\n", atoms.cell.array)
print("E (eV) =", E)
print("E/atom (eV) =", E/len(atoms))
print("F stats (eV/Å): |F|max =", np.sqrt((F**2).sum(axis=1)).max())

