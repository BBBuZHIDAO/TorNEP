from pathlib import Path
import time
import numpy as np

from ase import units
from ase.build import graphene
from ase.io import write
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation

from ase_tornep import NEP3Calculator


def main():
    # -------- paths --------
    HERE = Path(__file__).resolve().parent      # .../clean/test
    ROOT = HERE.parent                          # .../clean
    nep_path = HERE / "C_2022_NEP3.txt"

    # -------- build monolayer graphene (primitive cell) --------
    # a: graphene lattice constant (~2.46 Å); vacuum: add z cell and center
    a = 2.46
    vacuum = 20.0  # Å

    atoms = graphene(a=a)  # primitive graphene (2 atoms)
    atoms.pbc = (True, True, False)
    atoms.center(vacuum=vacuum, axis=2)  # add vacuum along z and center atoms

    # -------- replicate 10x10 --------
    atoms = atoms.repeat((2, 2, 1))

    # -------- attach NEP3 calculator --------
    atoms.calc = NEP3Calculator(
        nep_path=str(nep_path),
        device="cuda",              # 没有 GPU 就改 "cpu"
        neighbor_backend="ase",     # 周期体系推荐
        type_map={"C": 0},          # 只含 C 的 NEP 一般这样；若势文件自带 type_names 可不传
    )

    # -------- initial velocities (optional but recommended for NVE) --------
    T0 = 300.0  # K，按需修改
    MaxwellBoltzmannDistribution(atoms, temperature_K=T0)
    Stationary(atoms)      # remove COM drift
    ZeroRotation(atoms)    # remove overall rotation

    # -------- NVE (VelocityVerlet) --------
    dt_fs = 1.0  # fs，按需修改
    dyn = VelocityVerlet(atoms, dt_fs * units.fs)

    # dump 每步写一次（extxyz追加）
    dump_xyz = ROOT / "gr_md_dump.xyz"
    if dump_xyz.exists():
        dump_xyz.unlink()

    def thermo_and_dump():
        step = dyn.nsteps
        Epot = atoms.get_potential_energy()
        Ekin = atoms.get_kinetic_energy()
        Etot = Epot + Ekin
        T = atoms.get_temperature()
        print(f"step {step:5d}  Epot {Epot:14.6f} eV  Ekin {Ekin:14.6f} eV  Etot {Etot:14.6f} eV  T {T:10.2f} K")
        write(str(dump_xyz), atoms, append=True)

    dyn.attach(thermo_and_dump, interval=1)

    nsteps = 1
    t0 = time.perf_counter()
    dyn.run(nsteps)
    t1 = time.perf_counter()

    # -------- performance: atom*step/second --------
    elapsed = t1 - t0
    aps = len(atoms) * nsteps / elapsed
    print(f"\nNatoms = {len(atoms)}")
    print(f"MD wall time: {elapsed:.6f} s")
    print(f"Performance: {aps:.3f} atom*step/s")

    # -------- save final velocities + final structure --------
    v = atoms.get_velocities()  # Å/fs
    np.savetxt(
        str(ROOT / f"final_velocities_step{nsteps}.txt"),
        v,
        header="vx(Ang/fs) vy(Ang/fs) vz(Ang/fs)"
    )
    write(str(ROOT / f"gr_md_final_step{nsteps}.xyz"), atoms)

    print(f"\nSaved dump: {dump_xyz}")
    print(f"Saved final velocities: final_velocities_step{nsteps}.txt")
    print(f"Saved final structure: gr_md_final_step{nsteps}.xyz")


if __name__ == "__main__":
    main()
