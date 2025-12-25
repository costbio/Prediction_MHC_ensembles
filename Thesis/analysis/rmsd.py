import numpy as np
import mdtraj as md


def rmsd_real(pdb, xtc, sel):
    traj = md.load_xtc(xtc, top=pdb)
    ref = md.load_pdb(pdb)

    idx = ref.topology.select(sel)
    traj = traj.atom_slice(idx)
    ref = ref.atom_slice(idx)

    traj.superpose(ref)
    vals = md.rmsd(traj, ref)

    return summary(vals)


def rmsd_generated(pdb, xtc, sel):
    ref = md.load_pdb(pdb)
    idx = ref.topology.select(sel)
    ref = ref.atom_slice(idx)

    traj = md.load_xtc(xtc, top=ref)
    traj.superpose(ref)
    vals = md.rmsd(traj, ref)

    return summary(vals)


def summary(vals):
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals)),
        "max": float(np.max(vals)),
        "vals": vals
    }


