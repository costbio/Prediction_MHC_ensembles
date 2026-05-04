import os
import mdtraj as md
import numpy as np


def save_xtc(ref, coords, out_path):
    coords = np.asarray(coords, dtype=np.float32)
    # coords shape control
    # expected shape: (n_frames, n_atoms,3)
    if coords.ndim != 3:
        raise ValueError(
            f"coords must have shape (n_frames, n_atoms, 3), got {coords.shape}"
        )

    if coords.shape[1] != ref.n_atoms:
        raise ValueError(
            f"Atom mismatch: coords has {coords.shape[1]} atoms, but ref has {ref.n_atoms}"
        )

    if coords.shape[2] != 3:
        raise ValueError(
            f"coords last dimension must be 3 for xyz coordinates, got {coords.shape[2]}"
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    traj = md.Trajectory(coords, ref.topology)
    traj.save_xtc(out_path)