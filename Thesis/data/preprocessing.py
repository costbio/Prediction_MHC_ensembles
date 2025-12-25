import numpy as np
import mdtraj as md
from sklearn.preprocessing import StandardScaler


def load_and_align(xtc_file, pdb_file, atom_selection):
    traj = md.load_xtc(xtc_file, top=pdb_file)
    ref = md.load_pdb(pdb_file)

    sel = ref.topology.select(atom_selection)
    traj = traj.atom_slice(sel)
    ref = ref.atom_slice(sel)

    traj.superpose(ref)
    print(f"[INFO] Frames: {traj.n_frames}, Atoms: {traj.n_atoms}")
    return traj, ref

def split_dataset(
    coords,
    early_fraction=0.4,
    train_ratio=0.8,
    val_ratio=0.1,
    seed=42
):
    np.random.seed(seed)

    n_frames = coords.shape[0]
    cut = int(early_fraction * n_frames)

    early_coords = coords[:cut]
    future_coords = coords[cut:]

    idx = np.random.permutation(len(early_coords))

    train_end = int(train_ratio * len(idx))
    val_end = int((train_ratio + val_ratio) * len(idx))

    train_idx = idx[:train_end]
    val_idx = idx[train_end:val_end]

    X_train = early_coords[train_idx]
    X_val   = early_coords[val_idx]
    X_test  = future_coords

    return X_train, X_val, X_test




def scale_data(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler
