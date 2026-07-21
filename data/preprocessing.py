import os
import numpy as np
import mdtraj as md
from sklearn.preprocessing import StandardScaler


def load_and_align(xtc_file, pdb_file, atom_selection, protein_id, rep_num, output_root):
    """
    Load trajectory, align all protein atoms to the first frame,
    then slice the requested atom selection (backbone).

    Outputs:
    - aligned backbone trajectory
    - backbone reference pdb
    """
    traj = md.load_xtc(xtc_file, top=pdb_file)

    full_sel = traj.topology.select("protein")
    select = traj.topology.select(atom_selection)

    #control
    if len(full_sel) == 0:
        raise ValueError("No atoms found for full protein selection.")
    if len(select) == 0:
        raise ValueError(f"No atoms found for atom_selection: {atom_selection}")

    ref_frame = traj[0]
    # to avoid global rotation/translation
    traj.superpose(ref_frame, atom_indices=full_sel)

    traj = traj.atom_slice(select)
    ref = ref_frame.atom_slice(select)

    traj_out = os.path.join(output_root, f"{protein_id}_rep_{rep_num}_backbone.xtc")
    ref_out = os.path.join(output_root, "backbone.pdb")

    traj.save_xtc(traj_out)
    if not os.path.exists(ref_out):
        ref.save_pdb(ref_out)

    print(f"[INFO] Frames: {traj.n_frames}, Atoms: {traj.n_atoms}")
    return traj, ref

def load_preprocessed_traj(xtc_file, pdb_file):
    """
    Trajectory is already aligned / atom-selected, for frame-equalized.
    """
    traj = md.load_xtc(xtc_file, top=pdb_file)
    ref = md.load_pdb(pdb_file)

    print(f"[INFO] Loaded preprocessed trajectory")
    print(f"[INFO] Frames: {traj.n_frames}, Atoms: {traj.n_atoms}")

    return traj, ref


def split_dataset(
    
    coords,
    discard_frames=150,
    use_all_frames=True,
    train_fraction=0.4,
    val_fraction=0.1,
    test_fraction=0.2,
    shuffle_train=True,
    seed=42
):
    """
    [ train (from beginning) | unused gap | validation (fixed, near end) | test (fixed, end) ]
    """

    n_frames = coords.shape[0]


    if not use_all_frames:

        usable_idx = np.arange(discard_frames, n_frames)
    else:
        usable_idx = np.arange(n_frames)

    n_usable = len(usable_idx)
    n_train = int(train_fraction * n_usable)
    n_val = int(val_fraction * n_usable)
    n_test = int(test_fraction * n_usable)

    test_start = n_usable - n_test
    val_start = test_start - n_val


    train_idx = usable_idx[:n_train]
    val_idx = usable_idx[val_start:test_start]
    test_idx = usable_idx[test_start:]

    if shuffle_train:
        rng = np.random.default_rng(seed)
        train_idx = rng.permutation(train_idx)

    X_train = coords[train_idx]
    X_val = coords[val_idx]
    X_test = coords[test_idx]

    return X_train, X_val, X_test, train_idx, val_idx, test_idx


def scale_data(X_train, X_val, X_test):
    """
    Fit scaler on train only, then transform val and test.
    """
    scaler = StandardScaler()
    #with fit_transform, learn mean and std from traain dataset
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler