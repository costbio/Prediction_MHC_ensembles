import numpy as np
import mdtraj as md
from sklearn.preprocessing import StandardScaler
import os


def load_and_align(xtc_file, pdb_file, atom_selection, protein_id, rep_num):
    traj = md.load_xtc(xtc_file, top=pdb_file)
    ref = md.load_pdb(pdb_file)

    sel = ref.topology.select(atom_selection)
    traj.superpose(ref, atom_indices = sel) 
    traj = traj.atom_slice(sel)
    ref = ref.atom_slice(sel)


    traj.save_xtc(f"../../data/palantir_data/{protein_id}/{protein_id}_rep_{rep_num}_backbone.xtc")
    ref.save_pdb(f"../../data/palantir_data/{protein_id}/backbone.pdb")


    ref_out = f"../../data/palantir_data/{protein_id}/backbone.pdb"
    if not os.path.exists(ref_out):
        ref.save_pdb(ref_out)
   
    print(f"[INFO] Frames: {traj.n_frames}, Atoms: {traj.n_atoms}")
    return traj, ref

def split_dataset(
    coords,
    discard_frames=150,
    use_all_frames=True,
    train_fraction=0.4,
    val_fraction=0.1,
    shuffle_learn=True,
    seed=42
):
    """
    Controlled ensemble-based split:

    - Optionally discard equilibration frames
    - Take the FIRST fraction of frames as learn set
    - Optionally shuffle within learn set
    - Remaining frames are test set
    """

    np.random.seed(seed)

    n_frames = coords.shape[0]

    # 1- Optional equilibration discard
    if not use_all_frames:
        assert discard_frames < n_frames, "discard_frames must be < total frames"
        usable_coords = coords[discard_frames:]
    else:
        usable_coords = coords

    n_usable = usable_coords.shape[0]

    # 2- Learn/Test split (NO random sampling)
    n_learn = int(train_fraction * n_usable)
    learn_coords = usable_coords[:n_learn]
    X_test = usable_coords[n_learn:]

    # 3- Optional shuffle inside learn set
    if shuffle_learn:
        perm = np.random.permutation(len(learn_coords))
        learn_coords = learn_coords[perm]

    # 4- Train / Validation split
    n_train = int((1 - val_fraction) * len(learn_coords))
    X_train = learn_coords[:n_train]
    X_val   = learn_coords[n_train:]

    return X_train, X_val, X_test




def scale_data(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler
