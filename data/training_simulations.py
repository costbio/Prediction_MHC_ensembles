import os
import json
import numpy as np
import mdtraj as md

from data.preprocessing import split_dataset

def save_training_xtcs(
    protein_id,
    condition,
    replicas=(0, 1, 2),
    fractions=(10, 20, 30, 40, 50, 60, 70),
    base_data_dir="../data/palantir_data",
    output_dir="../Training_XTCs",
    seed=42,
    discard_frames=150,
):
    """
    Saves the exact unscaled training frames used in the pipeline as XTC.
    """

    condition_map = {
        "shuffle_allframes": {
            "use_all_frames": True,
            "shuffle_train": True,
        },
        "shuffle_not_allframes": {
            "use_all_frames": False,
            "shuffle_train": True,
        },
        "not_shuffle_allframes": {
            "use_all_frames": True,
            "shuffle_train": False,
        },
        "not_shuffle_not_allframes": {
            "use_all_frames": False,
            "shuffle_train": False,
        },
    }

    if condition not in condition_map:
        raise ValueError(
            f"Unknown condition: {condition}. "
            f"Choose from: {list(condition_map.keys())}"
        )

    use_all_frames = condition_map[condition]["use_all_frames"]
    shuffle_train = condition_map[condition]["shuffle_train"]

    pdb_path = os.path.join(base_data_dir, protein_id, "backbone.pdb")

    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"PDB not found: {pdb_path}")

    for rep in replicas:

        xtc_path = os.path.join(
            base_data_dir,
            protein_id,
            f"{protein_id}_rep_{rep}_backbone.xtc"
        )
        if not os.path.exists(xtc_path):
            print(f"[WARNING] Missing XTC: {xtc_path}")
            continue

        traj = md.load_xtc(xtc_path, top=pdb_path)
        coords = traj.xyz.reshape(traj.n_frames, -1)

        print(f"\n=== {protein_id} | rep {rep} | {condition} ===")
        print(f"Original frames: {traj.n_frames}")

        fixed_val_idx = None
        fixed_test_idx = None

        for frac_percent in fractions:

            frac = frac_percent / 100.0
            frac_seed = int(seed + frac_percent)

            X_train, X_val, X_test, train_idx, val_idx, test_idx = split_dataset(
                coords,
                discard_frames=discard_frames,
                use_all_frames=use_all_frames,
                train_fraction=frac,
                val_fraction=0.1,
                test_fraction=0.2,
                shuffle_train=shuffle_train,
                seed=frac_seed,
            )

            if fixed_val_idx is None:
                fixed_val_idx = val_idx.copy()
                fixed_test_idx = test_idx.copy()
            else:
                if not np.array_equal(val_idx, fixed_val_idx):
                    raise ValueError("Validation indices changed")
                if not np.array_equal(test_idx, fixed_test_idx):
                    raise ValueError("Test indices changed")

            train_traj = traj[train_idx]

            save_dir = os.path.join(
                output_dir,
                condition,
                protein_id,
                f"{protein_id}_rep_{rep}",
                f"fraction_{frac_percent}"
            )
            os.makedirs(save_dir, exist_ok=True)

            train_xtc_path = os.path.join(
                save_dir,
                f"training_{protein_id}_rep_{rep}_frac_{frac_percent}.xtc"
            )

            train_idx_path = os.path.join(
                save_dir,
                f"training_indices_{protein_id}_rep_{rep}_frac_{frac_percent}.npy"
            )

            metadata_path = os.path.join(save_dir, "training_xtc_metadata.json")

            train_traj.save_xtc(train_xtc_path)
            np.save(train_idx_path, train_idx)

            metadata = {
                "protein_id": protein_id,
                "replica": rep,
                "condition": condition,
                "fraction_percent": frac_percent,
                "frac_seed": frac_seed,
                "use_all_frames": use_all_frames,
                "shuffle_train": shuffle_train,
                "discard_frames": discard_frames,
                "n_original_frames": int(traj.n_frames),
                "n_train_frames": int(len(train_idx)),
                "n_val_frames": int(len(val_idx)),
                "n_test_frames": int(len(test_idx)),
                "note": (
                    "Coordinates are unscaled. "
                    "If shuffle_train=True, frame order is shuffled and should not be interpreted as MD time."
                ),
            }

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            print(
                f"[Fraction {frac_percent}] "
                f"Seed: {frac_seed} | "
                f"Train frames: {len(train_idx)} | "
                f"Saved: {train_xtc_path}"
            )

        pdb_save_dir = os.path.join(output_dir, condition, protein_id)
        os.makedirs(pdb_save_dir, exist_ok=True)

        pdb_save_path = os.path.join(pdb_save_dir, "backbone.pdb")
        if not os.path.exists(pdb_save_path):
            traj[0].save_pdb(pdb_save_path)