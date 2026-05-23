import os
import json
import numpy as np
import torch
import joblib

from data.preprocessing import (
    load_and_align,
    load_preprocessed_traj,
    split_dataset,
    scale_data
)
from model.xyz_vae import XYZVAE
from training.train_loop import train_vae
from model_analysis.save_xtc import save_xtc
from utils.seed import set_seed
from inference.generate_trajectory import generate_filtered_trajectory


def run_pipeline(
    xtc_file,
    pdb_file,
    protein_id,
    rep_num,
    output_root,
    atom_selection,
    out_dir,
    already_preprocessed=False,
    latent_dim=8,
    epochs=300,
    batch_size=64,
    kl_beta=5e-5,
    use_all_frames=True,
    shuffle_train=True,
    seed=42,
    device="cuda",
    fractions=(10, 20, 30, 40, 50, 60, 70),
    temperature=1.5,
    target_n_frames=1000,
    generation_batch_size=64,
    max_generation_attempts=50000,
    rama_outlier_threshold=0.10,
    bond_bad_threshold=0.15,
):
    os.makedirs(out_dir, exist_ok=True)

    # 1) LOAD + ALIGN
    if already_preprocessed:
        traj, ref = load_preprocessed_traj(
            xtc_file,
            pdb_file
        )
    else:
        traj, ref = load_and_align(
            xtc_file,
            pdb_file,
            atom_selection,
            protein_id,
            rep_num,
            output_root
        )
    # frames * flattened coordinates vector 
    coords = traj.xyz.reshape(traj.n_frames, -1)

    fixed_val_idx = None
    fixed_test_idx = None

    # 2) FRACTION LOOP
    for frac_percent in fractions:
        frac = frac_percent / 100.0

        # deterministic seed
        frac_seed = int(seed + frac_percent)
        set_seed(frac_seed)

        out_dir_frac = os.path.join(out_dir, f"fraction_{frac_percent}")
        os.makedirs(out_dir_frac, exist_ok=True)

        X_train, X_val, X_test, train_idx, val_idx, test_idx = split_dataset(
            coords,
            discard_frames=150,
            use_all_frames=use_all_frames,
            train_fraction=frac,
            val_fraction=0.1,
            test_fraction=0.2,
            shuffle_train=shuffle_train,
            seed=frac_seed,  
        )

        print(
            f"[Fraction {frac_percent}] "
            f"Seed: {frac_seed} | "
            f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}"
        )

        if fixed_val_idx is None:
            fixed_val_idx = val_idx.copy()
            fixed_test_idx = test_idx.copy()
        else:
            if not np.array_equal(val_idx, fixed_val_idx):
                raise ValueError("Validation indices changed")
            if not np.array_equal(test_idx, fixed_test_idx):
                raise ValueError("Test indices changed")

        if frac_percent == min(fractions):
            test_traj = traj[test_idx]
            test_traj.save_xtc(os.path.join(out_dir, "fixed_md_test_subset.xtc"))

        #for debugging / reproducibility
        meta = {
            "protein_id": protein_id,
            "rep_num": rep_num,
            "fraction_percent": frac_percent,
            "frac_seed": frac_seed,
            "latent_dim": latent_dim,
            "epochs": epochs,
            "batch_size": batch_size,
            "kl_beta": kl_beta,
            "use_all_frames": use_all_frames,
            "shuffle_train": shuffle_train,
            "temperature": temperature,
            "train_size": int(len(train_idx)),
            "val_size": int(len(val_idx)),
            "test_size": int(len(test_idx)),
        }
        with open(os.path.join(out_dir_frac, "run_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # 3) SCALE
        X_train_s, X_val_s, X_test_s, scaler = scale_data(X_train, X_val, X_test)
        joblib.dump(scaler, os.path.join(out_dir_frac, "scaler.pkl"))

        # 4) MODEL
        model = XYZVAE(
            input_dim=X_train_s.shape[1],
            latent_dim=latent_dim
        ).to(device)

        # 5) TRAIN
        history = train_vae(
            model=model,
            X_train=X_train_s,
            X_val=X_val_s,
            epochs=epochs,
            batch_size=batch_size,
            kl_beta=kl_beta,
            device=device,
            out_dir=out_dir_frac,
            verbose=False,
            shuffle_batches=shuffle_train,   
            seed=frac_seed                   # deterministic batch order
        )

        np.save(os.path.join(out_dir_frac, f"history_{frac_percent}.npy"), history)

        # 7) FILTERED GENERATION

        gen_xyz, gen_stats, _ = generate_filtered_trajectory(
            model=model,
            scaler=scaler,
            ref=traj,
            target_n_frames=target_n_frames,
            device=device,
            seed=frac_seed,
            temperature=temperature,
            batch_size=generation_batch_size,
            max_attempts=max_generation_attempts,
            rama_outlier_threshold=rama_outlier_threshold,
            bond_bad_threshold=bond_bad_threshold,
            verbose=False,
        )

        # Save generated filtered trajectory
        save_xtc(
            ref,
            gen_xyz,
            os.path.join(out_dir_frac, f"generated_filtered_{frac_percent}.xtc")
        )

        # Save generation statistics
        with open(os.path.join(out_dir_frac, "generation_stats.json"), "w") as f:
            json.dump(gen_stats, f, indent=2)

    return
