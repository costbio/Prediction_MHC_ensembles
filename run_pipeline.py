# ============================================================
# run_pipeline.py
# ============================================================

import os
import argparse
import numpy as np
import torch
import json



from data.preprocessing import (
    load_and_align,
    split_dataset,
    scale_data
)

from models.xyz_vae import XYZVAE
from training.train_loop import train_vae
from Model_analysis.save_xtc import save_xtc
from utils.seed import set_seed
from inference.generate_trajectory import generate_trajectory



# ============================================================
# 1) RUN PIPELINE
# ============================================================
def run_pipeline(
    xtc_file,
    pdb_file,
    protein_id,
    rep_num,
    atom_selection,
    out_dir,
    latent_dim=8,
    epochs=300,
    batch_size=64,
    kl_beta=5e-5,
    use_all_frames=True,
    shuffle_learn=True,
    seed=42,
    device="cuda"
):

    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)

    # --------------------------------------------------------
    # 1) LOAD + ALIGN
    # --------------------------------------------------------
    traj, ref = load_and_align(
        xtc_file,
        pdb_file,
        atom_selection,
        protein_id,
        rep_num

    )

    coords = traj.xyz.reshape(traj.n_frames, -1)

    decoder_prior_std = {}

    # --------------------------------------------------------
    # 2) SPLIT
    # --------------------------------------------------------
    for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        out_dir_frac = f"{out_dir}/fraction_{int(frac*100)}"
        os.makedirs(out_dir_frac, exist_ok=True)

        X_train, X_val, X_test = split_dataset(
            coords,
            discard_frames=150,
            use_all_frames=use_all_frames,
            train_fraction=frac,
            val_fraction=0.1,
            shuffle_learn=shuffle_learn,
            seed=42
    )

    # --------------------------------------------------------
    # 3) SCALE
    # --------------------------------------------------------
        X_train_s, X_val_s, X_test_s, scaler = scale_data(
            X_train, X_val, X_test
        )

    # --------------------------------------------------------
    # 4) MODEL
    # --------------------------------------------------------
        model = XYZVAE(
            input_dim=X_train_s.shape[1],
            latent_dim=latent_dim
        ).to(device)

    # --------------------------------------------------------
    # 5) TRAIN
    # --------------------------------------------------------
        history = train_vae(
            model=model,
            X_train=X_train_s,
            X_val=X_val_s,
            epochs=epochs,
            batch_size=batch_size,
            kl_beta=kl_beta,
            device=device,
            out_dir=out_dir_frac,
            verbose=False
        )
        np.save(f"{out_dir_frac}/history_{int(frac*100)}.npy", history)

        # --------------------------------------------------------
        # 6) GENERATE
        # --------------------------------------------------------
        model.eval()
        with torch.no_grad():
            torch.manual_seed(seed)

            z = torch.randn(1000, latent_dim, device=device)
            x = model.decode(z)

            x_np = x.cpu().numpy()
            x_inv = scaler.inverse_transform(x_np)
            x_inv = torch.tensor(x_inv, device=device)

            std_mean = x_inv.std(dim=0).mean().item()

        decoder_prior_std[int(frac * 100)] = std_mean

        traj_gen = generate_trajectory(
            model=model,
            n_samples=traj.n_frames,
            latent_dim=latent_dim,
            device=device,
            seed=seed
        )

        traj_np = traj_gen.numpy()
        traj_inv = scaler.inverse_transform(traj_np)
        decoded_xyz = traj_inv.reshape(-1, traj.n_atoms, 3)

        gen_xtc = os.path.join(
            out_dir_frac,
            f"generated_frac_{int(frac * 100)}.xtc"
        )

        save_xtc(ref, decoded_xyz, gen_xtc)

    with open(f"{out_dir}/decoder_prior_std.json", "w") as f:
        json.dump(decoder_prior_std, f, indent=2)
    return 

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--xtc", required=True)
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--atom_sel", default="name CA")
    parser.add_argument("--out_dir", default="xyz_vae_results")
    parser.add_argument("--latent_dim", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--kl_beta", type=float, default=5e-5)

    args = parser.parse_args()

    run_pipeline(
        xtc_file=args.xtc,
        pdb_file=args.pdb,
        atom_selection=args.atom_sel,
        out_dir=args.out_dir,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        kl_beta=args.kl_beta
    )
