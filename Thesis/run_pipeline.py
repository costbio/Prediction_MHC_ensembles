# ============================================================
# run_pipeline.py
# ============================================================

import os
import argparse
import numpy as np
import torch

from data.preprocessing import (
    load_and_align,
    split_dataset,
    scale_data
)

from models.xyz_vae import XYZVAE
from training.train_loop import train_vae
from analysis.save_xtc import save_xtc
from analysis.rmsd import rmsd_real, rmsd_generated
from utils.seed import set_seed


# ============================================================
# 1) RUN PIPELINE
# ============================================================
def run_pipeline(
    xtc_file,
    pdb_file,
    atom_selection,
    out_dir,
    latent_dim=64,
    epochs=150,
    batch_size=64,
    kl_beta=1e-4,
    train_ratio=0.8,
    val_ratio=0.1,
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
        atom_selection
    )

    coords = traj.xyz.reshape(traj.n_frames, -1)

    # --------------------------------------------------------
    # 2) SPLIT
    # --------------------------------------------------------
    for frac in [0.1, 0.2, 0.4, 0.6]:
        print(f"\n[RUN] Early fraction = {frac}")
        out_dir_frac = f"{out_dir}/fraction_{int(frac*100)}"
        os.makedirs(out_dir_frac, exist_ok=True)

        X_train, X_val, X_test = split_dataset(
            coords,
            early_fraction=frac)

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
            out_dir=out_dir
        )
        np.save(f"{out_dir_frac}/history_{int(frac*100)}.npy", history)

        # --------------------------------------------------------
        # 6) GENERATE
        # --------------------------------------------------------
        model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(
                X_test_s, dtype=torch.float32
            ).to(device)

            x_hat, _, _ = model(X_test_t)
            decoded = x_hat.cpu().numpy()


        decoded_inv = scaler.inverse_transform(decoded)
        decoded_xyz = decoded_inv.reshape(
            -1, traj.n_atoms, 3
        )

        gen_xtc = os.path.join(out_dir, f"generated_test_{int(frac*100)}.xtc")
        save_xtc(ref, decoded_xyz, gen_xtc)

        # --------------------------------------------------------
        # 7) RMSD
        # --------------------------------------------------------
        real_stats = rmsd_real(
            pdb_file, xtc_file, atom_selection
        )

        gen_stats = rmsd_generated(
            pdb_file, gen_xtc, atom_selection
        )

        # --------------------------------------------------------
        # 8) SAVE METRICS
        # --------------------------------------------------------
        np.savetxt(
            os.path.join(out_dir, f"rmsd_real_vals_{int(frac*100)}.txt"),
            real_stats["vals"]
        )

        np.savetxt(
            os.path.join(out_dir, f"rmsd_gen_vals_{int(frac*100)}.txt"),
            gen_stats["vals"]
        )

        with open(os.path.join(out_dir, f"rmsd_summary_{int(frac*100)}.txt"), "w") as f:
            f.write("REAL\n")
            f.write(str(real_stats) + "\n\n")
            f.write("GENERATED\n")
            f.write(str(gen_stats))

        print("Pipeline finished successfully.")
    return history 


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--xtc", required=True)
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--atom_sel", default="name CA")
    parser.add_argument("--out_dir", default="xyz_vae_results")
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--kl_beta", type=float, default=2e-5)

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
