import matplotlib.pyplot as plt
import numpy as np
import torch
import json

def plot_training_dynamics(history, save_path=None, suptitle=None):
    epochs = np.arange(len(history["train_recon"]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)


    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=14, y=1.05)

    # 1️⃣ Reconstruction
    axes[0].plot(epochs, history["train_recon"], label="Train")
    axes[0].plot(epochs, history["val_recon"], label="Val")
    axes[0].set_title("Reconstruction Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # 2️⃣ KL
    axes[1].plot(epochs, history["train_kl"], label="Train")
    axes[1].plot(epochs, history["val_kl"], label="Val")
    axes[1].set_title("KL Divergence")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True)

    # 3️⃣ Beta
    axes[2].plot(epochs, history["beta"], label="Beta")
    axes[2].set_title("Beta Schedule")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


fractions = list(range(10, 90, 10))

def model_accuracy_results(out_dir,out_dir_1, out_dir_2):
    experiment_dirs = {
        "rep_0": out_dir,
        "rep_1": out_dir_1,
        "rep_2": out_dir_2,
    }

    for rep_name, base_dir in experiment_dirs.items():

        for frac in fractions:
            history_path = f"{base_dir}/fraction_{frac}/history_{frac}.npy"
            save_path = f"{base_dir}/fraction_{frac}/training_dynamics.png"

            history = np.load(history_path, allow_pickle=True).item()

            plot_training_dynamics(
                history,
                save_path=save_path,
                suptitle=f"{rep_name} | Training Dynamics (Data Fraction = {frac/100:.1f})"
            )
    return experiment_dirs

def model_performance_results(experiment_dirs):
    fractions_float = [f / 100 for f in fractions]

    for rep_name, base_dir in experiment_dirs.items():

        best_val_recon = []
        best_epochs = []
        prior_std = []

        # --- load prior std ---
        with open(f"{base_dir}/decoder_prior_std.json", "r") as f:
            prior_std_dict = json.load(f)

        for frac in fractions:
            frac_dir = f"{base_dir}/fraction_{frac}"

            ckpt = torch.load(f"{frac_dir}/best.ckpt", map_location="cpu")
            best_val_recon.append(ckpt["best_val_recon"])
            best_epochs.append(ckpt["epoch"])
            prior_std.append(prior_std_dict[str(frac)])

        # ===============================
        # SINGLE SUMMARY FIGURE
        # ===============================
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)

        # --- (1) Recon ---
        axes[0].plot(fractions_float, best_val_recon, marker="o")
        axes[0].set_xlabel("Training Data Fraction")
        axes[0].set_ylabel("Best Val Recon Loss")
        axes[0].set_title("Reconstruction")
        axes[0].grid(True)

        # --- (2) Epoch ---
        axes[1].plot(fractions_float, best_epochs, marker="o")
        axes[1].set_xlabel("Training Data Fraction")
        axes[1].set_ylabel("Best Epoch")
        axes[1].set_title("Early Stopping")
        axes[1].grid(True)

        # --- (3) Prior Std ---
        axes[2].plot(fractions_float, prior_std, marker="o")
        axes[2].set_xlabel("Training Data Fraction")
        axes[2].set_ylabel("Decoder Output Std (Å)")
        axes[2].set_title("Generative Diversity")
        axes[2].grid(True)

        fig.suptitle(
            f"{rep_name}: Model Performance vs Training Data Fraction",
            fontsize=14,
            y=1.05
        )

        plt.tight_layout()
        plt.savefig(
            f"{base_dir}/summary_performance.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

