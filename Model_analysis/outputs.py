import matplotlib.pyplot as plt
import numpy as np
import torch
import json

def plot_training_dynamics_reps(
    histories,  # dict: {"rep_0": history, ...}
    save_path=None,
    suptitle=None
):
    epochs = np.arange(len(next(iter(histories.values()))["train_recon"]))

    fig, axes = plt.subplots(1, 2, figsize=(15, 4), sharex=True)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=1.05)

    for rep_name, history in histories.items():
        # Recon
        axes[0].plot(
            epochs,
            history["train_recon"],
            label=f"{rep_name} Train",
            alpha=0.8
        )
        axes[0].plot(
            epochs,
            history["val_recon"],
            linestyle="--",
            label=f"{rep_name} Val",
            alpha=0.8
        )

        # KL
        axes[1].plot(
            epochs,
            history["train_kl"],
            label=f"{rep_name} Train",
            alpha=0.8
        )
        axes[1].plot(
            epochs,
            history["val_kl"],
            linestyle="--",
            label=f"{rep_name} Val",
            alpha=0.8
        )

    axes[0].set_title("Reconstruction Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(True)

    axes[1].set_title("KL Divergence")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(fontsize=8)
    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


fractions = list(range(10, 90, 10))
def model_accuracy_results_combined(experiment_dirs):
    for frac in fractions:
        histories = {}

        for rep_name, base_dir in experiment_dirs.items():
            history_path = f"{base_dir}/fraction_{frac}/history_{frac}.npy"
            histories[rep_name] = np.load(
                history_path, allow_pickle=True
            ).item()

        plot_training_dynamics_reps(
            histories,
            save_path=f"{list(experiment_dirs.values())[0]}/fraction_{frac}/training_dynamics_all_reps.png",
            suptitle=f"Training Dynamics (Fraction = {frac/100:.1f})"
        )

def model_performance_results_combined(experiment_dirs):
    fractions_float = [f / 100 for f in fractions]

    metrics = {}

    for rep_name, base_dir in experiment_dirs.items():
        best_val_recon = []
        best_epochs = []
        prior_std = []

        with open(f"{base_dir}/decoder_prior_std.json", "r") as f:
            prior_std_dict = json.load(f)

        for frac in fractions:
            frac_dir = f"{base_dir}/fraction_{frac}"
            ckpt = torch.load(f"{frac_dir}/best.ckpt", map_location="cpu")

            best_val_recon.append(ckpt["best_val_recon"])
            best_epochs.append(ckpt["epoch"])
            prior_std.append(prior_std_dict[str(frac)])

        metrics[rep_name] = {
            "recon": best_val_recon,
            "epoch": best_epochs,
            "std": prior_std
        }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)

    for rep_name, m in metrics.items():
        axes[0].plot(fractions_float, m["recon"], marker="o", label=rep_name)
        axes[1].plot(fractions_float, m["epoch"], marker="o", label=rep_name)
        axes[2].plot(fractions_float, m["std"], marker="o", label=rep_name)

    axes[0].set_title("Best Val Recon")
    axes[1].set_title("Best Epoch")
    axes[2].set_title("Decoder Output Std (Å)")

    for ax in axes:
        ax.set_xlabel("Training Data Fraction")
        ax.grid(True)
        ax.legend()

    fig.suptitle("Model Performance Across Replicas", fontsize=14, y=1.05)
    plt.tight_layout()
    plt.savefig(
        f"{list(experiment_dirs.values())[0]}/summary_performance_all_reps.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
