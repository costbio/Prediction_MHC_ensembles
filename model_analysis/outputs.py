import os
import numpy as np
import matplotlib.pyplot as plt
import torch


fractions = list(range(10, 80, 10))


def load_histories(experiment_dirs, frac):
    histories = {}
    for rep_name, base_dir in experiment_dirs.items():
        history_path = os.path.join(base_dir, f"fraction_{frac}", f"history_{frac}.npy")
        if not os.path.exists(history_path):
            raise FileNotFoundError(f"Missing history file: {history_path}")
        histories[rep_name] = np.load(history_path, allow_pickle=True).item()
    return histories


def _get_condition_root(experiment_dirs):
    first_rep_dir = list(experiment_dirs.values())[0]
    return os.path.dirname(first_rep_dir)


def _get_outputs_dir(experiment_dirs):
    outputs_dir = os.path.join(_get_condition_root(experiment_dirs), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    return outputs_dir


def plot_replica_wise_metric(
    histories,
    train_key,
    val_key,
    metric_title,
    y_label,
    save_path=None,
    suptitle=None,
):
    rep_keys = ["rep_0", "rep_1", "rep_2"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, rep_key in zip(axes, rep_keys):
        if rep_key not in histories:
            ax.set_title(f"{rep_key} (missing)")
            ax.axis("off")
            continue

        h = histories[rep_key]
        if train_key not in h or val_key not in h:
            ax.set_title(f"{rep_key} (missing metric)")
            ax.axis("off")
            continue

        epochs = np.arange(len(h[train_key]))
        ax.plot(epochs, h[train_key], label="Train", linewidth=2)
        ax.plot(epochs, h[val_key], "--", label="Val", linewidth=2)

        ax.set_title(rep_key)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(y_label)

    fig.suptitle(suptitle or metric_title, fontsize=15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def model_accuracy_results_replica_wise(experiment_dirs):
    outputs_dir = _get_outputs_dir(experiment_dirs)

    for frac in fractions:
        histories = load_histories(experiment_dirs, frac)

        plot_replica_wise_metric(
            histories=histories,
            train_key="train_recon",
            val_key="val_recon",
            metric_title="Reconstruction Loss",
            y_label="Reconstruction Loss",
            save_path=os.path.join(outputs_dir, f"replica_wise_recon_frac_{frac}.png"),
            suptitle=f"Reconstruction Loss (Fraction = {frac/100:.1f})",
        )

        plot_replica_wise_metric(
            histories=histories,
            train_key="train_kl",
            val_key="val_kl",
            metric_title="KL Divergence",
            y_label="KL Divergence",
            save_path=os.path.join(outputs_dir, f"replica_wise_kl_frac_{frac}.png"),
            suptitle=f"KL Divergence (Fraction = {frac/100:.1f})",
        )

def model_performance_results_combined(experiment_dirs):
    outputs_dir = _get_outputs_dir(experiment_dirs)
    fractions_float = [f / 100 for f in fractions]

    all_reps_recon = []
    all_reps_kl = []

    for rep_name, base_dir in experiment_dirs.items():
        rep_recon = []
        rep_kl = []

        for frac in fractions:
            ckpt_path = os.path.join(base_dir, f"fraction_{frac}", "best.ckpt")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

            ckpt = torch.load(ckpt_path, map_location="cpu")
            rep_recon.append(ckpt["best_val_recon"])
            rep_kl.append(ckpt.get("best_val_kl", np.nan))

        all_reps_recon.append(rep_recon)
        all_reps_kl.append(rep_kl)

    all_reps_recon = np.array(all_reps_recon, dtype=float)
    all_reps_kl = np.array(all_reps_kl, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    mean_recon = all_reps_recon.mean(axis=0)
    std_recon = all_reps_recon.std(axis=0)

    axes[0].plot(fractions_float, mean_recon, marker="o", linewidth=2, label="Mean")
    axes[0].fill_between(
        fractions_float,
        mean_recon - std_recon,
        mean_recon + std_recon,
        alpha=0.3,
        label="SD"
    )

    axes[0].set_title("Best Validation Reconstruction")
    axes[0].set_xlabel("Training Fraction")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    mean_kl = np.nanmean(all_reps_kl, axis=0)
    std_kl = np.nanstd(all_reps_kl, axis=0)

    axes[1].plot(fractions_float, mean_kl, marker="o", linewidth=2, label="Mean")
    axes[1].fill_between(
        fractions_float,
        mean_kl - std_kl,
        mean_kl + std_kl,
        alpha=0.3,
        label="SD"
    )

    axes[1].set_title("Best Validation KL")
    axes[1].set_xlabel("Training Fraction")
    axes[1].set_ylabel("KL")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(outputs_dir, "summary_performance_all_reps.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

