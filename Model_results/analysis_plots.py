import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# RMSD
# ==========================
def plot_rmsd(rmsd_ref, rmsd_gen):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(rmsd_ref, label="MD", alpha=0.8)
    ax.plot(rmsd_gen, label="Generated", alpha=0.8)
    ax.set_xlabel("Frame")
    ax.set_ylabel("RMSD (Å)")
    ax.set_title("Internal RMSD")
    ax.legend()
    return fig


# ==========================
# RMSF
# ==========================
def plot_rmsf(res_ids, rmsf_ref, rmsf_gen):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(res_ids, rmsf_ref, label="MD")
    ax.plot(res_ids, rmsf_gen, label="Generated")
    ax.set_xlabel("Residue index")
    ax.set_ylabel("RMSF (Å)")
    ax.set_title("Residue-wise RMSF")
    ax.legend()
    return fig


# ==========================
# Radius of Gyration
# ==========================
def plot_rg(rg_ref, rg_gen):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(rg_ref * 10.0, label="MD")
    ax.plot(rg_gen * 10.0, label="Generated")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Rg (Å)")
    ax.set_title("Radius of Gyration")
    ax.legend()
    return fig


# ==========================
# Ramachandran
# ==========================
def plot_ramachandran(phi_ref, psi_ref, phi_gen, psi_gen):
    fig, axes = plt.subplots(1, 2, figsize=(10,4), sharex=True, sharey=True)

    axes[0].hist2d(phi_ref, psi_ref, bins=100, cmap="Blues")
    axes[0].set_title("MD")
    axes[0].set_xlabel("Phi (deg)")
    axes[0].set_ylabel("Psi (deg)")

    axes[1].hist2d(phi_gen, psi_gen, bins=100, cmap="Reds")
    axes[1].set_title("Generated")
    axes[1].set_xlabel("Phi (deg)")

    fig.suptitle("Ramachandran Plot")
    return fig


# ==========================
# DSSP
# ==========================
def plot_dssp_fractions(frac_md, frac_gen):
    labels = ["H", "E", "C"]
    md_vals = [frac_md[l] for l in labels]
    gen_vals = [frac_gen[l] for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(x - width/2, md_vals, width, label="MD")
    ax.bar(x + width/2, gen_vals, width, label="Generated")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction")
    ax.set_title("Secondary Structure Composition")
    ax.legend()
    return fig


def plot_dssp_delta_helix(delta_helix):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(delta_helix)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Residue index")
    ax.set_ylabel("Δ Helix Probability")
    ax.set_title("Per-residue Helix Probability Difference (Gen − MD)")
    return fig


# ==========================
# Contact Maps
# ==========================
def plot_contact_map(contact_map, title="Contact Map"):
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(contact_map, cmap="viridis", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("Residue index")
    ax.set_ylabel("Residue index")
    fig.colorbar(im, ax=ax, label="Contact probability")
    return fig


def plot_contact_map_difference(cm_gen, cm_ref):
    diff = cm_gen - cm_ref
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(diff, cmap="bwr", origin="lower", vmin=-1, vmax=1)
    ax.set_title("Δ Contact Map (Gen − MD)")
    fig.colorbar(im, ax=ax, label="Δ Contact probability")
    return fig


# ==========================
# PCA
# ==========================
def plot_pca(X_ref_pca, X_gen_pca):
    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(X_ref_pca[:,0], X_ref_pca[:,1],
               s=10, alpha=0.5, label="MD")
    ax.scatter(X_gen_pca[:,0], X_gen_pca[:,1],
               s=10, alpha=0.5, label="Generated")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA Projection (MD fit)")
    ax.legend()
    return fig
