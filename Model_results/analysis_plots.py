import matplotlib.pyplot as plt
import os

# ============================================================
def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 3-panel RMSD
# ============================================================
def plot_rmsd_panels(rmsd_refs, rmsd_gens):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for i, ax in enumerate(axes):
        ax.plot(rmsd_refs[i], color="black", lw=2, label="MD")
        ax.plot(rmsd_gens[i], color="tab:blue", alpha=0.8, label="GEN")
        ax.set_title(f"rep_{i}")
        ax.set_xlabel("Frame")

    axes[0].set_ylabel("RMSD (Å)")
    axes[0].legend()
    fig.suptitle("RMSD: MD vs Generated (replica-wise)")
    return fig


# ============================================================
# 3-panel RMSF
# ============================================================
def plot_rmsf_panels(res_ids, rmsf_refs, rmsf_gens):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for i, ax in enumerate(axes):
        ax.plot(res_ids, rmsf_refs[i], color="black", lw=2, label="MD")
        ax.plot(res_ids, rmsf_gens[i], color="tab:blue", alpha=0.8, label="GEN")
        ax.set_title(f"rep_{i}")
        ax.set_xlabel("Residue")

    axes[0].set_ylabel("RMSF (Å)")
    axes[0].legend()
    fig.suptitle("RMSF: MD vs Generated (replica-wise)")
    return fig


# ============================================================
# 3-panel Rg
# ============================================================
def plot_rg_panels(rg_refs, rg_gens):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for i, ax in enumerate(axes):
        ax.plot(rg_refs[i], color="black", lw=2, label="MD")
        ax.plot(rg_gens[i], color="tab:blue", alpha=0.8, label="GEN")
        ax.set_title(f"rep_{i}")
        ax.set_xlabel("Frame")

    axes[0].set_ylabel("Rg (Å)")
    axes[0].legend()
    fig.suptitle("Radius of Gyration (replica-wise)")
    return fig


# ============================================================
# 3-panel PCA
# ============================================================
def plot_pca_panels(X_refs, X_gens):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)

    for i, ax in enumerate(axes):
        ax.scatter(X_refs[i][:,0], X_refs[i][:,1],
                   s=10, alpha=0.4, color="gray", label="MD")
        ax.scatter(X_gens[i][:,0], X_gens[i][:,1],
                   s=10, alpha=0.6, color="tab:blue", label="GEN")
        ax.set_title(f"rep_{i}")
        ax.set_xlabel("PC1")

    axes[0].set_ylabel("PC2")
    axes[0].legend()
    fig.suptitle("PCA Projection (MD-fit, replica-wise)")
    return fig


# ============================================================
# 3-panel Contact Map (MD | GEN)
# ============================================================
def plot_contact_map_panels(cm_refs, cm_gens):
    fig, axes = plt.subplots(3, 2, figsize=(8, 12))

    for i in range(3):
        im1 = axes[i,0].imshow(cm_refs[i], origin="lower", cmap="viridis")
        axes[i,0].set_title(f"MD rep_{i}")

        im2 = axes[i,1].imshow(cm_gens[i], origin="lower", cmap="viridis")
        axes[i,1].set_title(f"GEN rep_{i}")

    fig.colorbar(im2, ax=axes.ravel().tolist(), shrink=0.6)
    fig.suptitle("Contact Maps (replica-wise)")
    return fig


# ============================================================
# 3-panel Ramachandran
# ============================================================
def plot_rama_panels(phi_refs, psi_refs, phi_gens, psi_gens):
    fig, axes = plt.subplots(3, 2, figsize=(8, 12), sharex=True, sharey=True)

    for i in range(3):
        axes[i,0].hist2d(phi_refs[i], psi_refs[i], bins=100, cmap="Blues")
        axes[i,0].set_title(f"MD rep_{i}")

        axes[i,1].hist2d(phi_gens[i], psi_gens[i], bins=100, cmap="Reds")
        axes[i,1].set_title(f"GEN rep_{i}")

    fig.suptitle("Ramachandran Plots (replica-wise)")
    return fig
