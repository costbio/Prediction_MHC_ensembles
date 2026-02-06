import numpy as np
import mdtraj as md
from sklearn.decomposition import PCA

# ============================================================
# LOAD
# ============================================================
def load_traj(gen_xtc, ref_xtc, pdb):
    gen_traj = md.load_xtc(gen_xtc, top=pdb)
    ref_traj = md.load_xtc(ref_xtc, top=pdb)
    return gen_traj, ref_traj


# ============================================================
# RMSD (replica-wise)
# ============================================================
def rmsd_internal(gen_traj, ref_traj):
    idx = ref_traj.topology.select("backbone")

    ref_traj.superpose(ref_traj[0], atom_indices=idx)
    gen_traj.superpose(ref_traj[0], atom_indices=idx)

    rmsd_ref = md.rmsd(ref_traj, ref_traj[0], atom_indices=idx) * 10.0
    rmsd_gen = md.rmsd(gen_traj, ref_traj[0], atom_indices=idx) * 10.0

    return rmsd_ref, rmsd_gen


# ============================================================
# RMSF (replica-wise)
# ============================================================
def rmsf_md_vs_gen(gen_traj, ref_traj):
    idx_ca = ref_traj.topology.select("name CA")
    idx_align = ref_traj.topology.select("backbone")

    ref_traj.superpose(ref_traj[0], atom_indices=idx_align)
    gen_traj.superpose(gen_traj[0], atom_indices=idx_align)

    rmsf_ref = md.rmsf(ref_traj, ref_traj[0], atom_indices=idx_ca) * 10.0
    rmsf_gen = md.rmsf(gen_traj, gen_traj[0], atom_indices=idx_ca) * 10.0

    res_ids = [ref_traj.topology.atom(i).residue.index for i in idx_ca]
    return res_ids, rmsf_ref, rmsf_gen


# ============================================================
# Radius of gyration
# ============================================================
def radius_of_gyration(gen_traj, ref_traj):
    rg_ref = md.compute_rg(ref_traj) * 10.0
    rg_gen = md.compute_rg(gen_traj) * 10.0
    return rg_ref, rg_gen


# ============================================================
# Ramachandran
# ============================================================
def compute_phi_psi(gen_traj, ref_traj):
    phi_r, psi_r = md.compute_phi(ref_traj)[1], md.compute_psi(ref_traj)[1]
    phi_g, psi_g = md.compute_phi(gen_traj)[1], md.compute_psi(gen_traj)[1]

    return (
        np.degrees(phi_r).flatten(),
        np.degrees(psi_r).flatten(),
        np.degrees(phi_g).flatten(),
        np.degrees(psi_g).flatten(),
    )


# ============================================================
# Contact map
# ============================================================
def compute_contact_map(traj, cutoff=0.8):
    idx_ca = traj.topology.select("name CA")
    n = len(idx_ca)

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    dist = md.compute_distances(traj, pairs)
    contacts = dist < cutoff

    cm = np.zeros((n, n))
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            cm[i, j] = contacts[:, k].mean()
            cm[j, i] = cm[i, j]
            k += 1
    return cm


# ============================================================
# PCA (MD-fit, replica-wise)
# ============================================================
def pca_projection(gen_traj, ref_traj):
    ca = ref_traj.topology.select("name CA")

    X_ref = ref_traj.xyz[:, ca, :].reshape(ref_traj.n_frames, -1)
    X_gen = gen_traj.xyz[:, ca, :].reshape(gen_traj.n_frames, -1)

    pca = PCA(n_components=2)
    pca.fit(X_ref)

    return pca.transform(X_ref), pca.transform(X_gen)
