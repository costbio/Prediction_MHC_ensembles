import numpy as np
import mdtraj as md
from sklearn.decomposition import PCA



def data_preprocessing(gen_traj, ref_traj, backbone_pdb):
    gen_traj = md.load_xtc(gen_traj, top=backbone_pdb)
    ref_traj = md.load_xtc(ref_traj, top=backbone_pdb)
    return gen_traj, ref_traj

# --------------------------
# RMSD analysis
# --------------------------

def rmsd_internal(gen_traj, ref_traj):
    idx_align = ref_traj.topology.select("backbone")

    ref_traj.superpose(ref_traj[0], atom_indices=idx_align)
    gen_traj.superpose(ref_traj[0], atom_indices=idx_align)

    rmsd_ref = md.rmsd(ref_traj, ref_traj[0], atom_indices=idx_align)
    rmsd_gen = md.rmsd(gen_traj, ref_traj[0], atom_indices=idx_align)
    rmsd_gen_A = rmsd_gen * 10.0
    rmsd_ref_A = rmsd_ref * 10.0

    return rmsd_gen_A, rmsd_ref_A


# --------------------------
# RMSF 
# --------------------------
def rmsf_md_vs_gen(gen_traj, ref_traj):
    idx_ca = ref_traj.topology.select("name CA")
    idx_align = ref_traj.topology.select("backbone")
    ref_ref = ref_traj[0]
    gen_ref = gen_traj[0]

    ref_traj.superpose(ref_ref, atom_indices=idx_align)
    gen_traj.superpose(gen_ref, atom_indices=idx_align)

    rmsf_ref = md.rmsf(ref_traj, ref_ref, atom_indices=idx_ca) * 10.0
    rmsf_gen = md.rmsf(gen_traj, gen_ref, atom_indices=idx_ca) * 10.0


    res_ids = [ref_traj.topology.atom(i).residue.index for i in idx_ca]

    return res_ids, rmsf_ref, rmsf_gen


# --------------------------
# Radius of gyration
# --------------------------
def radius_of_gyration(gen_traj, ref_traj):
    rg_gen = md.compute_rg(gen_traj)
    rg_ref = md.compute_rg(ref_traj)
    return rg_gen, rg_ref

# --------------------------
# Dihedrals (phi/psi) and Ramachandran
# --------------------------
def compute_phi_psi(gen_traj, ref_traj):
    phi_ref, psi_ref = md.compute_phi(ref_traj)[1], md.compute_psi(ref_traj)[1]
    phi_gen, psi_gen = md.compute_phi(gen_traj)[1], md.compute_psi(gen_traj)[1]

    phi_ref = np.degrees(phi_ref)
    psi_ref = np.degrees(psi_ref)

    phi_gen = np.degrees(phi_gen)
    psi_gen = np.degrees(psi_gen)

    phi_ref_flat = phi_ref.flatten()
    psi_ref_flat = psi_ref.flatten()

    phi_gen_flat = phi_gen.flatten()
    psi_gen_flat = psi_gen.flatten()

    return phi_ref_flat, psi_ref_flat, phi_gen_flat, psi_gen_flat

# --------------------------
# DSSP
# --------------------------
def simplify_dssp(dssp):
    simple = np.copy(dssp)
    simple[np.isin(simple, ['H', 'G', 'I'])] = 'H'
    simple[np.isin(simple, ['E', 'B'])] = 'E'
    simple[np.isin(simple, ['T', 'S'])] = 'C'
    return simple

def dssp_fraction(dssp, ss_type):
    return np.mean(dssp == ss_type)

def residue_ss_probability(dssp, ss_type):
    return np.mean(dssp == ss_type, axis=0)

def dssp_analysis(ref_traj, gen_traj):
    dssp_ref = simplify_dssp(md.compute_dssp(ref_traj))
    dssp_gen = simplify_dssp(md.compute_dssp(gen_traj))

    fractions_md = {
    'H': dssp_fraction(dssp_ref, 'H'),
    'E': dssp_fraction(dssp_ref, 'E'),
    'C': dssp_fraction(dssp_ref, 'C')
    }

    fractions_gen = {
        'H': dssp_fraction(dssp_gen, 'H'),
        'E': dssp_fraction(dssp_gen, 'E'),
        'C': dssp_fraction(dssp_gen, 'C')
    }

    delta_helix = (
    residue_ss_probability(dssp_gen, 'H') -
    residue_ss_probability(dssp_ref, 'H'))

    return dssp_ref,dssp_gen, fractions_md, fractions_gen, delta_helix

# --------------------------
# Contact maps & similarity
# --------------------------

def compute_contact_map(traj, cutoff=0.8):
    idx_ca = traj.topology.select("name CA")
    n_res = len(idx_ca)

    pairs = np.array(
        [(i, j) for i in range(n_res) for j in range(i+1, n_res)],
        dtype=int
    )


    dist = md.compute_distances(traj, pairs, periodic=False)
    contacts = dist < cutoff

    contact_freq = np.zeros((n_res, n_res))

    pair_idx = 0
    for i in range(n_res):
        for j in range(i+1, n_res):
            contact_freq[i, j] = contacts[:, pair_idx].mean()
            contact_freq[j, i] = contact_freq[i, j]
            pair_idx += 1

    return contact_freq


# --------------------------
# PCA / t-SNE projection
# --------------------------
def pca(gen_traj, ref_traj):

    ca_idx_gen = gen_traj.topology.select("name CA")
    ca_idx_ref = ref_traj.topology.select("name CA")


    X_ref = ref_traj.xyz[:, ca_idx_ref, :].reshape(ref_traj.n_frames, -1)
    X_gen = gen_traj.xyz[:, ca_idx_gen, :].reshape(gen_traj.n_frames, -1)

    pca = PCA(n_components=2)
    pca.fit(X_ref)


    X_ref_pca = pca.transform(X_ref)
    X_gen_pca = pca.transform(X_gen)
    return X_ref_pca, X_gen_pca