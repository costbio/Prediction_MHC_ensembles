import numpy as np
import pandas as pd
import mdtraj as md


# ============================================================
# DEFAULT FILTER SETTINGS
# ============================================================

DEFAULT_RAMA_OUTLIER_THRESHOLD = 0.10
DEFAULT_BOND_BAD_THRESHOLD = 0.15

# Ideal backbone bond rules (in nm)
# N-CA ~ 1.46 A = 0.146 nm
# CA-C ~ 1.53 A = 0.153 nm
# C-N  ~ 1.33 A = 0.133 nm
IDEAL_BOND_RULES = {
    "N-CA": {"ideal_nm": 0.146, "tol_nm": 0.03},
    "CA-C": {"ideal_nm": 0.153, "tol_nm": 0.03},
    "C-N":  {"ideal_nm": 0.133, "tol_nm": 0.03},
}


# ============================================================
# RAMACHANDRAN HELPERS
# ============================================================

def in_box(phi, psi, phi_min, phi_max, psi_min, psi_max):
    return (phi_min <= phi <= phi_max) and (psi_min <= psi <= psi_max)


def classify_ramachandran_point(phi_deg, psi_deg):
    """
    Very simple general-purpose Ramachandran classification.
    Returns: favored / allowed / outlier
    """
    favored_boxes = [
        (-180, -90,  90, 180),   # beta
        (-120,  -20, -80,  20),  # right-handed alpha
        (  20,  100,   0, 120),  # left-handed alpha
    ]

    allowed_boxes = [
        (-180, -60,   60, 180),   # broader beta
        (-160,   0, -100,  60),   # broader alpha
        (   0, 120,  -40, 140),   # broader left-handed
        (-180, -100, -180, -120), # lower-left rare allowed
        (  40, 100, -180, -80),   # lower-right rare allowed
    ]

    for box in favored_boxes:
        if in_box(phi_deg, psi_deg, *box):
            return "favored"

    for box in allowed_boxes:
        if in_box(phi_deg, psi_deg, *box):
            return "allowed"

    return "outlier"


# ============================================================
# BACKBONE BOND HELPERS
# ============================================================

def get_backbone_bond_pairs(topology):
    """
    Collect backbone bond pairs for:
      - N-CA
      - CA-C
      - C-N(next residue)

    Returns
    -------
    bond_pairs : list[dict]
        Metadata for each backbone bond.
    """
    bond_pairs = []

    for chain in topology.chains:
        residues = list(chain.residues)

        for i, res in enumerate(residues):
            atom_dict = {atom.name: atom.index for atom in res.atoms}

            if "N" in atom_dict and "CA" in atom_dict:
                bond_pairs.append({
                    "bond_type": "N-CA",
                    "resSeq": res.resSeq,
                    "atom_i": atom_dict["N"],
                    "atom_j": atom_dict["CA"]
                })

            if "CA" in atom_dict and "C" in atom_dict:
                bond_pairs.append({
                    "bond_type": "CA-C",
                    "resSeq": res.resSeq,
                    "atom_i": atom_dict["CA"],
                    "atom_j": atom_dict["C"]
                })

            if i < len(residues) - 1:
                next_res = residues[i + 1]
                next_atom_dict = {atom.name: atom.index for atom in next_res.atoms}

                if "C" in atom_dict and "N" in next_atom_dict:
                    bond_pairs.append({
                        "bond_type": "C-N",
                        "resSeq": res.resSeq,
                        "next_resSeq": next_res.resSeq,
                        "atom_i": atom_dict["C"],
                        "atom_j": next_atom_dict["N"]
                    })

    return bond_pairs


# ============================================================
# TRAJECTORY-LEVEL SCORING
# ============================================================

def score_trajectory_ramachandran_from_traj(traj):

    phi_idx, phi = md.compute_phi(traj)   # radians
    psi_idx, psi = md.compute_psi(traj)   # radians

    n_common = min(phi.shape[1], psi.shape[1])
    phi = phi[:, :n_common]
    psi = psi[:, :n_common]

    rows = []

    for frame_i in range(traj.n_frames):
        favored = 0
        allowed = 0
        outlier = 0

        for j in range(n_common):
            phi_deg = np.degrees(phi[frame_i, j])
            psi_deg = np.degrees(psi[frame_i, j])

            label = classify_ramachandran_point(phi_deg, psi_deg)

            if label == "favored":
                favored += 1
            elif label == "allowed":
                allowed += 1
            else:
                outlier += 1

        total = favored + allowed + outlier

        rows.append({
            "frame": frame_i,
            "n_torsions": total,
            "favored_count": favored,
            "allowed_count": allowed,
            "outlier_count": outlier,
            "favored_fraction": favored / total if total > 0 else np.nan,
            "allowed_fraction": allowed / total if total > 0 else np.nan,
            "outlier_fraction": outlier / total if total > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def score_bonds_against_ideal(traj, ideal_rules=None, bond_pairs=None):

    if ideal_rules is None:
        ideal_rules = IDEAL_BOND_RULES

    if bond_pairs is None:
        bond_pairs = get_backbone_bond_pairs(traj.topology)

    if len(bond_pairs) == 0:
        raise ValueError("No backbone bond pairs found in topology.")

    pair_indices = np.array([[b["atom_i"], b["atom_j"]] for b in bond_pairs], dtype=int)
    distances = md.compute_distances(traj, pair_indices)  # nm

    rows = []

    for frame_i in range(traj.n_frames):
        bad = 0
        total = len(bond_pairs)

        for j, bond in enumerate(bond_pairs):
            bond_type = bond["bond_type"]
            ref = ideal_rules[bond_type]

            ideal_nm = ref["ideal_nm"]
            tol_nm = ref["tol_nm"]

            val = distances[frame_i, j]

            if abs(val - ideal_nm) > tol_nm:
                bad += 1

        rows.append({
            "frame": frame_i,
            "n_bonds": total,
            "bad_bonds": bad,
            "bond_bad_fraction": bad / total if total > 0 else np.nan,
        })

    return pd.DataFrame(rows), bond_pairs


def score_combined_physical_plausibility(
    traj,
    rama_outlier_threshold=DEFAULT_RAMA_OUTLIER_THRESHOLD,
    bond_bad_threshold=DEFAULT_BOND_BAD_THRESHOLD,
    ideal_rules=None,
    bond_pairs=None,
):

    rama_df = score_trajectory_ramachandran_from_traj(traj)
    bond_df, bond_pairs = score_bonds_against_ideal(
        traj,
        ideal_rules=ideal_rules,
        bond_pairs=bond_pairs
    )

    merged = pd.merge(rama_df, bond_df, on="frame", how="inner")

    merged["accept"] = (
        (merged["outlier_fraction"] <= rama_outlier_threshold) &
        (merged["bond_bad_fraction"] <= bond_bad_threshold)
    )

    return merged, bond_pairs


# ============================================================
# BATCH-LEVEL FILTERING FOR INFERENCE
# ============================================================

def build_traj_from_coords(coords_batch, ref):

    coords_batch = np.asarray(coords_batch, dtype=np.float32)

    if coords_batch.ndim != 3:
        raise ValueError(
            f"coords_batch must have shape (n_frames, n_atoms, 3), got {coords_batch.shape}"
        )

    if coords_batch.shape[1] != ref.n_atoms:
        raise ValueError(
            f"Atom mismatch: coords_batch has {coords_batch.shape[1]} atoms, "
            f"but ref has {ref.n_atoms}"
        )

    if coords_batch.shape[2] != 3:
        raise ValueError(
            f"coords_batch last dimension must be 3, got {coords_batch.shape[2]}"
        )

    return md.Trajectory(coords_batch, ref.topology)


def filter_coords_batch(
    coords_batch,
    ref,
    rama_outlier_threshold=DEFAULT_RAMA_OUTLIER_THRESHOLD,
    bond_bad_threshold=DEFAULT_BOND_BAD_THRESHOLD,
    ideal_rules=None,
    bond_pairs=None,
    return_score_df=True,
):

    traj = build_traj_from_coords(coords_batch, ref)

    score_df, bond_pairs = score_combined_physical_plausibility(
        traj=traj,
        rama_outlier_threshold=rama_outlier_threshold,
        bond_bad_threshold=bond_bad_threshold,
        ideal_rules=ideal_rules,
        bond_pairs=bond_pairs,
    )

    accept_mask = score_df["accept"].to_numpy(dtype=bool)

    if return_score_df:
        return accept_mask, score_df, bond_pairs

    return accept_mask, None, bond_pairs


def summarize_filter_results(score_df):
    if score_df is None or len(score_df) == 0:
        return {
            "batch_size": 0,
            "accepted": 0,
            "rejected": 0,
            "acceptance_rate": 0.0,
            "rama_failed": 0,
            "bond_failed": 0,
            "both_failed": 0,
            "mean_outlier_fraction": np.nan,
            "mean_bond_bad_fraction": np.nan,
        }

    rama_fail = score_df["outlier_fraction"] > DEFAULT_RAMA_OUTLIER_THRESHOLD
    bond_fail = score_df["bond_bad_fraction"] > DEFAULT_BOND_BAD_THRESHOLD
    both_fail = rama_fail & bond_fail
    accepted = score_df["accept"]

    return {
        "batch_size": int(len(score_df)),
        "accepted": int(accepted.sum()),
        "rejected": int((~accepted).sum()),
        "acceptance_rate": float(accepted.mean()),
        "rama_failed": int(rama_fail.sum()),
        "bond_failed": int(bond_fail.sum()),
        "both_failed": int(both_fail.sum()),
        "mean_outlier_fraction": float(score_df["outlier_fraction"].mean()),
        "mean_bond_bad_fraction": float(score_df["bond_bad_fraction"].mean()),
    }