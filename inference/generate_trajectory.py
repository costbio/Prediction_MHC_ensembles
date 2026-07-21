import numpy as np
import torch

from inference.physical_filter import (
    filter_coords_batch,
    get_backbone_bond_pairs,
)

# without filtered inference
@torch.no_grad()
def generate_trajectory(model, n_samples, device="cuda", seed=42, temperature=1.0):

    model.eval()
    latent_dim = model.latent_dim
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    z = torch.randn(
        n_samples,
        latent_dim,
        device=device,
        generator=generator
    ) * temperature
    x_gen = model.decode(z)

    return x_gen.cpu()

# with filtered inference
@torch.no_grad()
def generate_filtered_trajectory(
    model,
    scaler,
    ref,
    target_n_frames,
    device="cuda",
    seed=42,
    temperature=1.0,
    batch_size=64,
    max_attempts=50000,
    rama_outlier_threshold=0.10,
    bond_bad_threshold=0.15,
    ideal_rules=None,
    verbose=False,
):
    model.eval()
    latent_dim = model.latent_dim
    n_atoms = ref.n_atoms

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    # Precompute bond pairs 
    bond_pairs = get_backbone_bond_pairs(ref.topology)

    accepted_batches = []
    all_score_dfs = []

    total_attempted = 0
    total_accepted = 0
    rama_failed_total = 0
    bond_failed_total = 0
    both_failed_total = 0
    loop_count = 0

    while total_accepted < target_n_frames:
        remaining_budget = max_attempts - total_attempted
        if remaining_budget <= 0:
            break

        current_batch_size = min(batch_size, remaining_budget)
        loop_count += 1

        # 1) Sample latent vectors
        z = torch.randn(
            current_batch_size,
            latent_dim,
            device=device,
            generator=generator
        ) * temperature
        # Temperature was used to expand the latent sampling distribution.

        # 2) Decode in scaled coordinate space
        x_gen_scaled = model.decode(z)
        x_gen_scaled_np = x_gen_scaled.cpu().numpy()

        # 3) Inverse scale to real coordinate space
        x_gen_np = scaler.inverse_transform(x_gen_scaled_np)

        # 4) Reshape to xyz
        coords_batch = x_gen_np.reshape(current_batch_size, n_atoms, 3).astype(np.float32)

        # 5) Apply physical filter
        accept_mask, score_df, bond_pairs = filter_coords_batch(
            coords_batch=coords_batch,
            ref=ref,
            rama_outlier_threshold=rama_outlier_threshold,
            bond_bad_threshold=bond_bad_threshold,
            ideal_rules=ideal_rules,
            bond_pairs=bond_pairs,
            return_score_df=True,
        )

        accepted_now = int(accept_mask.sum())
        total_attempted += current_batch_size
        total_accepted += accepted_now

        # Count failure types using thresholds
        rama_fail = score_df["outlier_fraction"].to_numpy() > rama_outlier_threshold
        bond_fail = score_df["bond_bad_fraction"].to_numpy() > bond_bad_threshold
        both_fail = rama_fail & bond_fail

        rama_failed_total += int(rama_fail.sum())
        bond_failed_total += int(bond_fail.sum())
        both_failed_total += int(both_fail.sum())

        if accepted_now > 0:
            accepted_batches.append(coords_batch[accept_mask])

        score_df = score_df.copy()
        score_df["batch_id"] = loop_count
        all_score_dfs.append(score_df)

        if verbose:
            running_rate = total_accepted / total_attempted if total_attempted > 0 else 0.0
            print(
                f"[Generation batch {loop_count:03d}] "
                f"attempted_now={current_batch_size} | "
                f"accepted_now={accepted_now} | "
                f"total_attempted={total_attempted} | "
                f"total_accepted={total_accepted}/{target_n_frames} | "
                f"running_acceptance={running_rate:.4f}"
            )

    if len(accepted_batches) == 0:
        raise RuntimeError(
            "No valid frames were accepted. "
            "Model generation may be too unrealistic or thresholds may be too strict."
        )

    accepted_coords = np.concatenate(accepted_batches, axis=0)
    if accepted_coords.shape[0] < target_n_frames:
        raise RuntimeError(
            f"Stopped after max_attempts={max_attempts}, but only "
            f"{accepted_coords.shape[0]} accepted frames were collected "
            f"(target={target_n_frames})."
        )

    accepted_coords = accepted_coords[:target_n_frames]
    acceptance_rate = total_accepted / total_attempted if total_attempted > 0 else 0.0

    stats = {
        "target_n_frames": int(target_n_frames),
        "returned_n_frames": int(accepted_coords.shape[0]),
        "total_attempted_frames": int(total_attempted),
        "total_accepted_frames_before_trim": int(total_accepted),
        "acceptance_rate": float(acceptance_rate),
        "rama_failed_total": int(rama_failed_total),
        "bond_failed_total": int(bond_failed_total),
        "both_failed_total": int(both_failed_total),
        "temperature": float(temperature),
        "batch_size": int(batch_size),
        "max_attempts": int(max_attempts),
        "rama_outlier_threshold": float(rama_outlier_threshold),
        "bond_bad_threshold": float(bond_bad_threshold),
        "n_atoms": int(n_atoms),
        "n_generation_loops": int(loop_count),
        "seed": int(seed),
    }

    return accepted_coords, stats, all_score_dfs