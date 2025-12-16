# XYZ-VAE: Variational Autoencoder for Protein Conformational Analysis

## Overview
This repository contains an end-to-end deep learning pipeline for **learning and reconstructing protein backbone conformations** from molecular dynamics (MD) trajectories using a **Variational Autoencoder (VAE)**. The goal of this model is not to reproduce molecular dynamics trajectories, but to learn the conformational ensemble of a protein from precomputed MD simulations. Once trained, the VAE enables fast sampling of structurally plausible conformations, providing an efficient alternative for exploring conformational diversity without running additional long MD simulations.

The model operates directly on flattened **Cartesian XYZ coordinates** extracted from MD trajectories (`.xtc`) and is designed to:
- Learn a compact latent representation of protein conformational space
- Reconstruct physically plausible backbone trajectories
- Quantitatively evaluate generated structures using RMSD

This is **not** a toy autoencoder. The pipeline includes alignment, scaling, KL annealing, residual MLP blocks, trajectory reconstruction, and structural validation.

---

## Key Features
- MDTraj-based trajectory loading, atom selection, and alignment
- StandardScaler normalization (fit on training set only)
- Deep residual MLP VAE architecture
- KL annealing to prevent posterior collapse
- Combined MSE + MAE reconstruction loss
- Automatic RMSD analysis of real vs generated trajectories
- Outputs directly usable in VMD (`.xtc`)

---

## Pipeline Structure

```
XTC + PDB
   │
   ▼
Atom selection & alignment
   │
   ▼
Flatten XYZ coordinates (N_atoms × 3)
   │
   ▼
Train / Validation / Test split
   │
   ▼
Standard scaling
   │
   ▼
XYZ-VAE training (KL annealing)
   │
   ▼
Trajectory reconstruction
   │
   ▼
RMSD evaluation + XTC export
```

---

## Model Architecture

### Encoder
- Input: `(N_atoms × 3)` flattened vector
- Dense layers: `1024 → 512 → 256 → 128`
- Residual MLP blocks at each scale
- GELU activations + LayerNorm
- Outputs:
  - `z_mean`
  - `z_log_var`
  - sampled latent vector `z`

### Latent Space
- Dimensionality: configurable (`--latent_dim`)
- Gaussian prior
- Sampling via reparameterization trick

### Decoder
- Symmetric to encoder
- Reconstructs flattened XYZ coordinates

---

## Loss Function

The total VAE loss is:

```
L = Reconstruction + β · KL
```

Where:
- **Reconstruction loss** = 0.5 × MSE + 0.5 × MAE
- **KL divergence** regularizes the latent space
- **β (KL weight)** is gradually increased via annealing

KL annealing schedule:
- 0 until `warmup_start`
- Linear increase until `warmup_end`
- Fixed at `max_beta` afterwards

This is critical to avoid latent space collapse.

---

## Installation

### Requirements
- Python ≥ 3.9
- TensorFlow ≥ 2.12
- MDTraj
- NumPy
- scikit-learn
- matplotlib

Install dependencies:
```bash
pip install tensorflow mdtraj scikit-learn matplotlib
```

---

## Usage

### Command Line
```bash
python xyz_vae.py \
  --xtc trajectory.xtc \
  --pdb structure.pdb \
  --atom_sel "name N or name CA or name C or name O" \
  --out_dir results \
  --latent_dim 48 \
  --epochs 150 \
  --kl_beta 1e-4
```

### Outputs
The output directory will contain:
- `best.keras` – best VAE model checkpoint
- `training_loss.png` – train vs validation loss
- `generated_test.xtc` – reconstructed trajectory
- `rmsd_real_vals.txt` – RMSD of original trajectory
- `rmsd_gen_vals.txt` – RMSD of generated trajectory
- `rmsd_summary.txt` – statistical summary

---

## Evaluation: RMSD

RMSD is computed after:
- Atom selection
- Superposition to reference structure

Reported statistics:
- Mean
- Median
- Standard deviation
- Maximum RMSD

Generated trajectories can be visualized directly in **VMD** and compared against the original MD simulation.

---

## Design Choices (Read This)

- **XYZ coordinates instead of internal coordinates**: simpler pipeline, but rotational/translation invariance must be handled explicitly (alignment step is mandatory).
- **StandardScaler** instead of MinMax: better behaved gradients for deep MLPs.
- **Residual MLP blocks**: stabilizes deep fully-connected architectures.
- **No temporal modeling**: each frame is treated independently. This is intentional.

If you expect temporal coherence, this model will disappoint you.

---

## Limitations

Be honest about what this model does *not* do:
- No time dependency (not LSTM / Transformer)
- No bond/angle constraints enforced
- No energy-based regularization
- Reconstruction ≠ physically valid dynamics

This is a **representation learning** model, not a simulator.

---

## Possible Extensions

If you want to make this actually stronger:
- Add temporal context (LSTM-VAE / Transformer-VAE)
- Move to internal coordinates (bond, angle, dihedral)
- Include physics-based losses (distance restraints)
- Perform latent space interpolation and clustering
- Compare against PCA / tICA baselines

---

## Author Notes

This project is intended for **research and experimentation** in protein conformational learning. If you use it in a thesis or publication, document the assumptions clearly.

If you don't understand *why* each step exists, don't trust the results.

