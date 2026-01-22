# Protein Conformational Ensemble Generation with XYZ-VAE

This repository implements a **Variational Autoencoder (VAE)** framework for **protein conformational ensemble generation** directly from molecular dynamics (MD) trajectories.  
The model is trained on Cartesian backbone coordinates and evaluated by comparing generated ensembles against reference MD simulations using structural and statistical metrics.


---
## Scientific Motivation

Protein function is governed by ensembles of conformations, not a single static structure.
MD simulations provide rich ensemble data, but are expensive and limited in sampling.

This project explores whether a coordinate-based VAE can:

- Learn the distribution of backbone conformations from MD

- Generate new structures consistent with the reference ensemble

- Preserve key structural and dynamical observables (RMSD, RMSF, Rg, DSSP, contacts, PCA)

This work is **ensemble modeling**, not trajectory continuation.

## Core Idea

- Input: MD trajectory frames (backbone XYZ coordinates)
- Model: Fully-connected XYZ-VAE
- Output: New protein conformations sampled from the learned latent space
- Goal: Reproduce **ensemble-level properties**

---

## Repository Structure

```
.
├── data/
│   └── preprocessing.py        # Load, align, split, scale MD trajectories
│
├── models/
│   └── xyz_vae.py               # XYZ-based VAE architecture
│
├── training/
│   ├── train_loop.py            # Training + early stopping
│   ├── losses.py                # VAE loss (MSE + KL)
│   └── annealing.py             # KL beta warm-up schedule
│
├── inference/
│   └── generate_trajectory.py   # Latent sampling & decoding
│
├── utils/
│   └── seed.py                  # Reproducibility utilities
│
├── Model_analysis/
│   ├── outputs.py               # Training curves & performance plots
│   └── save_xtc.py              # Save generated trajectories
│
├── Model_notebooks/
│   ├── *_run.ipynb              # Training notebooks per protein
│   ├── Accuracy_metrics_pipeline.py
│   ├── analysis_plots.py
│   └── analysis_io.py
│
├── Model_results/
│   └── results_notebook/        # Post-training structural analysis
│
├── run_pipeline.py              # End-to-end pipeline
└── README.md
```

---

## Data Processing

1. **Alignment**
   - MD trajectories are aligned to a reference PDB using MDTraj
   - Atom selection (e.g. backbone) is applied consistently

2. **Flattening**
   - Each frame → `(N_atoms × 3)` Cartesian coordinates
   - No temporal ordering is preserved

3. **Dataset Split**
   - Optional equilibration discard (first 150 frames)
   - First fraction of frames used for learning
   - Remaining frames used as test set
   - Optional shuffling *within* the learning set only

4. **Scaling**
   - StandardScaler fitted on training data
   - Applied to validation and test sets

---

## Model: XYZ-VAE

- Encoder: Fully connected MLP with residual blocks and LayerNorm
- - Linear → GELU → Residual MLP → LayerNorm
- Latent space: Multivariate Gaussian
- - Mean + log-variance
- Decoder: MLP mapping latent vectors to XYZ coordinates
- - Latent → MLP → Coordinates
- - KL annealing to avoid posterior collapse
- Sampling: Reparameterization trick with log-variance clamping


---

## Training Strategy

- Loss:
  - Reconstruction: Mean Squared Error
  - Regularization: KL divergence
- KL Annealing:
  - Linear warm-up to prevent posterior collapse
- Early stopping:
  - Based on validation reconstruction loss
- Experiments:
  - Multiple training data fractions (10%–80%)
  - Multiple replicas
  - Shuffled vs non-shuffled learning sets
  - All frames vs post-equilibration frames

---

## Generation

After training:
- Latent vectors are sampled from N(0, I)
- Decoded structures are inverse-scaled
- Output saved as `.xtc` trajectories
- Decoder output variance is tracked as a proxy for generative diversity

---

## Evaluation Metrics

Generated ensembles are compared to reference MD using:

- RMSD (internal)
- RMSF (per-residue)
- Radius of Gyration
- Ramachandran (phi/psi distributions)
- DSSP secondary structure fractions
- Contact maps
- PCA projection (MD-fitted)

All analyses are ensemble-based.

---

## Running the Pipeline

### From Python

```python
from run_pipeline import run_pipeline

run_pipeline(
    xtc_file="protein_rep_0.xtc",
    pdb_file="protein.pdb",
    protein_id="3UTQ",
    rep_num="0",
    atom_selection="backbone",
    out_dir="Model_results/example_run",
    latent_dim=16,
    epochs=300,
    batch_size=64,
    kl_beta=1e-3,
    use_all_frames=True,
    shuffle_learn=True,
    seed=42,
    device="cuda"
)
```

### From Command Line

```bash
python run_pipeline.py   --xtc protein.xtc   --pdb protein.pdb   --latent_dim 16   --epochs 300
```


---

## Intended Use

- Studying data efficiency in ensemble learning
- Comparing generative diversity across training fractions
- Baseline generative model for protein conformations
- Methodological exploration prior to physics-informed models

---


