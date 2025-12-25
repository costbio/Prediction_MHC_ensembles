
---

# XYZ-VAE

### Learning Protein Conformational Ensembles from Limited Molecular Dynamics Data

## Overview

This repository implements a **Variational Autoencoder (VAE)**–based deep learning framework for learning **protein conformational ensembles** directly from **Cartesian XYZ coordinates** extracted from molecular dynamics (MD) trajectories.

Unlike time-series prediction approaches, this model focuses on **distributional learning**: capturing the conformational manifold sampled by a protein rather than predicting future frames sequentially.

A central objective of this work is to **quantify how much MD data is minimally required** to obtain a meaningful latent representation of protein conformations.

---

## Scientific Motivation

Molecular dynamics simulations are computationally expensive, especially for large proteins or long timescales.
In practice, researchers often face the question:

> *How much MD simulation is actually enough?*

This project investigates whether:

* A generative deep learning model can learn a protein’s conformational ensemble
* Using only an **early fraction of an MD trajectory**
* And still generalize to **chronologically unseen future conformations**

The problem is framed as a **minimum-data analysis** rather than pure performance optimization.

---

## Core Contributions

* End-to-end VAE pipeline operating directly on flattened XYZ coordinates
* Residual MLP-based encoder–decoder architecture with controlled capacity
* KL divergence annealing with configurable schedules
* Early stopping based on validation reconstruction loss
* **Fraction-based training experiments**:

  * Systematic variation of training data size
  * Strict separation between early (training) and future (test) frames
* Quantitative evaluation using RMSD between real and generated structures

---

## Model Architecture

### Encoder

```
Input (3N) → 512 → 256 → 128 → (μ, logσ²)
```

### Decoder

```
Latent → 128 → 256 → 512 → Output (3N)
```

### Architectural Details

* Residual MLP blocks at each hidden layer
* GELU activations
* Layer Normalization
* Gaussian latent distribution
* Reparameterization trick with noise injection
* Latent mean constrained via `tanh` to stabilize KL behavior

The decoder capacity is deliberately limited to avoid **posterior collapse** and excessive memorization.

---

## Loss Function

The VAE objective consists of:

* **Reconstruction loss** (mean squared error on scaled coordinates)
* **KL divergence**, normalized by latent dimension
* **β-annealing** to gradually introduce regularization

This setup balances faithful reconstruction with meaningful latent structure.

---

## Data Processing Pipeline

### 1. Loading and Alignment

* MD trajectory (`.xtc`) and reference structure (`.pdb`)
* Atom selection (e.g. Cα atoms)
* Structural alignment to remove rigid-body motion

### 2. Fraction-Based Splitting

Given a trajectory with `N` frames:

1. An **early fraction** (e.g. 10%, 20%, 40%, 60%) is selected
2. This subset is **shuffled** and split into:

   * Training set
   * Validation set
3. The **remaining future frames** are used **exclusively as test data**

> The test set is never seen during training or validation.

This enforces a **chronological generalization constraint**, making evaluation more meaningful than random splits.

---

## Training Strategy

* Adam optimizer
* Fixed batch size across experiments
* KL β-annealing schedule
* **Early stopping** based on validation reconstruction loss:

  * Prevents overfitting
  * Ensures fair comparison across different data fractions
* Best model checkpoint selected using validation reconstruction loss only

---

## Fraction-Based Experiments

The model is trained **from scratch** for each early fraction:

| Fraction of MD Data | Purpose                   |
| ------------------- | ------------------------- |
| 10%                 | Extreme low-data regime   |
| 20%                 | Minimal viable learning   |
| 40%                 | Expected saturation point |
| 60%                 | Near-full information     |

For each fraction, the following are recorded:

* Best validation reconstruction loss
* Epoch at which early stopping occurs
* RMSD statistics on future test frames

This allows direct analysis of:

* Data efficiency
* Convergence behavior
* Diminishing returns with increasing data

---

## Results Summary (Qualitative)

* Reconstruction quality improves rapidly up to ~40% of data
* Beyond this point, gains saturate
* Higher fractions converge in fewer epochs
* Latent space remains stable across fractions

These results suggest that **a substantial portion of the conformational ensemble can be learned from limited MD data**, supporting the feasibility of reduced simulation strategies.

---

## Repository Structure

```
.
├── data/
│   └── preprocessing.py      # loading, alignment, splitting, scaling
├── models/
│   └── xyz_vae.py            # VAE architecture
├── training/
│   ├── train_loop.py         # training + early stopping
│   ├── losses.py             # VAE loss definition
│   └── annealing.py          # KL beta schedule
├── analysis/
│   ├── rmsd.py               # RMSD evaluation
│   └── save_xtc.py           # XTC generation
├── run_pipeline.py           # main execution script
└── README.md
```

---

## Running the Pipeline

```bash
python run_pipeline.py \
  --xtc trajectory.xtc \
  --pdb structure.pdb \
  --atom_sel "name CA" \
  --latent_dim 8 \
  --epochs 150 \
  --batch_size 64 \
  --kl_beta 1e-3
```

---

