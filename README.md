# Protein Conformational Ensemble prediction with Deep Learning

This study applies a **Variational Autoencoder (VAE)** based framework to learn **the conformational assemblies of proteins** directly from **molecular dynamics (MD) simulations**.

The ultimate goal is to reproduce the **structural distribution (assembly)** observed in MD simulations.

---

## Core Idea 

* Input: MD trajectories aligned to the reference backbone (`.xtc`)
* Representation: Smoothed Cartesian backbone coordinates
* Model: VAE with **feature-based attention** at the latent bottleneck
* Output: **Novel protein conformations sampled from the learned latent distribution**

---

## Project Structure

```
.
├── data/
│   └── preprocessing.py
│
├── models/
│   └── xyz_vae.py
│
├── training/
│   ├── train_loop.py
│   ├── losses.py
│   └── annealing.py
│
├── inference/
│   └── generate_trajectory.py
│
├── utils/
│   └── seed.py
│
├── Model_analysis/
│   ├── outputs.py
│   └── save_xtc.py
│
├── Model_results/
│   ├── Accuracy_metrics_pipeline.py
│   ├── analysis_plots.py
│   └── results_notebook/
│       └── Analysis.ipynb
│
├── Model_notebooks/
│   ├── 3UTQ/
│   ├── 5C0F/
│   ├── 5N1Y/
│   └── pep_free/
│
└── run_pipeline.py
```

---

## Data Preprocessing (`data/preprocessing.py`)

### `load_and_align`

* Loads MD trajectory (`xtc`) and reference structure (`pdb`)
* Aligns trajectory to reference using selected atoms (default: backbone)
* Saves:

  * Aligned backbone trajectory
  * Backbone-only PDB reference

### `split_dataset`

**Critical design choice:**

* NO random sampling across the full trajectory
* Training data = **first fraction of frames**
* Test data = **remaining frames**
* Optional:

  * discard equilibration frames
  * shuffle only inside training set

This enforces **ensemble realism** and avoids artificial data leakage.

### `scale_data`

* StandardScaler fitted **only on training data**
* Same scaler applied to validation and test sets

---

## Model (`models/xyz_vae.py`)

### Architecture

* Fully-connected encoder/decoder
* Residual MLP blocks + LayerNorm
* **Feature-wise self-attention** before latent projection
* Gaussian latent space with reparameterization

### What the attention does (and doesn’t)

✔ Learns correlations between coordinate dimensions
✘ Does NOT encode spatial locality
✘ Does NOT impose physical constraints

---

## Training (`training/`)

### Loss

[
\mathcal{L} = \text{MSE}_{recon} + \beta \cdot \text{KL}
]

* Reconstruction: coordinate-wise MSE
* KL divergence annealed with linear warmup
* Posterior collapse detection via KL threshold

### Training logic

* Early stopping on validation reconstruction loss
* Best model checkpointed per fraction
* KL beta increases gradually (annealing)

---

## Pipeline (`run_pipeline.py`)

For **each protein replica** and **each training fraction (10–80%)**:

1. Load & align trajectory
2. Flatten coordinates
3. Split into train / val / test
4. Scale data
5. Train XYZ-VAE
6. Sample latent space
7. Decode new conformations
8. Save generated `.xtc`
9. Measure decoder output variance (latent prior sanity check)

Output structure:

```
Model_results/<condition>/<protein>/<protein_rep_X>/
└── fraction_30/
    ├── best.ckpt
    ├── history_30.npy
    └── generated_frac_30.xtc
```

---

## Inference (`inference/generate_trajectory.py`)

* Samples latent vectors from N(0, I)
* Decodes into Cartesian coordinates
* No conditioning, no autoregression, no time notion

---

## Analysis & Evaluation (`Model_results/`)

### Structural Metrics

* RMSD (internal, aligned)
* RMSF (residue-wise, CA atoms)
* Radius of gyration
* Ramachandran (φ/ψ)
* Contact maps (CA–CA)
* PCA (fit on MD, project generated)

### What these analyses test

✔ Distributional similarity
✔ Structural plausibility
✔ Ensemble overlap


---

## Experiments (`Model_notebooks/`)

Each notebook runs:

* 4 experimental conditions:

  * use_all_frames ∈ {True, False}
  * shuffle_learn ∈ {True, False}
* 3 replicas per protein
* Proteins:

  * 3UTQ
  * 5C0F
  * 5N1Y
  * pep_free

Training dynamics and performance are aggregated across replicas.

---

## What This Model *Is*

* A **distribution learner** in high-dimensional coordinate space
* A testbed for **ensemble fidelity vs data fraction**
* A controlled setup to study:

  * data efficiency
  * replica consistency
  * latent space behavior

---

## Reproducibility

* Global seed control
* Deterministic splits
* Identical preprocessing across replicas

---

## Dependencies

* Python ≥ 3.9
* PyTorch
* MDTraj
* NumPy / SciPy
* scikit-learn
* Matplotlib / Seaborn

---


