# 🧬 VAE-based Protein Conformational Ensemble Generation

This repository contains the implementation and analysis pipeline for a **Variational Autoencoder (VAE)** designed to learn and generate **protein conformational ensembles** from Molecular Dynamics (MD) trajectories.

---

## 🎯 Project Objective

The goal of this work is **not to replace MD simulations**, but to:

> Learn the distribution of protein conformations from limited MD data and generate plausible ensembles.

Key idea:

* Use **early / partial MD trajectories**
* Train a **coordinate-based VAE**
* Generate new conformations
* Apply **post-generation physical filtering**
* Evaluate against MD using structural and dynamical metrics

---

## 🧪 Dataset

* **4 proteins**:

  * 3UTQ
  * 5C0F
  * 5N1Y
  * pep_free

* **3 replicas per protein** (MD trajectories)

* Input files:

  * `.xtc` → trajectory
  * `.pdb` → topology

* Only **backbone atoms** are used.

---

## ⚙️ Pipeline Overview

```
Raw MD Data (XTC + PDB)
        ↓
Load & Align (to first frame)
        ↓
Backbone Extraction
        ↓
Train / Val / Test Split
        ↓
StandardScaler (fit on train)
        ↓
Flatten Coordinates
        ↓
VAE Training
        ↓
Latent Sampling
        ↓
Generated Structures
        ↓
Physical Filtering
        ↓
Final Analysis
```

---

## 🧠 Model

### Variational Autoencoder (VAE)

* Input: flattened backbone coordinates
* Encoder:

  * MLP + Feature Attention + Residual blocks
* Latent space:

  * μ (mean), logσ² (log variance)
* Sampling:

  * Reparameterization trick
* Decoder:

  * Symmetric MLP

---

## 📉 Loss Function

[
\mathcal{L} = \text{Reconstruction} + \beta \cdot KL
]

* Reconstruction: **MSE loss**
* KL divergence: latent regularization
* β: controlled via **KL annealing**

---

## 🔥 KL Annealing

KL weight is gradually increased:

* Early epochs → reconstruction learning
* Later epochs → latent regularization

This prevents:

* posterior collapse
* unstable training

---

## 🧪 Experimental Design

### Conditions (4)

| Condition                 | Description                   |
| ------------------------- | ----------------------------- |
| shuffle_allframes         | shuffled training, all frames |
| shuffle_not_allframes     | shuffled, discard first 150   |
| not_shuffle_allframes     | no shuffle, all frames        |
| not_shuffle_not_allframes | no shuffle, discard first 150 |

---

### Train Fractions

```
10% → 70% (step 10%)
```

Validation and test sets are fixed across all fractions.

---

### Replicas

Each protein:

* 3 independent MD replicas
* Results aggregated as **mean ± std**

---

## 📊 Evaluation Metrics

### Training Metrics

* Validation Reconstruction Loss → accuracy
* KL Divergence → latent regularization

---

### Structural Metrics

* RMSD distribution → structural diversity
* RMSF → residue-level flexibility

---


---

### Dynamical Metric

* RMSIP → overlap of motion subspaces

---

## 🧹 Physical Filtering

Generated structures are filtered using:

* **Ramachandran constraints**
* **Backbone bond constraints**

Thresholds:

* Ramachandran outlier fraction < 0.10
* Bad bond fraction < 0.15

---

## 📈 Outputs

Each experiment produces:

* Generated trajectories (`.xtc`)
* Filtered trajectories
* Training curves
* Metric summaries (`metric_summary.json`)
* Plots (RMSD, RMSF, RMSIP)

---

## 📁 Project Structure

```
.
├── data/
│   └── palantir_data/
│       └── {PROTEIN}/
│           ├── backbone.pdb
│           └── *_rep_*.xtc
│
├── Model_results/
│   └── {condition}/{protein}/{replica}/fraction_*/
│       ├── generated_frac_*.xtc
│       └── generated_filtered_*.xtc
│
├── Analysis_results/
│   └── {condition}/{protein}/
│       └── metric_summary.json
│
├── training/
│   ├── train_loop.py
│   ├── losses.py
│   └── annealing.py
│
├── models/
│   └── xyz_vae.py
│
├── inference/
│   └── generate_trajectory.py
│
└── Model_analysis/
```

---

## 🚀 How to Run

### 1. Preprocessing

* Align trajectories
* Extract backbone atoms
* Prepare dataset

---

### 2. Training

```
train_vae(...)
```

Key parameters:

* epochs = 300
* batch_size = 64
* latent_dim = 8 
* KL annealing enabled

---

### 3. Generation

* Sample from latent space
* Decode to coordinates

---

### 4. Filtering

* Apply Ramachandran + bond filters
* Generate final ensemble

---

### 5. Analysis

Run analysis notebooks for:

* RMSD 
* RMSF
* RMSIP


---




