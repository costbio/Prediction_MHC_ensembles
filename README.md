
# Prediction_MHC_ensembles

## Overview


The model is trained using Molecular Dynamics (MD) simulation data and learns the main conformational behavior of MHC class I proteins. After training, it can generate new protein conformations from the learned latent space.

The goal of this project is to investigate whether a VAE trained on shorter MD trajectories can reproduce conformational ensembles similar to those obtained from longer MD simulations.

---

## Features

- VAE model for protein backbone coordinates
- Training on MD trajectories
- KL annealing during training
- Ablation study for different VAE architectures
- Generation of new protein conformations
- Physical filtering of generated structures
- RMSD, RMSF, PCA and RMSIP analyses
- Equal-frame control experiment

---

## Repository Structure

```text
Prediction_MHC_ensembles/
│
├── ablation/              # Ablation study scripts
├── data/                  # Data preprocessing
├── inference/             # Conformation generation and filtering
├── model/                 # VAE architecture
├── training/              # Training code
├── model_analysis/        # Output and trajectory utilities
├── prody_analysis/        # RMSIP analyses
├── model_notebooks/       # Training notebooks
├── Model_results_M1/      # Result notebooks
├── utils/                 # Helper functions
├── run_pipeline.py        # Main pipeline
└── README.md
```

---

## Workflow

1. Load MD trajectories
2. Align protein structures
3. Select backbone atoms
4. Split data into training, validation and test sets
5. Scale coordinates
6. Train the VAE
7. Generate new conformations
8. Apply physical filtering
9. Evaluate the generated ensembles

---

## Model

The model is a Variational Autoencoder (VAE).

### Encoder

The encoder compresses protein coordinates into a low-dimensional latent space.

It uses:

- Fully connected layers
- GELU activation
- Layer Normalization
- Dropout

### Decoder

The decoder reconstructs protein coordinates from the latent representation.

### KL Annealing

KL divergence is gradually increased during training. This helps the model learn stable latent representations.

---

## Ablation Study

Several VAE architectures were compared before selecting the final model.

The tested models include:

- Basic VAE
- VAE with KL annealing
- VAE with residual blocks
- VAE with feature-wise attention

The best-performing model was selected for the main experiments.

---

## Training Strategy

Four different training conditions were evaluated.

- Shuffle + all frames
- Shuffle + remove first 150 frames
- No shuffle + all frames
- No shuffle + remove first 150 frames

The model was also trained using different fractions of the available training data:

- 10%
- 20%
- 30%
- 40%
- 50%
- 60%
- 70%

---

## Input Data

The project uses:

- PDB topology files
- XTC trajectory files

Three replicas are available for each protein system.

The analyzed systems are:

- 3UTQ
- 5C0F
- 5N1Y
- pep_free

Only backbone atoms are used as model input.

---

## Running the Pipeline

The main workflow is implemented in:

```text
run_pipeline.py
```

This script performs:

- preprocessing
- training
- trajectory generation
- physical filtering
- evaluation

Adjust the input paths and training parameters before running the script.

---

## Outputs

The pipeline produces:

- trained models
- reconstructed trajectories
- generated trajectories
- filtered trajectories
- training history
- generation statistics
- evaluation results

---

## Evaluation

Generated ensembles are compared with MD simulations using:

- Reconstruction Loss
- RMSD
- RMSF
- PCA
- RMSIP

These analyses evaluate both structural similarity and conformational dynamics.

---

## Main Findings

The main observations of this work are:

- The VAE reproduced the dominant conformational behavior of the studied MHC-I systems.
- About 40% of the training data was sufficient for stable ensemble generation.
- Using more data produced only small improvements.
- Rare conformational states were more difficult to reproduce.
- Physical filtering improved the quality of generated structures.

---
