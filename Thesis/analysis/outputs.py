import matplotlib.pyplot as plt
import numpy as np


# Reconstruction Loss
def recon_loss(history):
    epochs = np.arange(len(history["train_recon"]))

    plt.figure()
    plt.plot(epochs, history["train_recon"], label="Train Reconstruction")
    plt.plot(epochs, history["val_recon"], label="Validation Reconstruction")

    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.title("Reconstruction Loss vs Epoch")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# KL Divergence 
def kl_divergence(history):
    epochs = np.arange(len(history["train_recon"]))
    plt.figure()
    plt.plot(epochs, history["train_kl"], label="Train KL")
    plt.plot(epochs, history["val_kl"], label="Validation KL")

    plt.xlabel("Epoch")
    plt.ylabel("KL Divergence")
    plt.title("KL Divergence vs Epoch")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Beta Schedule
def beta(history):

    epochs = np.arange(len(history["train_recon"]))
    plt.figure()
    plt.plot(epochs, history["beta"], label="Beta")

    plt.xlabel("Epoch")
    plt.ylabel("KL Weight (β)")
    plt.title("Beta Schedule")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

