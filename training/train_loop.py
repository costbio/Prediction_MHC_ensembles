import torch
from torch.utils.data import DataLoader, TensorDataset
from training.losses import vae_loss
from training.annealing import kl_beta_schedule
def train_vae(
    model,
    X_train,
    X_val,
    epochs,
    batch_size,
    kl_beta,
    device,
    out_dir
):
    
    patience = 15
    min_delta = 1e-4

    epochs_no_improve = 0
    best_epoch = 0

    history = {
    "train_recon": [],
    "train_kl": [],
    "val_recon": [],
    "val_kl": [],
    "beta": []
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32)),
        batch_size=batch_size
    )

    best_val_recon = float("inf")

    for epoch in range(epochs):
        model.train()
        beta = kl_beta_schedule(epoch, kl_beta)

        train_recon = 0.0
        train_kl = 0.0

        for (x,) in train_loader:
            x = x.to(device)
            optimizer.zero_grad()

            x_hat, mu, logvar = model(x)
            loss, recon, kl = vae_loss(x, x_hat, mu, logvar, beta)

            loss.backward()
            optimizer.step()

            train_recon += recon.item()
            train_kl += kl.item()

        train_recon /= len(train_loader)
        train_kl /= len(train_loader)

        # Validation
        model.eval()
        val_recon = 0.0
        val_kl = 0.0

        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                x_hat, mu, logvar = model(x)

                _, recon, kl = vae_loss(x, x_hat, mu, logvar, beta)
                val_recon += recon.item()
                val_kl += kl.item()

        val_recon /= len(val_loader)
        val_kl /= len(val_loader)

        history["train_recon"].append(train_recon)
        history["train_kl"].append(train_kl)
        history["val_recon"].append(val_recon)
        history["val_kl"].append(val_kl)
        history["beta"].append(beta)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_recon={train_recon:.4f} "
            f"train_kl={train_kl:.4f} | "
            f"val_recon={val_recon:.4f} "
            f"val_kl={val_kl:.4f} "
            f"beta={beta:.2e}"
        )

        # --- MODEL SELECTION + EARLY STOP LOGIC ---
        if val_recon < best_val_recon - min_delta:
            best_val_recon = val_recon
            best_epoch = epoch
            epochs_no_improve = 0

            torch.save(model.state_dict(), f"{out_dir}/best.pt")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(
                f"[EARLY STOP] "
                f"No improvement in val_recon for {patience} epochs. "
                f"Best epoch: {best_epoch}, best val_recon: {best_val_recon:.4f}"
            )
            break

    # Restore best model
    model.load_state_dict(torch.load(f"{out_dir}/best.pt"))

    return history

