"""
Optimized XYZ-VAE Pipeline
--------------------------
Uses:
- GELU activations
- Residual MLP blocks
- Dropout
- LayerNorm
- StandardScaler instead of MinMax
- Huber reconstruction loss
- KL annealing
"""

import os
import argparse
import numpy as np
import mdtraj as md
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# ============================================================
# 1) LOAD + ALIGN
# ============================================================
def load_and_align(dcd_file, pdb_file, atom_selection):
    traj = md.load_dcd(dcd_file, top=pdb_file)
    ref = md.load_pdb(pdb_file)

    sel = ref.topology.select(atom_selection)
    traj = traj.atom_slice(sel)
    ref = ref.atom_slice(sel)

    traj.superpose(ref)

    print(f"[INFO] Loaded {traj.n_frames} frames, {traj.n_atoms} atoms selected.")
    return traj, ref


# ============================================================
# 2) SPLIT
# ============================================================
def split_dataset(coords, train_ratio=0.8, val_ratio=0.1, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(coords.shape[0])
    rng.shuffle(idx)

    n = coords.shape[0]
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    tr = idx[:n_train]
    va = idx[n_train:n_train + n_val]
    te = idx[n_train + n_val:]

    print(f"[INFO] Split: train={len(tr)}, val={len(va)}, test={len(te)}")
    return coords[tr], coords[va], coords[te]


# ============================================================
# 3) SCALE (StandardScaler)
# ============================================================
def scale_data(x_train, x_val, x_test):
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)
    x_test_s = scaler.transform(x_test)
    return x_train_s, x_val_s, x_test_s, scaler


# ============================================================
# 4) Sampling Layer
# ============================================================
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        z_log_var = tf.clip_by_value(z_log_var, -10.0, 10.0)
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps


# ============================================================
# 5) KL Annealing
# ============================================================
class KLAnnealingCallback(tf.keras.callbacks.Callback):
    def __init__(self, kl_layer, max_beta=2e-5, warmup_start=20, warmup_end=140):
        super().__init__()
        self.kl_layer = kl_layer
        self.max_beta = max_beta
        self.warmup_start = warmup_start
        self.warmup_end = warmup_end

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_start:
            beta = 0.0
        elif epoch < self.warmup_end:
            # Linear increase: 0 → max_beta
            progress = (epoch - self.warmup_start) / (self.warmup_end - self.warmup_start)
            beta = progress * self.max_beta
        else:
            beta = self.max_beta

        self.kl_layer.kl_beta = float(beta)

# ============================================================
# 6) VAE Loss Layer (Huber + KL)
# ============================================================
class VAELossLayer(layers.Layer):
    def __init__(self, kl_beta=1e-3):
        super().__init__()
        self.kl_beta = kl_beta

    def call(self, inputs):
        x_true, x_pred, z_mean, z_log_var = inputs

        mse = tf.reduce_mean(tf.square(x_true - x_pred))
        mae = tf.reduce_mean(tf.abs(x_true - x_pred))
        recon = 0.5 * mse + 0.5 * mae

        kl = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )

        self.add_loss(recon + self.kl_beta * kl)
        return x_pred


# ============================================================
# 7) Residual MLP Block
# ============================================================
def mlp_block(x, units, dropout=0.1):
    h = layers.Dense(units, activation="gelu")(x)
    h = layers.Dropout(dropout)(h)
    h = layers.Dense(units, activation="gelu")(h)
    return layers.Add()([x, h])


# ============================================================
# 8) Build Optimized XYZ-VAE
# ============================================================
def build_xyz_vae(input_dim, latent_dim=64, kl_beta=2e-5):

    # ---------- Encoder ----------
    inp = Input(shape=(input_dim,), name="vae_input")
    x = layers.Dense(1024, activation="gelu")(inp)
    x = mlp_block(x, 1024)
    x = layers.LayerNormalization()(x)

    x = layers.Dense(512, activation="gelu")(x)
    x = mlp_block(x, 512)
    x = layers.LayerNormalization()(x)

    x = layers.Dense(256, activation="gelu")(x)
    x = mlp_block(x, 256)
    x = layers.LayerNormalization()(x)

    x = layers.Dense(128, activation="gelu")(x)
    x = mlp_block(x, 128)
    x = layers.LayerNormalization()(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = Model(inp, [z_mean, z_log_var, z], name="encoder")

    # ---------- Decoder ----------
    lat_in = Input(shape=(latent_dim,), name="z_sampling")
    y = layers.Dense(128, activation="gelu")(lat_in)
    y = mlp_block(y, 128)
    y = layers.LayerNormalization()(y)

    y = layers.Dense(256, activation="gelu")(y)
    y = mlp_block(y, 256)
    y = layers.LayerNormalization()(y)

    y = layers.Dense(512, activation="gelu")(y)
    y = mlp_block(y, 512)
    y = layers.LayerNormalization()(y)

    y = layers.Dense(1024, activation="gelu")(y)
    y = mlp_block(y, 1024)
    y = layers.LayerNormalization()(y)

    out = layers.Dense(input_dim, activation="linear")(y)

    decoder = Model(lat_in, out, name="decoder")

    # ---------- Full VAE ----------
    z_mean, z_log_var, z = encoder(inp)
    x_decoded = decoder(z)

    loss_layer = VAELossLayer(kl_beta)
    output = loss_layer([inp, x_decoded, z_mean, z_log_var])

    vae = Model(inp, output, name="xyz_vae")
    vae.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

    return vae, encoder, decoder, loss_layer


# ============================================================
# 9) Save DCD
# ============================================================
def save_dcd(ref, coords, out_path):
    traj = md.Trajectory(coords, ref.topology)
    traj.superpose(ref)
    traj.save_dcd(out_path)
    return out_path


# ============================================================
# 10) RMSD
# ============================================================
def rmsd_summary(pdb_file, traj_path, atom_selection):
    traj = md.load_dcd(traj_path, top=pdb_file)
    ref = md.load_pdb(pdb_file)

    sel = ref.topology.select(atom_selection)
    traj = traj.atom_slice(sel)
    ref = ref.atom_slice(sel)

    traj.superpose(ref)
    vals = md.rmsd(traj, ref)

    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals)),
        "max": float(np.max(vals)),
        "vals": vals
    }


def rmsd_gen(pdb_file, traj_path, atom_selection ):
   
    ref = md.load_pdb(pdb_file)

    sel_idx = ref.topology.select(atom_selection)
    ref = ref.atom_slice(sel_idx)
    traj = md.load_dcd(traj_path, top=ref)
    traj.superpose(ref)
    vals = md.rmsd(traj, ref)

    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals)),
        "max": float(np.max(vals)),
        "vals": vals
    }


# ============================================================
# 11) RUN PIPELINE
# ============================================================
def run_pipeline(
    dcd_file, pdb_file, atom_selection, out_dir,
    latent_dim=64, kl_beta=2e-5, epochs=150,
    batch_size=64, train_ratio=0.8, val_ratio=0.1
):

    os.makedirs(out_dir, exist_ok=True)

    traj, ref = load_and_align(dcd_file, pdb_file, atom_selection)
    coords = traj.xyz.reshape(traj.n_frames, -1)

    X_train, X_val, X_test = split_dataset(coords, train_ratio, val_ratio)

    X_train_s, X_val_s, X_test_s, scaler = scale_data(X_train, X_val, X_test)

    vae, encoder, decoder, kl_layer = build_xyz_vae(
        X_train_s.shape[1], latent_dim, kl_beta
    )

    ckpt = os.path.join(out_dir, "best.keras")

    callbacks = [
        KLAnnealingCallback(kl_layer, max_beta=kl_beta),
        EarlyStopping(monitor="val_loss", patience=60, restore_best_weights=True),
        ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True)
    ]


    history = vae.fit(
        X_train_s, X_train_s,
        validation_data=(X_val_s, X_val_s),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2
    )

    # Loss plot
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "training_loss.png"))
    plt.close()

    # Reconstruct
    decoded = vae.predict(X_test_s)
    decoded_inv = scaler.inverse_transform(decoded)
    decoded_xyz = decoded_inv.reshape(-1, traj.n_atoms, 3)

    gen_dcd_path = os.path.join(out_dir, "generated_test.dcd")
    save_dcd(ref, decoded_xyz, gen_dcd_path)

    real_rmsd = rmsd_summary(pdb_file, dcd_file, atom_selection)
    gen_rmsd = rmsd_gen(pdb_file, gen_dcd_path, atom_selection)

    np.savetxt(os.path.join(out_dir, "rmsd_real_vals.txt"), real_rmsd["vals"])
    np.savetxt(os.path.join(out_dir, "rmsd_gen_vals.txt"), gen_rmsd["vals"])

    with open(os.path.join(out_dir, "rmsd_summary.txt"), "w") as f:
        f.write("REAL:\n")
        f.write(str(real_rmsd) + "\n\n")
        f.write("GENERATED:\n")
        f.write(str(gen_rmsd) + "\n")

    print("[DONE] Pipeline complete.")
    return real_rmsd, gen_rmsd


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dcd", required=True)
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--atom_sel", default="name N or name CA or name C or name O")
    parser.add_argument("--out_dir", default="xyz_vae_results")
    parser.add_argument("--latent_dim", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--kl_beta", type=float, default=1e-4)
    args = parser.parse_args()

    run_pipeline(
        args.dcd, args.pdb, args.atom_sel, args.out_dir,
        latent_dim=args.latent_dim, epochs=args.epochs, kl_beta=args.kl_beta
    )
