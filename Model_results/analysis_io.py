import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Directory helper
# ==========================
def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


# ==========================
# Figure saving
# ==========================
def save_figure(fig, path, dpi=300):
    _ensure_dir(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ==========================
# Array saving
# ==========================
def save_array_txt(array, path, header=None):
    _ensure_dir(path)
    np.savetxt(path, array, header=header if header else "")


def save_array_npy(array, path):
    _ensure_dir(path)
    np.save(path, array)


# ==========================
# Dict / JSON saving
# ==========================
def save_dict_json(dictionary, path):
    _ensure_dir(path)
    with open(path, "w") as f:
        json.dump(dictionary, f, indent=2)
