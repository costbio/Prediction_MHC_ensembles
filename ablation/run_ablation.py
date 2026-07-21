from run_pipeline import run_pipeline
ABLATIONS = {
    "M0_simple_vae_sym": {
        "model_kwargs": {
            "encoder_dims": (512, 256, 128),
            "decoder_dims": (128, 256, 512),
            "use_residual": False,
            "use_attention": False,
            "use_layernorm": True,
            "activation": "gelu",
        },
        "train_kwargs": {
            "lr": 5e-4,
            "use_kl_annealing": False,
        },
    },

    "M1_kl_annealing_sym": {
        "model_kwargs": {
            "encoder_dims": (512, 256, 128),
            "decoder_dims": (128, 256, 512),
            "use_residual": False,
            "use_attention": False,
            "use_layernorm": True,
            "activation": "gelu",
        },
        "train_kwargs": {
            "lr": 5e-4,
            "use_kl_annealing": True,
        },
    },

    "M2_residual_sym": {
        "model_kwargs": {
            "encoder_dims": (512, 256, 128),
            "decoder_dims": (128, 256, 512),
            "use_residual": True,
            "use_attention": False,
            "use_layernorm": True,
            "activation": "gelu",
        },
        "train_kwargs": {
            "lr": 5e-4,
            "use_kl_annealing": True,
        },
    },

    "M3_final_attention_sym": {
        "model_kwargs": {
            "encoder_dims": (512, 256, 128),
            "decoder_dims": (128, 256, 512),
            "use_residual": True,
            "use_attention": True,
            "use_layernorm": True,
            "activation": "gelu",
        },
        "train_kwargs": {
            "lr": 5e-4,
            "use_kl_annealing": True,
        },
    },
}