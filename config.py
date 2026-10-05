"""
Central configuration for the Hindi -> English Transformer pipeline.

Everything downstream (tokenizer training, dataset building, model
construction, training loop, inference) reads from this single dict so
there is exactly one place to change hyperparameters.
"""

import os

CONFIG = {
    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    "train_parquet": "dataset/train-00000-of-00001.parquet",
    "valid_parquet": "dataset/validation-00000-of-00001.parquet",
    "test_parquet": "dataset/test-00000-of-00001.parquet",

    "processed_dir": "processed_data_v2",
    "checkpoint_dir": "checkpoints",
    "spm_dir": "spm_models",

    # ------------------------------------------------------------------
    # Dataset size limits (set to None to use the full split)
    # Cutting the training set is a real quality/speed trade-off: fewer
    # examples -> faster epochs and lower RAM, but worse generalization,
    # especially on rare words. Only shrink this if an epoch genuinely
    # does not finish in reasonable time on your hardware.
    # ------------------------------------------------------------------
    "max_train_samples": None,
    "max_val_samples": None,
    "max_test_samples": None,

    # ------------------------------------------------------------------
    # Tokenizer (SentencePiece BPE, trained separately per language
    # because Hindi and English have very different scripts/morphology)
    # ------------------------------------------------------------------
    "src_vocab_size": 8000,   # Hindi
    "tgt_vocab_size": 8000,   # English
    "spm_model_type": "bpe",
    "spm_character_coverage_src": 0.9995,  # Devanagari needs high coverage
    "spm_character_coverage_tgt": 1.0,

    # Special token ids are fixed by how we train the SentencePiece
    # models (see tokenizer_utils.py) so they are identical for src/tgt.
    "pad_id": 0,
    "unk_id": 1,
    "sos_id": 2,
    "eos_id": 3,

    # ------------------------------------------------------------------
    # Sequence length
    # Chosen from the *subword* token-length distribution, not the
    # word-level one from the old notebook. Re-check this against the
    # printout from data_pipeline.py::inspect_length_distribution before
    # trusting it on a new dataset.
    # ------------------------------------------------------------------
    "max_len": 96,

    # ------------------------------------------------------------------
    # Model architecture
    # d_model=256 with an 8k/8k subword vocab keeps embeddings+output
    # projection to roughly (256*8000)*2 + (256*8000) ~= 6.1M params
    # instead of the ~31M the word-level vocab cost -- this is the
    # actual lever for "quality per unit of compute" on limited hardware.
    # ------------------------------------------------------------------
    "d_model": 256,
    "num_heads": 8,
    "num_encoder_layers": 4,
    "num_decoder_layers": 4,
    "d_ff": 1024,
    "dropout": 0.1,
    "pre_norm": True,          # Pre-LN: more stable training, no LR warmup fragility
    "tie_target_embeddings": True,  # decoder input embedding == output projection weight

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    "batch_size": 64,
    "gradient_accumulation_steps": 2,   # effective batch size = 128
    "num_epochs": 20,
    "learning_rate": 3e-4,       # peak LR reached after warmup (Noam-style)
    "warmup_steps": 4000,
    "label_smoothing": 0.1,
    "grad_clip_norm": 1.0,
    "use_amp": True,             # mixed precision, auto-disabled on CPU
    "num_workers": 2,
    "pin_memory": True,
    "seed": 42,

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    "resume": False,
    "resume_checkpoint": "checkpoints/latest.pt",
    "keep_last_n_checkpoints": 3,

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    "beam_size": 4,
    "length_penalty": 0.6,
}


def resolve_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dirs(cfg):
    for key in ("processed_dir", "checkpoint_dir", "spm_dir"):
        os.makedirs(cfg[key], exist_ok=True)
