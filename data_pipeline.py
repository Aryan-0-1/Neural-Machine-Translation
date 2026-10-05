"""
Preprocessing pipeline: parquet -> cleaned pairs -> SentencePiece BPE
tokenizers -> encoded id sequences saved as plain lists (never as a
pickled custom Dataset object -- that was the source of the
AttributeError in the original Transformer.ipynb, because unpickling a
custom class silently binds the *current* class definition to the
*old* object state).

Run as a script:
    python data_pipeline.py
"""

import os
import re
import json
import pickle
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd

from config import CONFIG, ensure_dirs


# ======================================================================
# 1. LOAD + CLEAN
# ======================================================================

def load_split(path):
    df = pd.read_parquet(path)
    out = pd.DataFrame()
    out["hindi"] = df["translation"].str["hi"]
    out["english"] = df["translation"].str["en"]
    return out


def normalize_text(text):
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Devanagari block + common punctuation/marks used in Hindi text.
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def contamination_ratio(text, pattern):
    """Fraction of alphabetic characters that belong to `pattern`'s script."""
    letters = re.findall(r"[^\W\d_]", text, flags=re.UNICODE)
    if not letters:
        return 0.0
    matches = pattern.findall(text)
    return len(matches) / max(len(letters), 1)


def clean_dataframe(df, min_chars=1, max_chars=400,
                     min_lang_purity=0.5, verbose_name=""):
    """
    Applies, in order:
      - normalization
      - drop empty / null
      - drop duplicate pairs
      - drop pairs that are absurdly long/short (character-level guard,
        cheap and independent of tokenizer choice)
      - drop pairs where the Hindi side is mostly Latin script or the
        English side is mostly Devanagari (language contamination /
        misaligned rows) -- a real, non-trivial issue in large scraped
        parallel corpora like this one.
    Returns the cleaned dataframe and a dict of drop counts for
    transparency (no silent data loss).
    """
    n0 = len(df)
    stats = {"start": n0}

    df = df.dropna(subset=["hindi", "english"]).copy()
    stats["dropped_na"] = n0 - len(df)

    df["hindi"] = df["hindi"].apply(normalize_text)
    df["english"] = df["english"].apply(normalize_text)

    n1 = len(df)
    df = df[(df["hindi"].str.len() >= min_chars) &
            (df["english"].str.len() >= min_chars)]
    stats["dropped_empty"] = n1 - len(df)

    n2 = len(df)
    df = df.drop_duplicates(subset=["hindi", "english"])
    stats["dropped_duplicate"] = n2 - len(df)

    n3 = len(df)
    df = df[(df["hindi"].str.len() <= max_chars) &
            (df["english"].str.len() <= max_chars)]
    stats["dropped_too_long_chars"] = n3 - len(df)

    n4 = len(df)
    hi_purity = df["hindi"].apply(lambda t: contamination_ratio(t, _DEVANAGARI_RE))
    en_purity = df["english"].apply(lambda t: contamination_ratio(t, _LATIN_RE))
    df = df[(hi_purity >= min_lang_purity) & (en_purity >= min_lang_purity)]
    stats["dropped_language_contamination"] = n4 - len(df)

    df = df.reset_index(drop=True)
    stats["final"] = len(df)

    if verbose_name:
        print(f"[{verbose_name}] {json.dumps(stats, indent=2)}")

    return df, stats


def inspect_length_distribution(df, spm_hi=None, spm_en=None, verbose_name=""):
    """
    Reports sentence length in characters (always) and, if SentencePiece
    models are already trained, in subword tokens (what actually
    matters for choosing MAX_LEN). Call this AFTER training the
    tokenizers, then set config["max_len"] from the printed percentiles.
    """
    report = {}
    for col in ("hindi", "english"):
        char_lens = df[col].str.len()
        report[f"{col}_char_p50"] = float(np.percentile(char_lens, 50))
        report[f"{col}_char_p95"] = float(np.percentile(char_lens, 95))
        report[f"{col}_char_p99"] = float(np.percentile(char_lens, 99))

    if spm_hi is not None and spm_en is not None:
        hi_tok_lens = df["hindi"].apply(lambda t: len(spm_hi.encode(t)))
        en_tok_lens = df["english"].apply(lambda t: len(spm_en.encode(t)))
        report["hindi_subword_p50"] = float(np.percentile(hi_tok_lens, 50))
        report["hindi_subword_p95"] = float(np.percentile(hi_tok_lens, 95))
        report["hindi_subword_p99"] = float(np.percentile(hi_tok_lens, 99))
        report["english_subword_p50"] = float(np.percentile(en_tok_lens, 50))
        report["english_subword_p95"] = float(np.percentile(en_tok_lens, 95))
        report["english_subword_p99"] = float(np.percentile(en_tok_lens, 99))

    if verbose_name:
        print(f"[{verbose_name} length distribution] {json.dumps(report, indent=2)}")
    return report


# ======================================================================
# 2. SENTENCEPIECE TOKENIZERS
# ======================================================================

def train_sentencepiece(df_train, cfg):
    """
    Trains two independent BPE models (Hindi, English). Independent
    (not shared) vocabularies are the right call here: the scripts
    don't overlap, so a shared vocab would waste capacity on symbols
    that never co-occur.

    Special token ids are pinned to 0/1/2/3 = pad/unk/sos/eos so both
    tokenizers agree, which the rest of the pipeline relies on.
    """
    import sentencepiece as spm

    os.makedirs(cfg["spm_dir"], exist_ok=True)

    hi_txt = os.path.join(cfg["spm_dir"], "hi_corpus.txt")
    en_txt = os.path.join(cfg["spm_dir"], "en_corpus.txt")
    df_train["hindi"].to_csv(hi_txt, index=False, header=False)
    df_train["english"].to_csv(en_txt, index=False, header=False)

    common_args = dict(
        model_type=cfg["spm_model_type"],
        pad_id=cfg["pad_id"], unk_id=cfg["unk_id"],
        bos_id=cfg["sos_id"], eos_id=cfg["eos_id"],
        pad_piece="<pad>", unk_piece="<unk>",
        bos_piece="<sos>", eos_piece="<eos>",
    )

    spm.SentencePieceTrainer.train(
        input=hi_txt,
        model_prefix=os.path.join(cfg["spm_dir"], "hi"),
        vocab_size=cfg["src_vocab_size"],
        character_coverage=cfg["spm_character_coverage_src"],
        **common_args,
    )
    spm.SentencePieceTrainer.train(
        input=en_txt,
        model_prefix=os.path.join(cfg["spm_dir"], "en"),
        vocab_size=cfg["tgt_vocab_size"],
        character_coverage=cfg["spm_character_coverage_tgt"],
        **common_args,
    )

    sp_hi = spm.SentencePieceProcessor(model_file=os.path.join(cfg["spm_dir"], "hi.model"))
    sp_en = spm.SentencePieceProcessor(model_file=os.path.join(cfg["spm_dir"], "en.model"))
    return sp_hi, sp_en


def load_sentencepiece(cfg):
    import sentencepiece as spm
    sp_hi = spm.SentencePieceProcessor(model_file=os.path.join(cfg["spm_dir"], "hi.model"))
    sp_en = spm.SentencePieceProcessor(model_file=os.path.join(cfg["spm_dir"], "en.model"))
    return sp_hi, sp_en


# ======================================================================
# 3. ENCODE + SAVE (plain lists, not pickled Dataset objects)
# ======================================================================

def encode_split(df, sp_hi, sp_en, cfg):
    """
    Hindi (source) gets EOS only. English (target) gets SOS+EOS so the
    decoder can be taught with teacher forcing (input = [SOS ... ],
    label = [... EOS]). Truncation happens HERE, once, so every
    downstream consumer (train/val/test, and the model's positional
    encoding buffer) sees sequences <= max_len. This is the exact step
    that was missing in Transformer.ipynb and caused the positional-
    encoding shape mismatch.
    """
    max_len = cfg["max_len"]
    src_ids, tgt_ids = [], []
    for hi, en in zip(df["hindi"], df["english"]):
        h = sp_hi.encode(hi, out_type=int)[: max_len - 1] + [cfg["eos_id"]]
        e = [cfg["sos_id"]] + sp_en.encode(en, out_type=int)[: max_len - 2] + [cfg["eos_id"]]
        src_ids.append(h)
        tgt_ids.append(e)
    return src_ids, tgt_ids


def save_split(name, src_ids, tgt_ids, cfg):
    path = os.path.join(cfg["processed_dir"], f"{name}.pkl")
    with open(path, "wb") as f:
        # Plain python lists -- no custom classes, so this can be
        # unpickled by any script regardless of how a Dataset class is
        # defined at the time.
        pickle.dump({"src": src_ids, "tgt": tgt_ids}, f)
    print(f"Saved {name}: {len(src_ids)} pairs -> {path}")


def load_split_ids(name, cfg):
    path = os.path.join(cfg["processed_dir"], f"{name}.pkl")
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["src"], d["tgt"]


# ======================================================================
# MAIN
# ======================================================================

def main(cfg=CONFIG):
    ensure_dirs(cfg)
    np.random.seed(cfg["seed"])

    print("Loading splits...")
    df_train = load_split(cfg["train_parquet"])
    df_valid = load_split(cfg["valid_parquet"])
    df_test = load_split(cfg["test_parquet"])

    print("Cleaning...")
    df_train, _ = clean_dataframe(df_train, verbose_name="train")
    df_valid, _ = clean_dataframe(df_valid, verbose_name="valid")
    df_test, _ = clean_dataframe(df_test, verbose_name="test")

    if cfg["max_train_samples"]:
        df_train = df_train.sample(
            n=min(cfg["max_train_samples"], len(df_train)),
            random_state=cfg["seed"],
        ).reset_index(drop=True)
    if cfg["max_val_samples"]:
        df_valid = df_valid.iloc[: cfg["max_val_samples"]].reset_index(drop=True)
    if cfg["max_test_samples"]:
        df_test = df_test.iloc[: cfg["max_test_samples"]].reset_index(drop=True)

    print("Training SentencePiece BPE tokenizers on the TRAIN split only...")
    sp_hi, sp_en = train_sentencepiece(df_train, cfg)

    inspect_length_distribution(df_train, sp_hi, sp_en, verbose_name="train")

    print("Encoding splits...")
    for name, df in (("train", df_train), ("valid", df_valid), ("test", df_test)):
        src_ids, tgt_ids = encode_split(df, sp_hi, sp_en, cfg)
        save_split(name, src_ids, tgt_ids, cfg)

    with open(os.path.join(cfg["processed_dir"], "config_snapshot.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
