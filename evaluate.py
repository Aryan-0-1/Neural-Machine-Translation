"""
Evaluation on the TEST split only (never touched during training or
hyperparameter tuning -- that discipline is enforced by simply never
importing test data anywhere else in this project).

Metrics:
  - BLEU-1..4 and cumulative BLEU (sacrebleu / nltk) -- the standard
    MT overlap metric; measures n-gram precision against the
    reference with a brevity penalty.
  - chrF -- character n-gram F-score; more informative than BLEU for
    morphologically rich source/target and for shorter sentences,
    which matters for Hindi.
  - METEOR -- accounts for synonymy/stemming, correlates better with
    human judgment than BLEU on individual sentences; only computed if
    `nltk` with wordnet data is available.
  - ROUGE-L -- longest common subsequence overlap; included because it
    is cheap and complements BLEU's precision-only view with a
    recall-oriented one. (ROUGE-N is NOT reported: it's a summarization
    metric and adds nothing over BLEU/chrF for MT.)
  - Perplexity from test loss.
  - Parameter count, model size on disk, mean per-sentence inference
    time -- the actual "cost" side of the quality/cost trade-off.

Metrics NOT computed, and why: exact-match accuracy (useless for
free-form generation), word error rate (designed for ASR, not MT).
"""

import os
import time
import json

import torch
import torch.nn as nn
import numpy as np

from config import CONFIG, resolve_device
from data_pipeline import load_split_ids, load_sentencepiece
from dataset import build_dataloader
from model import build_model, count_parameters
from inference import greedy_translate, beam_search_translate


def compute_test_loss(model, sp_hi, sp_en, cfg, device):
    test_src, test_tgt = load_split_ids("test", cfg)
    loader = build_dataloader(test_src, test_tgt, cfg, shuffle=False, bucket_batches=False)
    criterion = nn.CrossEntropyLoss(ignore_index=cfg["pad_id"])

    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            decoder_input = tgt[:, :-1]
            decoder_target = tgt[:, 1:]
            output = model(src, decoder_input)
            loss = criterion(
                output.contiguous().view(-1, output.size(-1)),
                decoder_target.contiguous().view(-1),
            )
            n_tokens = (decoder_target != cfg["pad_id"]).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)


def decode_ids_to_text(ids, sp, cfg):
    ids = list(ids)
    if cfg["sos_id"] in ids:
        ids = ids[ids.index(cfg["sos_id"]) + 1:]
    if cfg["eos_id"] in ids:
        ids = ids[: ids.index(cfg["eos_id"])]
    ids = [i for i in ids if i != cfg["pad_id"]]
    return sp.decode(ids)


def generate_predictions(model, sp_hi, sp_en, cfg, device, n_examples=None,
                          use_beam=True):
    test_src, test_tgt = load_split_ids("test", cfg)
    if n_examples:
        test_src, test_tgt = test_src[:n_examples], test_tgt[:n_examples]

    hindi_texts, references, predictions, inference_times = [], [], [], []

    for src_ids, tgt_ids in zip(test_src, test_tgt):
        hi_text = sp_hi.decode([i for i in src_ids if i != cfg["eos_id"]])
        ref_text = decode_ids_to_text(tgt_ids, sp_en, cfg)

        t0 = time.time()
        if use_beam:
            pred = beam_search_translate(model, sp_hi, sp_en, hi_text, device, cfg)
        else:
            pred = greedy_translate(model, sp_hi, sp_en, hi_text, device, cfg)
        inference_times.append(time.time() - t0)

        hindi_texts.append(hi_text)
        references.append(ref_text)
        predictions.append(pred)

    return hindi_texts, references, predictions, inference_times


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def compute_bleu_chrf(references, predictions):
    import sacrebleu

    refs = [references]  # sacrebleu wants list-of-lists (one list per reference set)
    bleu = sacrebleu.corpus_bleu(predictions, refs)
    chrf = sacrebleu.corpus_chrf(predictions, refs)

    # Per-order BLEU-1..4 precisions (sacrebleu exposes them on the BLEU object)
    bleu_ngram = {f"BLEU-{i+1}": bleu.precisions[i] for i in range(4)}

    return {
        "BLEU_cumulative_4": bleu.score,
        **bleu_ngram,
        "chrF": chrf.score,
    }


def compute_meteor(references, predictions):
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score
        try:
            nltk.data.find("wordnet.zip", paths=[nltk.data.path[0]])
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)

        scores = [
            meteor_score([ref.split()], pred.split())
            for ref, pred in zip(references, predictions)
        ]
        return {"METEOR": float(np.mean(scores))}
    except Exception as e:
        print(f"[warn] METEOR skipped ({e}). Install nltk + run nltk.download('wordnet').")
        return {}


def compute_rouge_l(references, predictions):
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [scorer.score(ref, pred)["rougeL"].fmeasure
                  for ref, pred in zip(references, predictions)]
        return {"ROUGE-L": float(np.mean(scores))}
    except Exception as e:
        print(f"[warn] ROUGE-L skipped ({e}). pip install rouge-score.")
        return {}


# ----------------------------------------------------------------------
# Heuristic error analysis (pattern counts, not a replacement for
# reading examples yourself -- these are cheap signals to prioritize
# what to look at)
# ----------------------------------------------------------------------

def heuristic_error_analysis(hindi_texts, references, predictions):
    issues = {
        "empty_prediction": 0,
        "repeated_token_run": 0,     # e.g. "the the the"
        "much_shorter_than_ref": 0,  # possible premature EOS / dropped content
        "much_longer_than_ref": 0,   # possible hallucination / looping
        "devanagari_leaked_into_output": 0,  # untranslated Hindi span
    }
    examples = {k: [] for k in issues}

    for hi, ref, pred in zip(hindi_texts, references, predictions):
        pred_tokens = pred.split()
        ref_tokens = ref.split()

        if len(pred_tokens) == 0:
            issues["empty_prediction"] += 1
            examples["empty_prediction"].append((hi, ref, pred))
            continue

        for i in range(len(pred_tokens) - 2):
            if pred_tokens[i] == pred_tokens[i + 1] == pred_tokens[i + 2]:
                issues["repeated_token_run"] += 1
                examples["repeated_token_run"].append((hi, ref, pred))
                break

        if ref_tokens and len(pred_tokens) < 0.5 * len(ref_tokens):
            issues["much_shorter_than_ref"] += 1
            examples["much_shorter_than_ref"].append((hi, ref, pred))
        if ref_tokens and len(pred_tokens) > 1.8 * len(ref_tokens) + 3:
            issues["much_longer_than_ref"] += 1
            examples["much_longer_than_ref"].append((hi, ref, pred))

        if any("\u0900" <= ch <= "\u097F" for ch in pred):
            issues["devanagari_leaked_into_output"] += 1
            examples["devanagari_leaked_into_output"].append((hi, ref, pred))

    # keep only a few examples per bucket
    examples = {k: v[:5] for k, v in examples.items()}
    return issues, examples


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------

def model_size_on_disk_mb(checkpoint_path):
    return os.path.getsize(checkpoint_path) / (1024 ** 2)


def full_report(checkpoint_path="checkpoints/best.pt", cfg=CONFIG,
                 n_test_examples=None, n_qualitative=30, use_beam=True):
    device = resolve_device()
    sp_hi, sp_en = load_sentencepiece(cfg)
    model = build_model(cfg, sp_hi.get_piece_size(), sp_en.get_piece_size(), device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    total_params, trainable_params = count_parameters(model)

    test_loss = compute_test_loss(model, sp_hi, sp_en, cfg, device)
    perplexity = float(np.exp(test_loss))

    hindi_texts, references, predictions, inf_times = generate_predictions(
        model, sp_hi, sp_en, cfg, device, n_examples=n_test_examples, use_beam=use_beam
    )

    metrics = {}
    metrics.update(compute_bleu_chrf(references, predictions))
    metrics.update(compute_meteor(references, predictions))
    metrics.update(compute_rouge_l(references, predictions))
    metrics["Test_Loss"] = test_loss
    metrics["Perplexity"] = perplexity
    metrics["Parameters"] = total_params
    metrics["Model_Size_MB"] = model_size_on_disk_mb(checkpoint_path)
    metrics["Mean_Inference_Time_s"] = float(np.mean(inf_times))

    issues, issue_examples = heuristic_error_analysis(hindi_texts, references, predictions)

    print("\n===== METRICS =====")
    for k, v in metrics.items():
        print(f"{k:>28}: {v}")

    print("\n===== ERROR PATTERN COUNTS (out of {} test sentences) =====".format(len(predictions)))
    for k, v in issues.items():
        print(f"{k:>32}: {v} ({100*v/max(len(predictions),1):.1f}%)")

    print("\n===== QUALITATIVE SAMPLES =====")
    idxs = np.linspace(0, len(predictions) - 1, num=min(n_qualitative, len(predictions)), dtype=int)
    for i in idxs:
        print("-" * 70)
        print("Hindi     :", hindi_texts[i])
        print("Reference :", references[i])
        print("Prediction:", predictions[i])

    return {
        "metrics": metrics,
        "error_issues": issues,
        "error_examples": issue_examples,
        "qualitative_indices": idxs.tolist(),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--n_test_examples", type=int, default=None,
                         help="Evaluate on a subset for speed; omit for full test set.")
    parser.add_argument("--greedy", action="store_true", help="Use greedy instead of beam search")
    args = parser.parse_args()

    result = full_report(
        checkpoint_path=args.checkpoint,
        n_test_examples=args.n_test_examples,
        use_beam=not args.greedy,
    )
    with open("evaluation_report.json", "w") as f:
        json.dump({k: v for k, v in result.items() if k != "error_examples"}, f, indent=2, default=str)
