# Neural Machine Translation: Hindi → English

A Transformer encoder–decoder translating Hindi into English, written from scratch in PyTorch (no `nn.Transformer`). It covers the full pipeline: data cleaning, SentencePiece BPE tokenization, training with a Noam learning-rate schedule, beam-search inference and a multi-metric evaluation report.

## Highlights

- **From-scratch Transformer**: multi-head attention, position-wise feed-forward, sinusoidal positional encoding, encoder and decoder stacks.
- **Pre-LayerNorm** blocks (switchable to Post-LN) and **tied** decoder-embedding / output-projection weights.
- **SentencePiece BPE** tokenizers, trained separately for Hindi and English (8k vocabulary each).
- **Data cleaning**: NFC normalization, de-duplication, length filtering and script-contamination filtering (drops pairs where the Hindi side is mostly Latin or the English side is mostly Devanagari).
- **Training features**: mixed precision (AMP), gradient accumulation, gradient clipping, label smoothing, Noam warmup, AdamW, length-bucketed batching.
- **Checkpointing**: saved after every epoch (`latest.pt`), best-validation model (`best.pt`), old epoch checkpoints pruned, full resume support.
- **Inference**: greedy decoding and length-normalized beam search; the encoder runs once per sentence and its output is reused at every decoding step.
- **Evaluation**: BLEU (1–4 and cumulative), chrF, METEOR, ROUGE-L, perplexity, model size, inference time and a heuristic error analysis.

## Repository structure

| File | Purpose |
|---|---|
| `config.py` | Single `CONFIG` dict holding every path and hyperparameter |
| `data_pipeline.py` | Load parquet → clean → train SentencePiece → encode → save `.pkl` |
| `dataset.py` | `TranslationDataset`, collate function, `BucketBatchSampler`, `build_dataloader` |
| `model.py` | Transformer implementation and `build_model` |
| `train.py` | Training loop, Noam scheduler, checkpointing and resume |
| `inference.py` | Greedy and beam-search translation, CLI |
| `evaluate.py` | Test-set metrics and error analysis, CLI |
| `requirements.txt` | Python dependencies |

## Setup

```bash
git clone https://github.com/Aryan-0-1/Neural-Machine-Translation.git
cd Neural-Machine-Translation
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"   # needed for METEOR
```

A CUDA GPU is strongly recommended for training. The code runs on CPU, but very slowly.

### Dataset

The dataset is not included in the repo. Place three parquet files in a `dataset/` folder:

```
dataset/
├── train-00000-of-00001.parquet
├── validation-00000-of-00001.parquet
└── test-00000-of-00001.parquet
```

Each file must have a `translation` column whose entries are dicts like `{"en": "...", "hi": "..."}`. Change the paths in `config.py` if your files live elsewhere.

## Usage

Run the steps in order:

```bash
# 1. Clean the data, train the tokenizers, encode and save the splits
python data_pipeline.py

# 2. Train (checkpoints go to checkpoints/)
python train.py

# 3. Evaluate the best checkpoint on the test split
python evaluate.py --checkpoint checkpoints/best.pt
#    options: --n_test_examples 200   (quick subset)   --greedy   (skip beam search)

# 4. Translate a sentence
python inference.py --checkpoint checkpoints/best.pt --sentence "मुझे हिंदी पसंद है" --beam 4
```

**Resuming training:** set `"resume": True` in `config.py` (and point `resume_checkpoint` at `checkpoints/latest.pt`), then run `python train.py` again. Training continues from the next epoch.

## Configuration

Key settings in `config.py`:

| Setting | Value |
|---|---|
| Tokenizer | SentencePiece BPE, 8,000 tokens per language |
| `max_len` | 96 subword tokens |
| `d_model` / `d_ff` | 256 / 1024 |
| Attention heads | 8 |
| Encoder / decoder layers | 4 / 4 |
| Dropout | 0.1 |
| Batch size | 64 (×2 gradient accumulation = effective 128) |
| Optimizer | AdamW, peak LR 3e-4, 4,000 warmup steps, weight decay 0.01 |
| Label smoothing | 0.1 |
| Epochs | 20 |
| Beam search | beam size 4, length penalty 0.6 |

Special token IDs are fixed for both tokenizers: `<pad>`=0, `<unk>`=1, `<sos>`=2, `<eos>`=3.

## Data

| Split | Raw pairs | After cleaning |
|---|---|---|
| Train | 534,319 | 354,070 |
| Validation | 2,000 | 1,852 |
| Test | 2,000 | 1,837 |

Most of the train-set reduction came from removing duplicate pairs (159,721), followed by language-contamination filtering (14,729) and over-long sentences (5,799).

## Results

Trained for 20 epochs on a CUDA GPU (about 5.4 minutes per epoch). Evaluated on the 1,837 cleaned test sentences with beam search (beam = 4).

| Metric | Score |
|---|---|
| BLEU (cumulative, 4-gram) | 18.55 |
| BLEU-1 / 2 / 3 / 4 precision | 54.79 / 26.35 / 14.99 / 9.06 |
| chrF | 40.24 |
| METEOR | 0.414 |
| ROUGE-L | 0.549 |
| Test loss / perplexity | 2.457 / 11.67 |
| Parameters | 11,477,824 |
| Checkpoint size | 131.7 MB (includes optimizer state) |
| Mean inference time | 0.347 s / sentence |

> The final validation loss (3.438) is higher than the test loss (2.457) because training and validation use label smoothing of 0.1, while `evaluate.py` computes test loss without it. The two numbers are not directly comparable.

**Error analysis (1,837 test sentences):** 0 empty outputs, 2 repeated-token runs (0.1%), 81 outputs much shorter than the reference (4.4%), 4 much longer (0.2%), 0 untranslated Devanagari leaks.

## Author

**Aryan Choudhary**

GitHub: [@Aryan-0-1](https://github.com/Aryan-0-1)

Repository: [RAGnarok](https://github.com/Aryan-0-1/Neural-Machine-Translation/tree/main)

---

## ⭐ If You Found This Project Useful

Consider giving the repository a ⭐ on GitHub.

Contributions, suggestions, and improvements are welcome.
