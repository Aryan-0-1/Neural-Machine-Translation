"""
Inference: greedy decoding and configurable beam search.

Both re-encode the source once and reuse enc_output across decoding
steps (the old notebook's translate_sentence recomputed the full
forward pass -- encoder included -- at every generated token, which is
wasted compute on limited hardware).
"""

import torch

from config import CONFIG, resolve_device
from data_pipeline import load_sentencepiece
from model import build_model


@torch.no_grad()
def greedy_translate(model, sp_hi, sp_en, sentence, device, cfg):
    model.eval()
    max_len = cfg["max_len"]

    src_ids = sp_hi.encode(sentence, out_type=int)[: max_len - 1] + [cfg["eos_id"]]
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask = (src != cfg["pad_id"]).unsqueeze(1).unsqueeze(2)
    enc_output = model.encode(src, src_mask)

    tgt = torch.tensor([[cfg["sos_id"]]], dtype=torch.long, device=device)
    for _ in range(max_len - 1):
        tgt_len = tgt.size(1)
        nopeak = torch.tril(torch.ones(tgt_len, tgt_len, device=device, dtype=torch.bool))
        tgt_mask = nopeak.unsqueeze(0).unsqueeze(0)
        dec_output = model.decode(tgt, enc_output, src_mask, tgt_mask)
        logits = model.fc(dec_output[:, -1, :])
        next_token = logits.argmax(dim=-1).item()
        tgt = torch.cat([tgt, torch.tensor([[next_token]], device=device)], dim=1)
        if next_token == cfg["eos_id"]:
            break

    ids = tgt.squeeze(0).tolist()[1:]  # drop SOS
    if cfg["eos_id"] in ids:
        ids = ids[: ids.index(cfg["eos_id"])]
    return sp_en.decode(ids)


@torch.no_grad()
def beam_search_translate(model, sp_hi, sp_en, sentence, device, cfg, beam_size=None):
    """Standard length-normalized beam search (Wu et al. 2016 style
    length penalty). Falls back to behaving like greedy when
    beam_size=1."""
    model.eval()
    max_len = cfg["max_len"]
    beam_size = beam_size or cfg["beam_size"]
    alpha = cfg["length_penalty"]

    src_ids = sp_hi.encode(sentence, out_type=int)[: max_len - 1] + [cfg["eos_id"]]
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask = (src != cfg["pad_id"]).unsqueeze(1).unsqueeze(2)
    enc_output = model.encode(src, src_mask)

    # Each beam: (token_id_list, cumulative_log_prob, finished)
    beams = [([cfg["sos_id"]], 0.0, False)]

    for _ in range(max_len - 1):
        all_candidates = []
        active = [b for b in beams if not b[2]]
        finished = [b for b in beams if b[2]]

        if not active:
            break

        for tokens, score, _ in active:
            tgt = torch.tensor([tokens], dtype=torch.long, device=device)
            tgt_len = tgt.size(1)
            nopeak = torch.tril(torch.ones(tgt_len, tgt_len, device=device, dtype=torch.bool))
            tgt_mask = nopeak.unsqueeze(0).unsqueeze(0)
            dec_output = model.decode(tgt, enc_output, src_mask, tgt_mask)
            logits = model.fc(dec_output[:, -1, :])
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

            topk_log_probs, topk_ids = log_probs.topk(beam_size)
            for lp, tid in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                new_tokens = tokens + [tid]
                new_score = score + lp
                new_finished = (tid == cfg["eos_id"])
                all_candidates.append((new_tokens, new_score, new_finished))

        all_candidates.extend(finished)

        def length_norm_score(cand):
            tokens, score, _ = cand
            length = len(tokens)
            lp = ((5 + length) / 6) ** alpha
            return score / lp

        all_candidates.sort(key=length_norm_score, reverse=True)
        beams = all_candidates[:beam_size]

        if all(b[2] for b in beams):
            break

    best_tokens = max(beams, key=lambda b: b[1] / (((5 + len(b[0])) / 6) ** alpha))[0]
    ids = best_tokens[1:]
    if cfg["eos_id"] in ids:
        ids = ids[: ids.index(cfg["eos_id"])]
    return sp_en.decode(ids)


def load_model_for_inference(checkpoint_path, cfg=CONFIG, device=None):
    device = device or resolve_device()
    sp_hi, sp_en = load_sentencepiece(cfg)
    model = build_model(cfg, sp_hi.get_piece_size(), sp_en.get_piece_size(), device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, sp_hi, sp_en, device


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--sentence", required=True)
    parser.add_argument("--beam", type=int, default=4)
    args = parser.parse_args()

    model, sp_hi, sp_en, device = load_model_for_inference(args.checkpoint)
    greedy = greedy_translate(model, sp_hi, sp_en, args.sentence, device, CONFIG)
    beam = beam_search_translate(model, sp_hi, sp_en, args.sentence, device, CONFIG, beam_size=args.beam)
    print("Hindi   :", args.sentence)
    print("Greedy  :", greedy)
    print(f"Beam({args.beam}) :", beam)
