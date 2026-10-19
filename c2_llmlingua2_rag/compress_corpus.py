"""C2-t1: compress test+val corpus text fields with LLMLingua-2 small.

Frozen params (CONTRACT_C2 §1): rate=0.5, use_words=True, force_tokens unused,
drop_concurrent=False, max_batch_size=64. Deterministic classifier (no seed).
Only the text field is replaced; all metadata preserved; original text kept
as text_original. Quality gate: compressed_tokens/original_tokens in
[0.30, 0.60] per contract R1 — out of gate: stop and report, no retuning.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path("/share/latent2/vcrm_icae")
OUT_DIR = ROOT / "c2_llmlingua2_rag"
MODEL = str(ROOT / "models/llmlingua-2-bert-base-multilingual-cased-meetingbank")
# NOTE: dir name must contain "bert-base-multilingual-cased" — llmlingua's
# is_begin_of_new_word() branches on that substring for word segmentation.


def main():
    from llmlingua import PromptCompressor

    compressor = PromptCompressor(model_name=MODEL, use_llmlingua2=True,
                                  device_map="cuda")
    GATE = (0.30, 0.60)  # R1 (contract §9): was [0.45, 0.60]
    for split in ["test", "val"]:
        src = ROOT / f"data/prepared/templama_{split}.json"
        episodes = json.loads(src.read_text())
        # flatten (episode, event) preserving structure
        items = []
        for ep in episodes:
            for ev in ep["events"]:
                items.append((ep, ev))
        print(f"[{split}] episodes={len(episodes)} events={len(items)}",
              flush=True)

        texts = [ev["text"] for _, ev in items]
        n_tokens = {"orig": 0, "comp": 0}
        t0 = time.time()
        results = [None] * len(texts)
        # Contract §1.3 maps to llmlingua 0.2.2 API as follows:
        #   use_words=True  -> LLMLingua-2 is inherently word-level: the BERT
        #     tokenizer is word-aligned and drop decisions are made on merged
        #     word probs (__token_prob_to_word_prob, token_to_word="mean").
        #   drop_concurrent=False -> lib param is drop_consecutive=False.
        #   max_batch_size=64 -> constructor arg (default 50).
        compressor = PromptCompressor(model_name=MODEL, use_llmlingua2=True,
                                      device_map="cuda",
                                      llmlingua2_config={"max_batch_size": 64})
        B = 64
        for i in range(0, len(texts), B):
            batch = texts[i:i + B]
            out = compressor.compress_prompt(
                batch, rate=0.5, drop_consecutive=False,
                # CRITICAL: default use_context_level_filter=True drops entire
                # low-prob texts from compressed_prompt_list (list shorter than
                # input, items silently vanish). Per-item preservation requires
                # token-level filtering only.
                use_context_level_filter=False)
            # llmlingua2 returns joined string in compressed_prompt; the
            # per-input list is compressed_prompt_list.
            for j, c in enumerate(out["compressed_prompt_list"]):
                results[i + j] = c
            n_tokens["orig"] += out["origin_tokens"]
            n_tokens["comp"] += out["compressed_tokens"]
            if (i // B) % 20 == 0:
                el = time.time() - t0
                r = n_tokens["comp"] / max(n_tokens["orig"], 1)
                print(f"[{split}] {i + len(batch)}/{len(texts)} "
                      f"elapsed {el:.0f}s running_rate {r:.4f}", flush=True)
        el = time.time() - t0
        rate = n_tokens["comp"] / n_tokens["orig"]
        print(f"[{split}] DONE {len(texts)} events in {el:.0f}s "
              f"rate={rate:.4f} gate={GATE}", flush=True)

        out_doc = {
            "split": split,
            "compressor": MODEL,
            "llmlingua_version": __import__("llmlingua").__version__,
            "frozen_params": {"rate": 0.5, "use_words": True,
                              "force_tokens": None, "drop_concurrent": False,
                              "max_batch_size": 64},
            "origin_tokens": n_tokens["orig"],
            "compressed_tokens": n_tokens["comp"],
            "compression_rate": round(rate, 6),
            "gate": list(GATE), "gate_revision": "R1",
            "gate_passed": GATE[0] <= rate <= GATE[1],
            "compress_seconds": round(el, 1),
            "episodes": [],
        }
        # rebuild episode structure with only text replaced
        ep_map = {}
        for (ep, ev), comp in zip(items, results):
            if ep["id"] not in ep_map:
                ep_map[ep["id"]] = {k: v for k, v in ep.items() if k != "events"}
                ep_map[ep["id"]]["events"] = []
            new_ev = dict(ev)
            new_ev["text_original"] = ev["text"]
            new_ev["text"] = comp
            ep_map[ep["id"]]["events"].append(new_ev)
        out_doc["episodes"] = list(ep_map.values())

        dest = OUT_DIR / f"corpus_{split}_compressed.json"
        dest.write_text(json.dumps(out_doc, ensure_ascii=False))
        print(f"[{split}] written {dest}", flush=True)

        if not out_doc["gate_passed"]:
            print(f"[GATE FAIL] rate {rate:.4f} outside {GATE}; "
                  "STOP, no retuning per contract §1.6", flush=True)
            sys.exit(2)


if __name__ == "__main__":
    main()
