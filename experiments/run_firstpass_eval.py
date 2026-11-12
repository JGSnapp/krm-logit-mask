"""Experiment 10: score every catalog word from the next-token distributions, no search.

Order of words does not matter, so any catalog word could be written first. One
forward pass gives P(first token | resume) for all of them at once. We also read
the distribution at every word start of one free greedy keyword list (the model
"would like" to write a word at some point), and optionally refine the best
first tokens with one more batched forward (2-token prefixes).

Word value  u(l) = max over surfaces of [E(prefix) - alpha * max(prior(prefix), floor)] + beta * idf(l)
  E(prefix)  = best log-prob of the prefix over word-start positions,
  prior      = same prefix log-prob under the empty-text prompt (computed once per catalog).
Competency score:
  bag     - mean of the top-k word values of the competency,
  phrases - mean of top-2 word values inside a phrase, then mean of the top-2 phrases.

Usage: .venv/bin/python experiments/run_firstpass_eval.py [--eval-set ...] [--out ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_selection_eval import load_cases, summarize  # noqa: E402

from krm_mask.catalog import Catalog  # noqa: E402
from krm_mask.graph import PhraseGraph  # noqa: E402
from krm_mask.model import DEFAULT_MODEL, LocalCausalLM  # noqa: E402
from krm_mask.wordbag import (  # noqa: E402
    WordBagSelector, WordIndex, lm_prompt_tensor, pymorphy_lemmatizer, word_messages,
)


def word_start_logprobs(lm, prompt: list[int], max_new_tokens: int):
    """Log-softmax rows [positions, vocab] at the list start and after every newline."""
    import torch

    # Greedy decoding with KV cache; "," and ";" are replaced by a newline so that
    # every word starts a line (the same position the priors were computed for).
    tokenizer = lm.tokenizer
    newline = tokenizer.encode("\n", add_special_tokens=False)[0]
    ids = lm_prompt_tensor(lm, prompt)
    past = None
    rows, generated, at_start = [], [], True
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            out = lm.model(input_ids=ids, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
            if at_start:
                rows.append(logp)
            token = int(logp.argmax())
            if token == tokenizer.eos_token_id or token in (tokenizer.all_special_ids or []):
                break
            piece = tokenizer.decode([token])
            if "," in piece or ";" in piece:
                token, piece = newline, "\n"
            at_start = "\n" in piece
            generated.append(token)
            ids = torch.tensor([[token]], device=lm.device)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return torch.stack(rows), text, len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--eval-set", default="experiments/eval_set.json")
    parser.add_argument("--out", default="exp10_firstpass")
    parser.add_argument("--list-tokens", type=int, default=96)
    parser.add_argument("--expand", type=int, default=128, help="first tokens refined to 2 tokens")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()

    cases, spec = load_cases(role_filter=False, eval_set=args.eval_set)
    catalog = Catalog.from_json(ROOT / spec.get("catalog", "data/itmo_krm_3.json"))
    lm = LocalCausalLM.load(args.model)
    graph = PhraseGraph.from_catalog(catalog, None, {"indicators", "keywords"})
    index = WordIndex.build(graph, pymorphy_lemmatizer())

    def prompt(text: str) -> list[int]:
        return lm.prompt_ids(word_messages(text))[0].tolist()

    started = time.perf_counter()
    selector = WordBagSelector(lm, index, prompt)  # Prefix priors under the empty text.
    prior = selector.prior
    word_tokens = {surface: seq[:-1] for surface, seq in selector.tokens.items()}  # no delimiter
    print(f"priors for {len(word_tokens)} surfaces in {time.perf_counter() - started:.0f}s",
          flush=True)
    lemma_surfaces: dict[str, list[str]] = defaultdict(list)
    for surface, lemma in index.surfaces.items():
        if word_tokens[surface]:
            lemma_surfaces[lemma].append(surface)
    phrases_of: dict[str, list[set[str]]] = defaultdict(list)
    for competency_ids, lemmas in index.phrase_lemmas:
        for competency_id in competency_ids:
            phrases_of[competency_id].append(lemmas)

    configs = [
        # name, positions, depth, alpha, floor, beta, aggregate
        ("pos0_raw_bag", "first", 1, 0.0, None, 0.0, "bag"),
        ("pos0_pmi_bag", "first", 1, 1.0, None, 0.0, "bag"),
        ("list_raw_bag", "list", 1, 0.0, None, 0.0, "bag"),
        ("list_pmi_bag", "list", 1, 1.0, None, 0.0, "bag"),
        ("list_pmi_idf_bag", "list", 1, 1.0, None, 0.3, "bag"),
        ("list_pmi_phrases", "list", 1, 1.0, None, 0.0, "phrases"),
        ("list_pmi_idf_phrases", "list", 1, 1.0, None, 0.3, "phrases"),
        ("list2_pmi_bag", "list", 2, 1.0, None, 0.0, "bag"),
        ("list2_pmi_idf_bag", "list", 2, 1.0, None, 0.3, "bag"),
        ("list2_pmi_idf_phrases", "list", 2, 1.0, None, 0.3, "phrases"),
        ("list2_raw_idf_phrases", "list", 2, 0.0, None, 0.3, "phrases"),
    ]
    if args.only:
        configs = [c for c in configs if c[0] in args.only]

    def competency_scores(value: dict[str, float], aggregate: str, k: int = 4) -> dict[str, float]:
        scores = {}
        for competency_id, lemmas in index.competency_lemmas.items():
            if aggregate == "bag":
                top = sorted((value[l] for l in lemmas if l in value), reverse=True)[:k]
            else:
                phrase_values = []
                for phrase in phrases_of[competency_id]:
                    inside = sorted((value[l] for l in phrase if l in value), reverse=True)[:2]
                    if inside:
                        phrase_values.append(sum(inside) / len(inside))
                top = sorted(phrase_values, reverse=True)[:2]
            if top:
                scores[competency_id] = sum(top) / len(top)
        return scores

    report: dict[str, object] = {
        "experiment": args.out, "model": args.model, "eval_set": args.eval_set,
        "catalog": spec.get("catalog", "data/itmo_krm_3.json"),
        "list_tokens": args.list_tokens, "expand": args.expand, "configs": {},
    }
    rows_by_config: dict[str, list] = defaultdict(list)
    for case in cases:
        started = time.perf_counter()
        context = prompt(case["text"])
        rows, free_list, positions = word_start_logprobs(lm, context, args.list_tokens)
        evidence = {"first": rows[0], "list": rows.max(dim=0).values}
        # 2-token refinement at the list start: one batched forward over the best first tokens.
        first_tokens = {seq[0] for seq in word_tokens.values() if len(seq) > 1}
        ranked = sorted(first_tokens, key=lambda t: float(evidence["list"][t]), reverse=True)
        expanded = ranked[: args.expand]
        second_wanted = [
            sorted({seq[1] for seq in word_tokens.values() if len(seq) > 1 and seq[0] == t})
            for t in expanded
        ]
        second = lm.next_token_logprobs(context, [(t,) for t in expanded], second_wanted)
        pair_logprob = {
            (t, t2): v for t, tokens, values in zip(expanded, second_wanted, second)
            for t2, v in zip(tokens, values)
        }
        forward_seconds = time.perf_counter() - started
        for name, positions_kind, depth, alpha, floor, beta, aggregate in configs:
            first = evidence[positions_kind]

            def prefix_value(seq):
                if depth == 2 and len(seq) > 1 and (seq[0], seq[1]) in pair_logprob:
                    prefix = seq[:2]
                    logp = float(first[seq[0]]) + pair_logprob[(seq[0], seq[1])]
                else:
                    prefix = seq[:1]
                    logp = float(first[seq[0]])
                base = prior.get(prefix, 0.0)
                if floor is not None:
                    base = max(base, floor)
                return logp - alpha * base

            value = {
                lemma: max(prefix_value(word_tokens[s]) for s in surfaces)
                + beta * index.idf.get(lemma, 0.0)
                for lemma, surfaces in lemma_surfaces.items()
            }
            scores = competency_scores(value, aggregate)
            ranking = sorted(scores, key=scores.get, reverse=True)
            selected = ranking[: case["max_competencies"]]
            top_words = sorted(value, key=value.get, reverse=True)[:12]
            rows_by_config[name].append({
                "id": case["id"], "kind": case["kind"], "acceptable": case["acceptable"],
                "selected": selected,
                "best_acceptable_rank": min(
                    (ranking.index(c) + 1 for c in case["acceptable"] if c in ranking),
                    default=None),
                "top_words": top_words, "seconds": round(forward_seconds, 2),
            })
            print(name, case["id"], selected, top_words[:6], flush=True)
        print(case["id"], f"{forward_seconds:.1f}s positions={positions}",
              repr(free_list[:200]), flush=True)
    for name, *params in configs:
        rows = rows_by_config[name]
        summary = summarize(rows)
        web_ranks = [r["best_acceptable_rank"] for r in rows if r["kind"] == "web"]
        summary["best_acceptable_rank_web"] = web_ranks
        report["configs"][name] = {"params": params, "summary": summary, "rows": rows}
        print(name, json.dumps(summary), flush=True)
    out = ROOT / "experiments/results" / f"{args.out}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", "utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
