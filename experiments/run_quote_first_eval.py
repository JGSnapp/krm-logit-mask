"""Experiment 25: quote first, then competencies for the quote, then yes/no.

1. Quotes: every sentence of the resume is scored as the answer to "Выведи
   из текста цитату, которая показывает навык или опыт кандидата. Пиши сразу
   дословно, как в тексте." (mean log-probability per token, one batch); the
   best 3 sentences are kept.
2. For each fragment, the word-signal search runs on the fragment alone and
   narrows to a group (one signal is enough, <= 5 competencies; experiment 24),
   up to K candidates per fragment.
3. The simple yes/no question (experiment 19) for each (fragment, competency):
   the profile is the fragment ("fragment") or the whole resume ("resume").
4. Assembly: each quote gets its best confirmed competency ("per_quote"), or
   the candidates of all quotes are ranked together ("top").
Compared with the current scheme (search on the whole resume, pool 15, the same
1.5B check; experiment 21).

Usage: .venv/bin/python experiments/run_quote_first_eval.py [--eval-set ...] [--out ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_selection_eval import load_cases  # noqa: E402
from run_simple_verify_eval import NO, YES, first_tokens, question  # noqa: E402

from krm_mask.catalog import Catalog  # noqa: E402
from krm_mask.graph import NONE_ID, PhraseGraph  # noqa: E402
from krm_mask.model import DEFAULT_MODEL, LocalCausalLM  # noqa: E402
from krm_mask.selection import split_sentences  # noqa: E402
from krm_mask.wordbag import (  # noqa: E402
    WordBagConfig, WordBagSelector, WordIndex, pymorphy_lemmatizer, word_messages,
)

FRAGMENT_REQUEST = (
    "Выведи из текста цитату, которая показывает навык или опыт кандидата. "
    "Пиши сразу дословно, как в тексте."
)
SEARCH = {"alpha": 0.0, "beta": 0.3, "batch_paths": True}
GROUP = {"group_size": 5, "min_signals": 1, "group_at_cap": 5}
# name -> (quotes kept, candidates per quote, check profile, assembly)
#   per_quote - every quote gets its best confirmed competency (distinct quotes);
#   top       - the candidates of all quotes together, best margins overall.
VARIANTS = {
    "q3_per_quote_fragment": (3, 5, "fragment", "per_quote"),
    "q3_per_quote_resume": (3, 5, "resume", "per_quote"),
    "q3_top_fragment": (3, 5, "fragment", "top"),
    "q3_top_resume": (3, 5, "resume", "top"),
}


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--eval-set", default="experiments/eval_set.json")
    parser.add_argument("--out", default="exp25_quote_first")
    args = parser.parse_args()
    cases, spec = load_cases(role_filter=False, eval_set=args.eval_set)
    catalog = Catalog.from_json(ROOT / spec.get("catalog", "data/itmo_krm_3.json"))
    names = {item.id: item.name for item in catalog.competencies}
    lm = LocalCausalLM.load(args.model)
    tokenizer = lm.tokenizer
    graph = PhraseGraph.from_catalog(catalog, None, {"indicators", "keywords"})
    index = WordIndex.build(graph, pymorphy_lemmatizer())
    yes_ids, no_ids = first_tokens(tokenizer, YES), first_tokens(tokenizer, NO)

    def prompt(text: str) -> list[int]:
        return lm.prompt_ids(word_messages(text))[0].tolist()

    margin_cache: dict[tuple[str, str], float] = {}

    def margin(text: str, competency_id: str) -> float:
        key = (text, competency_id)
        if key not in margin_cache:
            ids = lm.prompt_ids(question(text, names[competency_id]))
            with torch.inference_mode():
                logits = lm.model(input_ids=ids, logits_to_keep=1).logits[0, -1].float()
            logp = torch.log_softmax(logits, dim=-1)
            margin_cache[key] = float(
                torch.logsumexp(logp[yes_ids], 0) - torch.logsumexp(logp[no_ids], 0))
        return margin_cache[key]

    def fragments(text: str) -> list[tuple[str, float]]:
        spans = split_sentences(text)
        messages = [{"role": "user", "content": f"Профиль кандидата: {text}\n{FRAGMENT_REQUEST}"}]
        continuations = [tokenizer.encode(text[a:b], add_special_tokens=False)
                         + [tokenizer.eos_token_id] for a, b in spans]
        totals = lm.score_continuations(lm.prompt_ids(messages), continuations)
        scored = [(text[a:b], total / len(tokens))
                  for (a, b), total, tokens in zip(spans, totals, continuations)]
        return sorted(scored, key=lambda item: -item[1])

    selector = WordBagSelector(lm, index, prompt, WordBagConfig(**SEARCH, **GROUP))
    report: dict[str, object] = {"experiment": args.out, "eval_set": args.eval_set,
                                 "search": SEARCH | GROUP, "variants": {}, "profiles": {}}
    rows: dict[str, list[dict[str, object]]] = {name: [] for name in VARIANTS}
    for case in cases:
        text = case["text"]
        started = time.perf_counter()
        ranked = fragments(text)
        fragment_seconds = time.perf_counter() - started
        largest = max(v[0] for v in VARIANTS.values())
        searched: list[dict[str, object]] = []
        started = time.perf_counter()
        for fragment, score in ranked[:largest]:
            found, calls = selector.select_paths(fragment, 5, coherent=True)
            searched.append({"fragment": fragment, "score": round(score, 3),
                             "candidates": [c for c, _ in found], "calls": calls})
        search_seconds = time.perf_counter() - started
        report["profiles"][case["id"]] = {"fragments": searched,
                                          "fragment_seconds": round(fragment_seconds, 2),
                                          "search_seconds": round(search_seconds, 2)}
        print(f"\n## {case['id']}  fragments {fragment_seconds:.1f}s, search {search_seconds:.1f}s",
              flush=True)
        for item in searched:
            print(f"   «{item['fragment'][:90]}» → {item['candidates']}", flush=True)
        for name, (keep, per_fragment, profile, assembly) in VARIANTS.items():
            started = time.perf_counter()
            best: dict[str, tuple[float, str]] = {}
            per_quote: list[list[tuple[float, str]]] = []
            for item in searched[:keep]:
                checked = []
                for competency_id in item["candidates"][:per_fragment]:
                    value = margin(item["fragment"] if profile == "fragment" else text, competency_id)
                    checked.append((value, competency_id))
                    if competency_id not in best or value > best[competency_id][0]:
                        best[competency_id] = (value, item["fragment"])
                per_quote.append(sorted(checked, reverse=True))
            check_seconds = time.perf_counter() - started
            order = sorted(best, key=lambda c: -best[c][0])
            if assembly == "per_quote":
                selected = []
                quotes_of: dict[str, str] = {}
                for item, checked in zip(searched[:keep], per_quote):
                    for value, competency_id in checked:
                        if value > 0 and competency_id not in selected:
                            selected.append(competency_id)
                            quotes_of[competency_id] = item["fragment"]
                            break
                selected = selected[: case["max_competencies"]]
                best = {c: (best[c][0], quotes_of.get(c, best[c][1])) for c in best}
            else:
                selected = [c for c in order if best[c][0] > 0][: case["max_competencies"]]
            rows[name].append({
                "id": case["id"], "kind": case["kind"], "acceptable": case["acceptable"],
                "pool": order, "acceptable_in_pool": len(set(order) & set(case["acceptable"])),
                "selected": selected or [NONE_ID],
                "quotes": {c: best[c][1] for c in selected},
                "margins": {c: round(best[c][0], 2) for c in order},
                "seconds": round(fragment_seconds + search_seconds + check_seconds, 2),
            })
            print(f"   {name}: {[(c, round(best[c][0], 1), c in case['acceptable']) for c in selected]}",
                  flush=True)
    for name, variant_rows in rows.items():
        web = [r for r in variant_rows if r["kind"] == "web"]
        syn = [r for r in variant_rows if r["kind"] == "synthetic"]
        predicted = sum(len([c for c in r["selected"] if c != NONE_ID]) for r in web)
        hits = sum(len([c for c in r["selected"] if c in r["acceptable"]]) for r in web)
        summary = {
            "web_precision": round(hits / predicted, 3) if predicted else 0.0,
            "web_hits": hits, "web_predicted": predicted,
            "synthetic_hit_at_1": sum(r["selected"][0] in r["acceptable"] for r in syn) / len(syn),
            "acceptable_in_pool_web": [r["acceptable_in_pool"] for r in web],
            "pool_size_web": [len(r["pool"]) for r in web],
            "distinct_quotes_web": [len(set(r["quotes"].values())) for r in web],
            "seconds_web": round(sum(r["seconds"] for r in web) / len(web), 2),
        }
        report["variants"][name] = {"params": VARIANTS[name], "summary": summary, "rows": variant_rows}
        print(name, json.dumps(summary, ensure_ascii=False), flush=True)
    out = ROOT / "experiments/results" / f"{args.out}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", "utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
