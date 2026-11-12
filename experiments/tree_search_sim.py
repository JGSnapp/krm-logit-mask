"""Experiment 4: offline simulation of beam search over the phrase tree.

collect   (GPU, once): per-token log P(phrase tokens | text) for every catalog
          phrase and profile, plus per-token log P(... | empty text). Saved to
          experiments/results/exp04_token_logprobs.json.gz so any search
          variant can be replayed later without the model.
simulate  (CPU, seconds): beam search over the trie of phrase tokens. At each
          depth the children of the live nodes are ranked by the prefix score
              cond(prefix) - alpha * prior(prefix)
          and the best `width` are kept. The per-request model cost of such a
          search is one forward per depth with a branching node, for the whole
          beam at once; it does not depend on the catalog size.

Usage:
    .venv/bin/python experiments/tree_search_sim.py collect
    .venv/bin/python experiments/tree_search_sim.py simulate
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
DATA = ROOT / "experiments/results/exp04_token_logprobs.json.gz"
RESULT = ROOT / "experiments/results/exp04_tree_search.json"
WIDTHS = [1, 2, 4, 8, 16, 32, 64, 128]


def collect() -> None:
    from run_selection_eval import load_cases

    from krm_mask.catalog import Catalog
    from krm_mask.graph import PhraseGraph
    from krm_mask.model import DEFAULT_MODEL, LocalCausalLM
    from krm_mask.pipeline import EMPTY_HISTORY, EMPTY_SIGNALS, _signal_messages

    catalog = Catalog.from_json(ROOT / "data/itmo_krm_3.json")
    lm = LocalCausalLM.load(DEFAULT_MODEL)
    graph = PhraseGraph.from_catalog(catalog, None, {"indicators", "keywords"})
    phrases = list(graph.phrases)
    delimiter = lm.tokenizer.encode("\n", add_special_tokens=False)
    tokens = [lm.tokenizer.encode(p, add_special_tokens=False) + delimiter for p in phrases]

    def score(text: str) -> list[list[float]]:
        prompt = lm.prompt_ids(_signal_messages(text, EMPTY_HISTORY, EMPTY_SIGNALS))
        values = lm.score_continuations(prompt, tokens, per_token=True)
        return [[round(v, 4) for v in row] for row in values]

    started = time.perf_counter()
    prior = score("")
    print(f"prior {time.perf_counter() - started:.1f}s", flush=True)
    cases, _ = load_cases(role_filter=False)
    profiles = {}
    for case in cases:
        started = time.perf_counter()
        profiles[case["id"]] = {
            "kind": case["kind"], "acceptable": case["acceptable"], "cond": score(case["text"]),
        }
        print(case["id"], f"{time.perf_counter() - started:.1f}s", flush=True)
    payload = {
        "model": DEFAULT_MODEL,
        "phrases": phrases,
        "competencies": [sorted(graph.phrases[p]) for p in phrases],
        "tokens": tokens,
        "prior": prior,
        "profiles": profiles,
    }
    with gzip.open(DATA, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False)
    print("saved", DATA)


class Trie:
    def __init__(self, tokens: list[list[int]], cond: list[list[float]], prior: list[list[float]]):
        self.children: dict[tuple, set[int]] = defaultdict(set)
        self.terminal: dict[tuple, list[int]] = defaultdict(list)
        self.cond: dict[tuple, float] = {}
        self.prior: dict[tuple, float] = {}
        for index, sequence in enumerate(tokens):
            c = q = 0.0
            for depth, token in enumerate(sequence):
                c += cond[index][depth]
                q += prior[index][depth]
                prefix = tuple(sequence[: depth + 1])
                self.children[tuple(sequence[:depth])].add(token)
                self.cond.setdefault(prefix, c)
                self.prior.setdefault(prefix, q)
            self.terminal[tuple(sequence)].append(index)

    def score(self, node: tuple, alpha: float) -> float:
        return self.cond[node] - alpha * self.prior[node]

    def beam(self, width: int, alpha: float) -> tuple[list[int], int, int]:
        """Return found phrase indices, depth steps, and steps that need a model call."""
        live: list[tuple] = [()]
        found: list[int] = []
        steps = calls = 0
        while live:
            candidates = [node + (token,) for node in live for token in self.children[node]]
            if not candidates:
                break
            steps += 1
            calls += any(len(self.children[node]) > 1 for node in live)
            candidates.sort(key=lambda node: self.score(node, alpha), reverse=True)
            live = []
            for node in candidates[:width]:
                found.extend(self.terminal.get(node, []))
                if self.children.get(node):
                    live.append(node)
        return found, steps, calls


def top_competencies(indices, scores, competencies, k=3) -> list[str]:
    best: dict[str, float] = {}
    for index in indices:
        for competency_id in competencies[index]:
            best[competency_id] = max(best.get(competency_id, float("-inf")), scores[index])
    return sorted(best, key=lambda c: -best[c])[:k]


def simulate() -> None:
    with gzip.open(DATA, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    tokens, competencies, prior = data["tokens"], data["competencies"], data["prior"]
    everything = list(range(len(tokens)))
    report: dict[str, object] = {"widths": WIDTHS, "configs": {}}
    for alpha in (0.0, 1.0):
        config: dict[str, dict] = {}
        for width in WIDTHS + ["full"]:
            hits = agree = 0
            web = 0
            synthetic_hits = 0
            steps_sum = calls_sum = 0
            for profile in data["profiles"].values():
                cond = profile["cond"]
                full_scores = [sum(c) - alpha * sum(q) for c, q in zip(cond, prior)]
                full_top = top_competencies(everything, full_scores, competencies)
                if width == "full":
                    found, steps, calls = everything, 0, 0
                else:
                    found, steps, calls = Trie(tokens, cond, prior).beam(width, alpha)
                top = top_competencies(found, full_scores, competencies)
                acceptable = set(profile["acceptable"])
                if profile["kind"] == "web":
                    web += 1
                    hits += len(set(top) & acceptable)
                    agree += len(set(top) & set(full_top))
                    steps_sum += steps
                    calls_sum += calls
                else:
                    synthetic_hits += bool(top) and top[0] in acceptable
            config[str(width)] = {
                "web_precision_top3": round(hits / (3 * web), 3),
                "agreement_with_full_top3": round(agree / (3 * web), 3),
                "synthetic_hit_at_1": synthetic_hits,
                "mean_depth_steps": round(steps_sum / web, 1),
                "mean_model_calls": round(calls_sum / web, 1),
            }
            print(f"alpha={alpha} width={width}: {config[str(width)]}", flush=True)
        report["configs"][f"alpha_{alpha}"] = config
    RESULT.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", "utf-8")
    print("saved", RESULT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["collect", "simulate"])
    {"collect": collect, "simulate": simulate}[parser.parse_args().command]()
