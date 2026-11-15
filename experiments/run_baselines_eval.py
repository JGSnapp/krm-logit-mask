"""Experiment 27: baselines on the same profiles and metric.

1. BM25: a competency is one document (name, formulation, indicators, topics),
   lemmatized like the word search; the query is the lemmatized resume.
2. Embeddings: intfloat/multilingual-e5-base; every resume sentence ("query: ")
   against every catalog phrase of a competency ("passage: "); a competency
   scores its best sentence-phrase cosine.
3. Retrieval + check: the 15 best competencies of BM25 / embeddings are checked
   with the simple yes/no question of experiment 19 (Qwen 1.5B) and re-ranked by
   the margin; the top ones with ДА are selected (the usual RAG baseline).
4. Whole catalog in the prompt: the model sees all competencies ("code - name")
   and answers with up to `max_competencies` codes (greedy), Qwen 1.5B and 3B.
Reference (experiment 26): word search + check, quote first, 1.5B.

Usage: .venv/bin/python experiments/run_baselines_eval.py [--eval-set ...] [--out ...]
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_selection_eval import load_cases  # noqa: E402
from run_simple_verify_eval import NO, YES, first_tokens, question  # noqa: E402

from krm_mask.catalog import Catalog  # noqa: E402
from krm_mask.graph import NONE_ID, PhraseGraph  # noqa: E402
from krm_mask.model import DEFAULT_MODEL, LocalCausalLM  # noqa: E402
from krm_mask.selection import split_sentences  # noqa: E402
from krm_mask.wordbag import SENTENCE_RE, pymorphy_lemmatizer, split_words  # noqa: E402

EMBEDDER = "intfloat/multilingual-e5-base"
POOL = 15
CATALOG_MODELS = {"catalog_prompt_1.5b": DEFAULT_MODEL, "catalog_prompt_3b": "Qwen/Qwen2.5-3B-Instruct"}


def competency_phrases(catalog: Catalog) -> dict[str, list[str]]:
    graph = PhraseGraph.from_catalog(catalog, None, {"indicators", "keywords"})
    phrases: dict[str, list[str]] = {c.id: [c.name] for c in catalog.competencies}
    for phrase, ids in graph.phrases.items():
        for piece in SENTENCE_RE.split(phrase):
            for competency_id in ids:
                if competency_id in phrases and piece.strip():
                    phrases[competency_id].append(piece.strip())
    return {c: list(dict.fromkeys(p)) for c, p in phrases.items()}


class BM25:
    def __init__(self, documents: dict[str, list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = {key: Counter(tokens) for key, tokens in documents.items()}
        self.lengths = {key: len(tokens) for key, tokens in documents.items()}
        self.average = sum(self.lengths.values()) / len(self.lengths)
        frequency = Counter(term for counts in self.docs.values() for term in counts)
        total = len(self.docs)
        self.idf = {t: math.log(1 + (total - n + 0.5) / (n + 0.5)) for t, n in frequency.items()}

    def scores(self, query: list[str]) -> dict[str, float]:
        result = {}
        for key, counts in self.docs.items():
            norm = self.k1 * (1 - self.b + self.b * self.lengths[key] / self.average)
            result[key] = sum(
                self.idf.get(t, 0.0) * counts[t] * (self.k1 + 1) / (counts[t] + norm)
                for t in query if t in counts
            )
        return result


def embed(texts: list[str], tokenizer, model, batch: int = 64):
    import torch

    out = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch):
            encoded = tokenizer(texts[start:start + batch], padding=True, truncation=True,
                                max_length=256, return_tensors="pt").to(model.device)
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            out.append(torch.nn.functional.normalize(pooled.float(), dim=-1))
    return torch.cat(out)


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    web = [r for r in rows if r["kind"] == "web"]
    syn = [r for r in rows if r["kind"] == "synthetic"]
    predicted = sum(len([c for c in r["selected"] if c != NONE_ID]) for r in web)
    hits = sum(len([c for c in r["selected"] if c in r["acceptable"]]) for r in web)
    summary = {
        "web_precision": round(hits / predicted, 3) if predicted else 0.0,
        "web_hits": hits, "web_predicted": predicted,
        "synthetic_hit_at_1": sum(r["selected"][0] in r["acceptable"] for r in syn) / len(syn),
        "seconds_web": round(sum(r["seconds"] for r in web) / len(web), 2),
    }
    if all("pool" in r for r in web):
        summary["acceptable_in_pool_web"] = [len(set(r["pool"]) & set(r["acceptable"])) for r in web]
    return summary


def main() -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", default="experiments/eval_set.json")
    parser.add_argument("--out", default="exp27_baselines")
    args = parser.parse_args()
    cases, spec = load_cases(role_filter=False, eval_set=args.eval_set)
    catalog = Catalog.from_json(ROOT / spec.get("catalog", "data/itmo_krm_3.json"))
    names = {c.id: c.name for c in catalog.competencies}
    phrases = competency_phrases(catalog)
    lemmatize = pymorphy_lemmatizer()

    def lemmas(text: str) -> list[str]:
        return [lemma for word in split_words(text) if (lemma := lemmatize(word))]

    report: dict[str, object] = {"experiment": args.out, "eval_set": args.eval_set,
                                 "embedder": EMBEDDER, "pool": POOL, "methods": {}}
    rankings: dict[str, dict[str, list[str]]] = {"bm25": {}, "embeddings": {}}
    rows: dict[str, list[dict[str, object]]] = {}

    def add(method: str, case: dict, selected: list[str], seconds: float, **extra) -> None:
        rows.setdefault(method, []).append({
            "id": case["id"], "kind": case["kind"], "acceptable": case["acceptable"],
            "selected": selected or [NONE_ID], "seconds": round(seconds, 3), **extra})

    # 1. BM25
    bm25 = BM25({c: lemmas(" ".join(p)) for c, p in phrases.items()})
    for case in cases:
        started = time.perf_counter()
        scores = bm25.scores(lemmas(case["text"]))
        ranking = sorted(scores, key=lambda c: -scores[c])
        rankings["bm25"][case["id"]] = ranking
        add("bm25", case, ranking[: case["max_competencies"]], time.perf_counter() - started,
            pool=ranking[:POOL])

    # 2. Embeddings
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDER)
    encoder = AutoModel.from_pretrained(EMBEDDER, dtype=torch.float16).to("cuda").eval()
    flat = [(c, p) for c, ps in phrases.items() for p in ps]
    passages = embed([f"passage: {p}" for _, p in flat], tokenizer, encoder)
    owners = [c for c, _ in flat]
    for case in cases:
        started = time.perf_counter()
        sentences = [case["text"][a:b] for a, b in split_sentences(case["text"])] or [case["text"]]
        queries = embed([f"query: {s}" for s in sentences], tokenizer, encoder)
        best = (queries @ passages.T).max(dim=0).values.tolist()
        scores: dict[str, float] = {}
        for owner, value in zip(owners, best):
            scores[owner] = max(scores.get(owner, -1.0), value)
        ranking = sorted(scores, key=lambda c: -scores[c])
        rankings["embeddings"][case["id"]] = ranking
        add("embeddings", case, ranking[: case["max_competencies"]], time.perf_counter() - started,
            pool=ranking[:POOL])
    del encoder, passages
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Retrieval + yes/no check (1.5B), and 4. catalog in the prompt
    for method, model_name in CATALOG_MODELS.items():
        lm = LocalCausalLM.load(model_name)
        if method == "catalog_prompt_1.5b":
            yes_ids, no_ids = first_tokens(lm.tokenizer, YES), first_tokens(lm.tokenizer, NO)
            cache: dict[tuple[str, str], float] = {}

            def margin(text: str, competency_id: str) -> float:
                if (text, competency_id) not in cache:
                    ids = lm.prompt_ids(question(text, names[competency_id]))
                    with torch.inference_mode():
                        logits = lm.model(input_ids=ids, logits_to_keep=1).logits[0, -1].float()
                    logp = torch.log_softmax(logits, dim=-1)
                    cache[(text, competency_id)] = float(
                        torch.logsumexp(logp[yes_ids], 0) - torch.logsumexp(logp[no_ids], 0))
                return cache[(text, competency_id)]

            for retriever in ("bm25", "embeddings"):
                for case in cases:
                    started = time.perf_counter()
                    pool = rankings[retriever][case["id"]][:POOL]
                    margins = {c: margin(case["text"], c) for c in pool}
                    selected = [c for c in sorted(pool, key=lambda c: -margins[c])
                                if margins[c] > 0][: case["max_competencies"]]
                    add(f"{retriever}+check", case, selected, time.perf_counter() - started,
                        pool=pool, margins={c: round(m, 2) for c, m in margins.items()})
        listing = "\n".join(f"{c.id} — {c.name}" for c in catalog.competencies)
        codes = sorted(names, key=len, reverse=True)
        code_re = re.compile("|".join(re.escape(c) for c in codes))
        for case in cases:
            started = time.perf_counter()
            limit = case["max_competencies"]
            messages = [{"role": "user", "content": (
                f"Профиль кандидата:\n{case['text']}\n\nСписок компетенций:\n{listing}\n\n"
                f"Выбери не более {limit} компетенций из списка, которые есть у кандидата. "
                "Ответь только кодами компетенций через запятую. "
                "Если ни одна не подходит, ответь НЕТ."
            )}]
            answer_ids = lm.generate(lm.prompt_ids(messages), [], max_new_tokens=48)
            answer = lm.tokenizer.decode(answer_ids, skip_special_tokens=True)
            selected = list(dict.fromkeys(code_re.findall(answer)))[:limit]
            add(method, case, selected, time.perf_counter() - started, answer=answer.strip())
        del lm
        gc.collect()
        torch.cuda.empty_cache()

    for method, method_rows in rows.items():
        summary = summarize(method_rows)
        report["methods"][method] = {"summary": summary, "rows": method_rows}
        print(method, json.dumps(summary, ensure_ascii=False), flush=True)
    out = ROOT / "experiments/results" / f"{args.out}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", "utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
