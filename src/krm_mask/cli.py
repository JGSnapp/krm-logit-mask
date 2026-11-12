from __future__ import annotations

import argparse
import json
from pathlib import Path

from .catalog import Catalog, extract_catalog
from .evaluation import read_jsonl, score_profiles
from .model import DEFAULT_MODEL, LocalCausalLM
from .pipeline import ExtractionPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="krm-mask")
    commands = parser.add_subparsers(dest="command", required=True)

    extract = commands.add_parser("extract-krm", help="XLSX -> normalized JSON")
    extract.add_argument("xlsx")
    extract.add_argument("--output", required=True)

    run = commands.add_parser("run", help="run constrained extraction locally")
    run.add_argument("--catalog", required=True)
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--text-file")
    run.add_argument("--role")
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--max-competencies", type=int, default=8)
    run.add_argument("--max-quotes", type=int, default=3)
    run.add_argument("--allow-abort-token", action="store_true")
    run.add_argument("--resolution-extra-chars", type=int, default=0)
    run.add_argument("--output")

    evaluate = commands.add_parser("evaluate", help="run a JSONL synthetic set")
    evaluate.add_argument("--catalog", required=True)
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--model", default=DEFAULT_MODEL)
    evaluate.add_argument("--max-competencies", type=int, default=4)
    evaluate.add_argument("--max-quotes", type=int, default=3)
    evaluate.add_argument("--allow-abort-token", action="store_true")
    evaluate.add_argument("--resolution-extra-chars", type=int, default=0)
    evaluate.add_argument("--output", required=True)

    serve = commands.add_parser("serve", help="launch the local trace interface")
    serve.add_argument("--catalog", required=True)
    serve.add_argument("--model", default=DEFAULT_MODEL)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "extract-krm":
        catalog = extract_catalog(args.xlsx)
        catalog.to_json(args.output)
        print(
            json.dumps(
                {
                    "competencies": len(catalog.competencies),
                    "indicators": sum(len(c.indicators) for c in catalog.competencies),
                    "roles": len(catalog.roles),
                    "output": args.output,
                },
                ensure_ascii=False,
            )
        )
        return

    if args.command == "serve":
        from .server import serve

        serve(args.catalog, args.model, args.host, args.port)
        return

    catalog = Catalog.from_json(args.catalog)
    if args.command == "evaluate":
        pipeline = ExtractionPipeline(catalog, LocalCausalLM.load(args.model))
        records = read_jsonl(args.dataset)
        for record in records:
            record["predictions"] = pipeline.run(
                str(record["text"]),
                record.get("role_id"),
                args.max_competencies,
                args.max_quotes,
                args.allow_abort_token,
                args.resolution_extra_chars,
            )
        report = {"metrics": score_profiles(records), "profiles": records}
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        Path(args.output).write_text(rendered, encoding="utf-8")
        print(rendered)
        return

    text = args.text or Path(args.text_file).read_text(encoding="utf-8")
    pipeline = ExtractionPipeline(catalog, LocalCausalLM.load(args.model))
    result = pipeline.run(
        text,
        args.role,
        args.max_competencies,
        args.max_quotes,
        args.allow_abort_token,
        args.resolution_extra_chars,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

