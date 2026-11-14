"""Smoke test of the running service: POST /api/select for one example profile.

Usage: .venv/bin/python experiments/smoke_select_api.py [profile_id] [catalog_id]
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "http://localhost:8000"


def main() -> None:
    profile_id = sys.argv[1] if len(sys.argv) > 1 else "ai-security-reviewer"
    catalog_id = sys.argv[2] if len(sys.argv) > 2 else None
    for _ in range(60):
        try:
            urllib.request.urlopen(f"{URL}/api/catalogs", timeout=3)
            break
        except OSError:
            time.sleep(2)
    profiles = {p["id"]: p for p in json.loads((ROOT / "web/profiles.json").read_text("utf-8"))}
    body = json.dumps({"text": profiles[profile_id]["text"], "catalog_id": catalog_id,
                       "max_competencies": 3, "pool": 15}).encode()
    request = urllib.request.Request(f"{URL}/api/select", data=body,
                                     headers={"Content-Type": "application/json"})
    result = json.load(urllib.request.urlopen(request, timeout=900))
    print({key: round(value, 1) for key, value in result["timings"].items()})
    for item in result["selected"]:
        quote = (item.get("quote") or {}).get("text") or ""
        print(item["id"], item["name"][:55], item.get("margin"), "|", quote[:130])


if __name__ == "__main__":
    main()
