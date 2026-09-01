"""
verify_links.py — Checks the availability and HTTP status of official policy source URLs.
"""

import json
import os
import sys
import httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def verify_all_links():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sources_path = os.path.join(root_dir, "data", "policy_sources.json")
    if not os.path.exists(sources_path):
        sources_path = "policy_sources.json"

    with open(sources_path, "r", encoding="utf-8") as f:
        sources = json.load(f)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    print("=" * 70)
    print("AUDITING OFFICIAL POLICY LINKS (HTTP STATUS CHECK)")
    print("=" * 70)

    for s in sources:
        url = s["source_url"]
        if "example.com" in url:
            print(f"[{s['id']}] {s['name']}\n  URL: {url} -> [INTERNAL STORE POLICY MOCK]\n")
            continue
        try:
            resp = httpx.get(url, headers=headers, follow_redirects=True, timeout=12.0)
            status = resp.status_code
            status_tag = "✅ 200 OK" if status == 200 else f"❌ {status}"
            print(f"[{s['id']}] {s['name']}\n  URL: {url}\n  Status: {status_tag}\n")
        except Exception as e:
            print(f"[{s['id']}] {s['name']}\n  URL: {url}\n  Error: {e}\n")


if __name__ == "__main__":
    verify_all_links()
