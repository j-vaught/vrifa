#!/usr/bin/env python3
"""Search CrossRef for a paper by title (and optional author/year)
and print the top candidates so a real DOI can be chosen by hand.

Usage:
  python3 find_doi.py "title to search for"
  python3 find_doi.py "title" --author "Lastname" --year 2014
"""

import argparse
import json
import sys
from urllib.parse import urlencode

import requests

USER_AGENT = "vrifa-doi-finder/1.0 (mailto:jvaught@sc.edu)"


def search(title, author=None, year=None, rows=5):
    params = {"query.title": title, "rows": rows}
    if author:
        params["query.author"] = author
    url = "https://api.crossref.org/works?" + urlencode(params)
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
    resp.raise_for_status()
    items = resp.json()["message"]["items"]
    if year:
        # Soft filter — keep ±1 year
        kept = []
        for it in items:
            for k in ("published-print", "published-online", "issued", "created"):
                date = it.get(k) or {}
                parts = date.get("date-parts", [[None]])
                if parts and parts[0] and parts[0][0]:
                    y = parts[0][0]
                    if abs(int(y) - int(year)) <= 1:
                        kept.append(it)
                    break
        items = kept or items
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("title")
    ap.add_argument("--author", default=None)
    ap.add_argument("--year", default=None)
    ap.add_argument("--rows", type=int, default=5)
    args = ap.parse_args()

    results = search(args.title, args.author, args.year, args.rows)
    if not results:
        print("No results")
        return 1
    for i, it in enumerate(results, 1):
        title = (it.get("title") or [""])[0]
        doi = it.get("DOI", "<none>")
        authors = ", ".join(
            f"{a.get('family','?')}, {a.get('given','?')}"
            for a in it.get("author", [])[:5]
        )
        for k in ("published-print", "published-online", "issued", "created"):
            date = it.get(k) or {}
            parts = date.get("date-parts", [[None]])
            if parts and parts[0] and parts[0][0]:
                year = parts[0][0]
                break
        else:
            year = "?"
        container = (it.get("container-title") or [""])[0]
        print(f"[{i}] doi: {doi}")
        print(f"    title: {title}")
        print(f"    authors: {authors}")
        print(f"    year: {year}  venue: {container}")
        print()


if __name__ == "__main__":
    sys.exit(main() or 0)
