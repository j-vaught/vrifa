#!/usr/bin/env python3
"""Verify each entry in refs.bib against CrossRef metadata.

For every entry with a DOI: fetch the CrossRef record at that DOI and check
that bib title, year, and author surnames are consistent.

For every entry without a DOI: search CrossRef by title and report the best
candidate so the user can decide whether to add a DOI.

CrossRef is the canonical metadata source for DOIs, is free, and requires no
auth; the User-Agent header below opts into their polite pool.

Usage: python3 verify_refs.py [path/to/refs.bib] [--delay 0.3]
"""

import argparse
import re
import sys
import time
from collections import namedtuple
from pathlib import Path
from urllib.parse import quote, urlencode

import requests

USER_AGENT = "vrifa-ref-verifier/1.0 (mailto:jvaught@sc.edu)"
CROSSREF_TIMEOUT = 20

Entry = namedtuple("Entry", "key kind fields")


def parse_bib(text: str) -> list[Entry]:
    """Brace-counting parser that tolerates nested braces in titles/authors."""
    entries = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "@":
            i += 1
            continue
        # Skip line comments
        line_start = text.rfind("\n", 0, i) + 1
        if text[line_start:i].lstrip().startswith("%"):
            i += 1
            continue
        m = re.match(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text[i:])
        if not m:
            i += 1
            continue
        kind = m.group(1).lower()
        key = m.group(2)
        cursor = i + m.end()
        # Find matching closing brace of the entry body
        depth = 1
        body_start = cursor
        while cursor < n and depth > 0:
            c = text[cursor]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            cursor += 1
        body = text[body_start:cursor - 1]
        i = cursor

        if kind in {"comment", "preamble", "string"}:
            continue

        fields = parse_fields(body)
        entries.append(Entry(key, kind, fields))
    return entries


def parse_fields(body: str) -> dict:
    """Walk the body and pull out 'name = {value}' or 'name = "value"' pairs."""
    fields = {}
    j = 0
    n = len(body)
    while j < n:
        m = re.match(r"\s*(\w+)\s*=\s*", body[j:])
        if not m:
            j += 1
            continue
        name = m.group(1).lower()
        j += m.end()
        if j >= n:
            break
        if body[j] == "{":
            depth = 1
            j += 1
            val_start = j
            while j < n and depth > 0:
                c = body[j]
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                j += 1
            value = body[val_start:j - 1]
        elif body[j] == '"':
            j += 1
            val_start = j
            while j < n and body[j] != '"':
                j += 1
            value = body[val_start:j]
            if j < n:
                j += 1
        else:
            # Unquoted value (number, single token)
            val_start = j
            while j < n and body[j] not in ",\n":
                j += 1
            value = body[val_start:j].strip()
        # Skip trailing comma/whitespace
        while j < n and body[j] in ", \n\t":
            j += 1
        fields[name] = re.sub(r"\s+", " ", value).strip()
    return fields


def normalize_title(s: str) -> str:
    s = s.lower().replace("{", "").replace("}", "")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def title_jaccard(a: str, b: str) -> float:
    a_tokens = set(normalize_title(a).split())
    b_tokens = set(normalize_title(b).split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def extract_surnames(author_field: str) -> list[str]:
    surnames = []
    for chunk in author_field.split(" and "):
        chunk = chunk.strip().rstrip(",")
        if "," in chunk:
            surname = chunk.split(",")[0]
        else:
            tokens = chunk.split()
            surname = tokens[-1] if tokens else ""
        surname = surname.lower().replace("{", "").replace("}", "")
        surname = re.sub(r"[^a-z\- ]", "", surname).strip()
        if surname:
            surnames.append(surname)
    return surnames


def fetch_doi(doi: str):
    url = f"https://api.crossref.org/works/{quote(doi)}"
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=CROSSREF_TIMEOUT)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()["message"]


def fetch_title_search(title: str, rows: int = 3):
    url = "https://api.crossref.org/works?" + urlencode({
        "query.title": title,
        "rows": rows,
    })
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=CROSSREF_TIMEOUT)
    resp.raise_for_status()
    return resp.json()["message"]["items"]


def crossref_year(meta: dict) -> str | None:
    for key in ("published-print", "published-online", "issued", "created"):
        date = meta.get(key) or {}
        parts = date.get("date-parts", [[None]])
        if parts and parts[0] and parts[0][0]:
            return str(parts[0][0])
    return None


def assess_against_doi(entry: Entry, meta: dict) -> tuple[float, list[str]]:
    issues = []
    bib_title = entry.fields.get("title", "")
    cr_title = (meta.get("title") or [""])[0]
    sim = title_jaccard(bib_title, cr_title)
    if sim < 0.5:
        issues.append(f"title mismatch (Jaccard {sim:.2f}): bib='{bib_title}' crossref='{cr_title}'")

    bib_year = entry.fields.get("year", "").strip()
    cr_year = crossref_year(meta)
    if bib_year and cr_year and bib_year != cr_year:
        issues.append(f"year mismatch: bib={bib_year} crossref={cr_year}")

    bib_authors = extract_surnames(entry.fields.get("author", ""))
    cr_authors = [
        (a.get("family") or "").lower()
        for a in meta.get("author", [])
        if a.get("family")
    ]
    missing = [s for s in bib_authors if s and s not in cr_authors]
    if missing:
        issues.append(f"surname(s) absent from crossref author list: {missing} (crossref: {cr_authors})")

    return sim, issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bibfile", nargs="?", default="refs.bib")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Seconds between CrossRef calls (default 0.3)")
    parser.add_argument("--only", default=None,
                        help="Comma-separated list of bib keys to verify (skip others)")
    args = parser.parse_args()

    path = Path(args.bibfile)
    if not path.exists():
        print(f"refs.bib not found at {path}")
        return 1

    entries = parse_bib(path.read_text(encoding="utf-8"))
    if not entries:
        print(f"No entries parsed from {path}")
        return 1

    only = set(s.strip() for s in args.only.split(",")) if args.only else None
    if only:
        entries = [e for e in entries if e.key in only]
        if not entries:
            print("No matching keys in --only filter")
            return 1

    print(f"Verifying {len(entries)} entries from {path}\n")

    passes, warns, fails = [], [], []

    for i, entry in enumerate(entries, 1):
        doi = entry.fields.get("doi", "").strip().rstrip(".")
        prefix = f"[{i:>2}/{len(entries)}] {entry.key}"
        try:
            if doi:
                meta = fetch_doi(doi)
                if meta is None:
                    msg = f"DOI {doi} returns 404"
                    print(f"FAIL {prefix}: {msg}")
                    fails.append((entry.key, msg))
                else:
                    sim, issues = assess_against_doi(entry, meta)
                    if not issues:
                        print(f"PASS {prefix} (DOI ok, title Jaccard {sim:.2f})")
                        passes.append(entry.key)
                    else:
                        print(f"WARN {prefix}:")
                        for iss in issues:
                            print(f"     - {iss}")
                        warns.append((entry.key, "; ".join(issues)))
            else:
                title = entry.fields.get("title", "")
                if not title:
                    msg = "no DOI and no title to search"
                    print(f"FAIL {prefix}: {msg}")
                    fails.append((entry.key, msg))
                else:
                    results = fetch_title_search(title)
                    if not results:
                        msg = "no DOI; title search returned nothing"
                        print(f"WARN {prefix}: {msg}")
                        warns.append((entry.key, msg))
                    else:
                        top = results[0]
                        top_title = (top.get("title") or [""])[0]
                        top_doi = top.get("DOI", "<none>")
                        sim = title_jaccard(title, top_title)
                        if sim >= 0.7:
                            msg = f"no DOI; consider adding {top_doi} (Jaccard {sim:.2f}: '{top_title}')"
                        else:
                            msg = f"no DOI; weak title-search match (Jaccard {sim:.2f}, top: '{top_title}')"
                        print(f"WARN {prefix}: {msg}")
                        warns.append((entry.key, msg))
        except requests.RequestException as e:
            msg = f"network error: {e}"
            print(f"FAIL {prefix}: {msg}")
            fails.append((entry.key, msg))
        time.sleep(args.delay)

    print()
    print("=" * 72)
    print(f"Summary: {len(passes)} pass, {len(warns)} warn, {len(fails)} fail")

    if fails:
        print("\nFailures (must fix):")
        for key, msg in fails:
            print(f"  - {key}: {msg}")
    if warns:
        print("\nWarnings (review):")
        for key, msg in warns:
            print(f"  - {key}: {msg}")

    return 0 if not fails else 2


if __name__ == "__main__":
    sys.exit(main())
