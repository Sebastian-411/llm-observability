"""Download a curated set of public Paul Graham essays into data/yc/essays/.

These essays are publicly available on paulgraham.com. We strip the HTML and
save plain text so the existing loaders can chunk them like any other doc.

Usage:
    python -m scripts.fetch_yc_docs
    python -m scripts.fetch_yc_docs --limit 5

Then ingest:
    python -m scripts.ingest_example data/yc
"""
from __future__ import annotations

import argparse
import re
import sys
from html import unescape
from pathlib import Path

import httpx

from app.core.logging import configure_logging, get_logger


PG_ESSAYS = [
    ("ds.html", "do_things_that_dont_scale"),
    ("startupideas.html", "how_to_get_startup_ideas"),
    ("aord.html", "default_alive_or_default_dead"),
    ("makersschedule.html", "makers_schedule_managers_schedule"),
    ("growth.html", "startup_equals_growth"),
    ("start.html", "how_to_start_a_startup"),
    ("hp.html", "hiring_is_obsolete"),
    ("ramenprofitable.html", "ramen_profitable"),
    ("mit.html", "what_you_cant_say"),
    ("good.html", "be_good"),
]

BASE_URL = "https://www.paulgraham.com/"


# Quick-and-dirty HTML→text. paulgraham.com has very simple markup, so a regex
# pass is enough — we don't need a full HTML parser for this corpus.
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"[ \t]+")
_BLANKLINE_RE = re.compile(r"\n\s*\n\s*\n+")


def _html_to_text(html: str) -> str:
    # Drop <script> and <style> blocks first.
    html = re.sub(r"<script.*?</script>", "", html, flags=re.S | re.I)
    html = re.sub(r"<style.*?</style>", "", html, flags=re.S | re.I)
    # Convert <br> and </p> to newlines so structure survives.
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    html = re.sub(r"</p>", "\n\n", html, flags=re.I)
    # Strip remaining tags.
    text = _TAG_RE.sub("", html)
    text = unescape(text)
    text = _SPACE_RE.sub(" ", text)
    text = _BLANKLINE_RE.sub("\n\n", text)
    return text.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Max essays to fetch.")
    parser.add_argument(
        "--out", type=Path, default=Path("data/yc/essays"), help="Output directory."
    )
    args = parser.parse_args()

    configure_logging("INFO")
    log = get_logger("scripts.fetch_yc")

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    essays = PG_ESSAYS[: args.limit] if args.limit else PG_ESSAYS

    saved = 0
    with httpx.Client(timeout=20.0, follow_redirects=True) as client:
        for slug, fname in essays:
            url = BASE_URL + slug
            try:
                r = client.get(url)
                r.raise_for_status()
            except httpx.HTTPError as exc:
                log.warning("fetch.failed", url=url, error=str(exc))
                continue

            text = _html_to_text(r.text)
            if len(text) < 500:
                log.warning("fetch.too_short", url=url, length=len(text))
                continue

            out_path = out_dir / f"{fname}.txt"
            out_path.write_text(text, encoding="utf-8")
            saved += 1
            log.info("fetch.saved", file=str(out_path), chars=len(text))

    log.info("fetch.done", saved=saved, total=len(essays), out_dir=str(out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
