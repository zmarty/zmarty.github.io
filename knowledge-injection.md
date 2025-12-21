```python
#!/usr/bin/env python3
"""
llms.txt Python consumer crawler (NO manifest; resumes from local files only)

Resuming logic:
- A URL is considered "already crawled" if its mapped local file exists and is non-empty.
- Cached files (.md/.txt/llms*.txt) are parsed to rebuild the queue on reruns.

Features:
- --same-domain-only, --drop-query
- --skip-path-infix (repeatable): skip URLs containing substring (prints why)
- --remove-from-path (repeatable): remove substring from URL PATH before visiting
- [fetch] prints full URL
- On HTTP error, prints a bounded excerpt of response body (if any)

Usage:
  python llms_consumer_crawler.py https://developer.cybersource.com/llms.txt -o cybersource_docs \
    --same-domain-only --drop-query --max-depth 1000000 --max-files 1000000 --delay 0.3
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urlunparse
from collections import deque

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# --- Markdown link parsing ---
INLINE_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
REF_DEF_RE = re.compile(r"^\s*\[[^\]]+\]:\s*(\S+)", re.MULTILINE)
AUTO_LINK_RE = re.compile(r"<(https?://[^>]+)>")

MARKDOWN_EXTS = {".md", ".markdown", ".mdx"}
TEXT_EXTS = {".txt"}

DEFAULT_ACCEPT = "text/markdown,text/plain;q=0.9,*/*;q=0.1"


def normalize_url(url: str) -> str:
    """Normalize URL for de-duplication: remove fragment (#...)."""
    p = urlparse(url)
    p = p._replace(fragment="")
    return urlunparse(p)


def apply_remove_from_path(url: str, removals: List[str]) -> str:
    """Remove substrings from URL *path* only (safer than global replace)."""
    if not removals:
        return url
    p = urlparse(url)
    path = p.path or "/"
    for s in removals:
        if s:
            path = path.replace(s, "")
    if not path.startswith("/"):
        path = "/" + path
    if path == "":
        path = "/"
    return urlunparse(p._replace(path=path))


def extract_links_from_markdown(md: str) -> List[str]:
    links: List[str] = []
    links.extend(INLINE_LINK_RE.findall(md))
    links.extend(REF_DEF_RE.findall(md))
    links.extend(AUTO_LINK_RE.findall(md))
    return [u.strip() for u in links]


def safe_path_component(s: str) -> str:
    s = s.replace("\\", "_").replace(":", "_").replace("..", "_")
    return s


def url_to_local_path(out_dir: Path, url: str) -> Path:
    """
    Mirror URL path under: out_dir/<host>/<path>.
    If path ends with '/', use 'index'.
    If no extension, save as '.bin' (we generally only fetch md/txt/llms.txt anyway).
    """
    p = urlparse(url)
    host = safe_path_component(p.netloc or "unknown-host")
    path = p.path or "/"

    if path.endswith("/"):
        path = path + "index"

    ext = Path(path).suffix
    if not ext:
        path = path + ".bin"

    rel = Path(*[safe_path_component(x) for x in path.split("/") if x])
    return out_dir / host / rel


def make_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent, "Accept": DEFAULT_ACCEPT})

    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


@dataclass
class FetchResult:
    url: str
    local_path: str
    status_code: Optional[int]
    content_type: Optional[str]
    ok: bool
    cached: bool = False
    error: Optional[str] = None


def should_follow(url: str, same_domain_only: bool, allowed_hosts: Set[str]) -> bool:
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        return False
    if same_domain_only and p.netloc not in allowed_hosts:
        return False
    return True


def looks_like_textual(content_type: str | None) -> bool:
    if not content_type:
        return False
    ct = content_type.lower()
    return ("text/markdown" in ct) or ("text/plain" in ct) or ct.startswith("text/")


def collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def read_local_text(path: Path) -> str:
    b = path.read_bytes()
    try:
        return b.decode("utf-8")
    except Exception:
        return b.decode("utf-8", errors="replace")


def first_matching_infix(url: str, infixes: List[str]) -> Optional[str]:
    for inf in infixes:
        if inf and inf in url:
            return inf
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Site root URL (https://example.com/) or direct llms.txt URL")
    ap.add_argument("-o", "--out", default="llms_crawl_out", help="Output directory")
    ap.add_argument("--delay", type=float, default=0.2, help="Delay between requests (seconds)")
    ap.add_argument("--timeout", type=float, default=20.0, help="Per-request timeout (seconds)")
    ap.add_argument("--max-files", type=int, default=500, help="Max number of downloads this run")
    ap.add_argument("--max-depth", type=int, default=10, help="Max recursion depth from llms.txt links")
    ap.add_argument("--same-domain-only", action="store_true", help="Only crawl links on the same host as root")
    ap.add_argument("--drop-query", action="store_true", help="Ignore URL query params when de-duping")
    ap.add_argument("--user-agent", default="llms-txt-consumer-crawler/2.0", help="User-Agent header")
    ap.add_argument("--stats-every", type=int, default=25, help="Print summary stats every N processed items")
    ap.add_argument("--verbose", action="store_true", help="Verbose skip/debug messages (can be noisy)")
    ap.add_argument("--error-body-max", type=int, default=800, help="Max chars of HTTP error body to print")

    ap.add_argument(
        "--skip-path-infix",
        action="append",
        default=[],
        help="Repeatable. If this substring occurs anywhere in a URL, skip it.",
    )

    ap.add_argument(
        "--remove-from-path",
        action="append",
        default=[],
        help="Repeatable. Remove this substring from the URL PATH before visiting (e.g., --remove-from-path /en/).",
    )

    args = ap.parse_args()

    skip_infixes: List[str] = [s for s in (args.skip_path_infix or []) if s]
    remove_from_path: List[str] = [s for s in (args.remove_from_path or []) if s]

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    root = args.root.strip()
    if not root.startswith(("http://", "https://")):
        print("Root must be an http(s) URL", file=sys.stderr)
        sys.exit(2)

    # Determine llms.txt URL + site root
    if root.rstrip("/").endswith("/llms.txt"):
        llms_url = root
        site_root = root[: root.rstrip("/").rfind("/llms.txt")]
        if not site_root.endswith("/"):
            site_root += "/"
    else:
        site_root = root if root.endswith("/") else root + "/"
        llms_url = urljoin(site_root, "llms.txt")

    root_host = urlparse(site_root).netloc
    allowed_hosts = {root_host}

    session = make_session(args.user_agent)

    def canon(u: str) -> str:
        u = urljoin(site_root, u)
        u = normalize_url(u)
        u = apply_remove_from_path(u, remove_from_path)
        if args.drop_query:
            p = urlparse(u)
            u = urlunparse(p._replace(query=""))
        return u

    def is_llms(url: str) -> bool:
        p = urlparse(url).path.lower()
        return p.endswith("/llms.txt") or p.endswith("/llms-full.txt") or p.endswith("llms.txt") or p.endswith("llms-full.txt")

    def is_parse_candidate(url: str) -> bool:
        p = urlparse(url)
        ext = Path(p.path).suffix.lower()
        return is_llms(url) or ext in MARKDOWN_EXTS or ext in TEXT_EXTS

    def enqueue_from_markdown(text: str, base_url: str, next_depth: int) -> Tuple[int, int, int]:
        raw_links = extract_links_from_markdown(text)

        added = 0
        kept = 0
        skipped = 0

        for raw in raw_links:
            raw = raw.strip()
            if not raw or raw.startswith(("mailto:", "javascript:", "data:")):
                skipped += 1
                continue

            abs_url = urljoin(base_url, raw)
            abs_url = normalize_url(abs_url)
            abs_url = apply_remove_from_path(abs_url, remove_from_path)
            if args.drop_query:
                p = urlparse(abs_url)
                abs_url = urlunparse(p._replace(query=""))

            m2 = first_matching_infix(abs_url, skip_infixes)
            if m2 is not None:
                skipped += 1
                print(f"[skip] matched --skip-path-infix '{m2}' url={abs_url}")
                continue

            if not should_follow(abs_url, args.same_domain_only, allowed_hosts):
                skipped += 1
                if args.verbose:
                    print(f"[skip] out of scope link: {abs_url}")
                continue

            path = urlparse(abs_url).path.lower()
            if path.endswith("/llms.txt") or path.endswith("/llms-full.txt"):
                kept += 1
            else:
                ext2 = Path(path).suffix.lower()
                if ext2 not in (MARKDOWN_EXTS | TEXT_EXTS):
                    skipped += 1
                    if args.verbose:
                        print(f"[skip] non-md/txt link: {abs_url}")
                    continue
                kept += 1

            if abs_url not in visited:
                queue.append((abs_url, next_depth))
                added += 1

        return (len(raw_links), kept, added)

    visited: Set[str] = set()
    queue = deque([(canon(llms_url), 0)])

    results: Dict[str, FetchResult] = {}

    downloaded_files = 0
    cached_hits = 0
    filtered_skips = 0
    processed_items = 0
    started = time.time()

    print(f"[init] site_root={site_root} host={root_host}")
    print(f"[init] llms_url={canon(llms_url)}")
    print(f"[init] out_dir={out_dir}")
    if args.same_domain_only:
        print("[init] same-domain-only enabled")
    if args.drop_query:
        print("[init] drop-query enabled")
    if skip_infixes:
        print(f"[init] skip-path-infix={skip_infixes}")
    if remove_from_path:
        print(f"[init] remove-from-path={remove_from_path}")
    print("[init] resume mode: cache is determined ONLY by existing local files (no manifest)")

    while queue:
        url, depth = queue.popleft()
        url = canon(url)

        if url in visited:
            continue
        visited.add(url)

        match = first_matching_infix(url, skip_infixes)
        if match is not None:
            filtered_skips += 1
            print(f"[skip] matched --skip-path-infix '{match}' url={url}")
            continue

        if depth > args.max_depth:
            if args.verbose:
                print(f"[skip] depth {depth} > max_depth {args.max_depth}: {url}")
            continue

        if not should_follow(url, args.same_domain_only, allowed_hosts):
            if args.verbose:
                print(f"[skip] out of scope (domain/scheme): {url}")
            continue

        processed_items += 1

        local_path = url_to_local_path(out_dir, url)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Cache rule: file exists and is non-empty => cached
        cached = local_path.exists() and local_path.stat().st_size > 0

        if cached:
            cached_hits += 1
            results[url] = FetchResult(
                url=url,
                local_path=str(local_path),
                status_code=200,
                content_type=None,
                ok=True,
                cached=True,
                error=None,
            )

            print(f"[cache] depth={depth} queue={len(queue)} url={url}")

            if is_parse_candidate(url):
                try:
                    text = read_local_text(local_path)
                    found, kept, enqueued = enqueue_from_markdown(text, url, depth + 1)
                    if found:
                        print(f"[parse] found={found} kept={kept} enqueued={enqueued} from={url}")
                except Exception as e:
                    print(f"[warn] cache-parse failed: {type(e).__name__}: {e} file={local_path}")

            if args.stats_every > 0 and processed_items % args.stats_every == 0:
                elapsed = max(0.001, time.time() - started)
                rate = downloaded_files / elapsed
                print(
                    f"[stats] downloaded={downloaded_files} cached={cached_hits} filtered={filtered_skips} "
                    f"visited={len(visited)} queue={len(queue)} rate={rate:.2f} dl/s elapsed={elapsed:.1f}s"
                )
            continue

        # Download limit (counts actual downloads only)
        if downloaded_files >= args.max_files:
            print(f"[stop] reached --max-files={args.max_files} (downloads)")
            break

        print(f"[fetch] dl={downloaded_files+1}/{args.max_files} depth={depth} queue={len(queue)} url={url}")

        try:
            r = session.get(url, timeout=args.timeout)
            status = r.status_code
            ct = r.headers.get("Content-Type")

            if not r.ok:
                print(f"[warn] status={status} ct={ct} url={url}")

                body_excerpt = ""
                try:
                    if looks_like_textual(ct) or (ct and "json" in ct.lower()):
                        txt = collapse_ws(r.text or "")
                        if txt:
                            body_excerpt = txt[: args.error_body_max]
                except Exception:
                    body_excerpt = ""

                if body_excerpt:
                    print(f"[warn] body: {body_excerpt}")

                results[url] = FetchResult(
                    url=url,
                    local_path=str(local_path),
                    status_code=status,
                    content_type=ct,
                    ok=False,
                    cached=False,
                    error=f"HTTP {status}",
                )
                time.sleep(args.delay)
                continue

            data = r.content
            local_path.write_bytes(data)
            downloaded_files += 1

            results[url] = FetchResult(
                url=url,
                local_path=str(local_path),
                status_code=status,
                content_type=ct,
                ok=True,
                cached=False,
                error=None,
            )

            # Parse and enqueue links if candidate
            if is_parse_candidate(url) or (looks_like_textual(ct) and is_parse_candidate(url)):
                try:
                    text = r.text
                except Exception:
                    text = data.decode("utf-8", errors="replace")

                found, kept, enqueued = enqueue_from_markdown(text, url, depth + 1)
                if found:
                    print(f"[parse] found={found} kept={kept} enqueued={enqueued} from={url}")

            if args.stats_every > 0 and processed_items % args.stats_every == 0:
                elapsed = max(0.001, time.time() - started)
                rate = downloaded_files / elapsed
                print(
                    f"[stats] downloaded={downloaded_files} cached={cached_hits} filtered={filtered_skips} "
                    f"visited={len(visited)} queue={len(queue)} rate={rate:.2f} dl/s elapsed={elapsed:.1f}s"
                )

        except Exception as e:
            print(f"[error] {type(e).__name__}: {e} url={url}")
            results[url] = FetchResult(
                url=url,
                local_path=str(local_path),
                status_code=None,
                content_type=None,
                ok=False,
                cached=False,
                error=str(e),
            )

        time.sleep(args.delay)

    elapsed = max(0.001, time.time() - started)
    print(f"[done] out_dir={out_dir}")
    print(
        f"[done] downloaded={downloaded_files} cached={cached_hits} filtered={filtered_skips} "
        f"visited={len(visited)} queue_remaining={len(queue)} elapsed={elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
```

```console
python3 llms_consumer_crawler.py https://developer.cybersource.com/llms.txt   -o cybersource_docs   --same-domain-only   --drop-query   --max-depth 1000000   --max-files 1000000   --delay 1.0 --remove-from-path content/cybsdeveloper2021/amer/en/
```

<img width="2775" height="2075" alt="image" src="https://github.com/user-attachments/assets/9a9fbdaa-7050-4b0e-ac25-fa5e2e93afe6" />

```console
 uv venv --python 3.12 --seed
source .venv/bin/activate

pip install synthetic-data-kit
mkdir -p data/{input,parsed,generated,curated,final}
```

```console
# (Different terminal)

cd /git/vllm/
.venv/bin/activate

vllm serve \
    /models/awq/QuantTrio-MiniMax-M2-AWQ \
    --served-model-name MiniMax-M2-AWQ \
    --max-num-seqs 10 \
    --max-model-len 128000 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 1 \
    --enable-auto-tool-choice \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think \
    --host 0.0.0.0 \
    --port 8000
```

```console
# Back to synthetic-data-kit terminal

# For some strange reason synthetic-data-kit does not support parsing .md files, just .txt files... so let's just rename all Markdown files to text....
find developer.cybersource.com/ -type f -name '*.md' -exec bash -c 'for f; do mv -- "$f" "${f%.md}.txt"; done' _ {} +
synthetic-data-kit ingest developer.cybersource.com/ --preview
```
