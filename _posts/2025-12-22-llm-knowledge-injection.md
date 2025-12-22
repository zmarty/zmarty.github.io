---
layout: post
title: "Injecting Knowledge into LLMs via Fine-Tuning"
date: 2025-12-21 09:30:00 -0800
categories: [AI, Development]
tags: [llm, vllm, fine-tuning, gpt-oss]
description: "A practical guide to injecting new knowledge into LLM models through fine-tuning, using Q&A pairs generated from documentation."
---

It's common wisdom that you cannot add new knowledge to LLM models by fine-tuning them. But I will attempt to dispel that notion by proving practically that you can in fact inject new facts into models. 

Here's the plan: 

1) Crawl the documents in a developer portal

2) Use a teacher model ([Minimax M2](https://huggingface.co/MiniMaxAI/MiniMax-M2)) to extract questions and answers from each separate document. Also extract Chain-of-thought sections.

3) Use this dataset to teach a student model ([gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)) new facts from that developer portal.

4) Compare the original model to the fine-tuned one.

## Crawling the raw documents

I chose to crawl the <u>**public documents**</u> from the [Cybersource developer portal](https://developer.cybersource.com/), since I am somewhat familiar with it. I found out that this portal uses the [llms.txt standard](https://llmstxt.org/) and provides Markdown documents that are easy to crawl and ingest into LLMs.

Code is cheap nowadays and since I couldn't find a simple llms.txt crawler in 2 minutes of Googling, I just used Opus 4.5 to write my own in about 10 minutes of iteration. I am including the crawler code below.

<details markdown="1">
<summary markdown="span">File `llms_consumer_crawler.py` (click to expand)</summary>

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

</details>

Command I used to run it locally saves the Markdown files in a `cybersource_docs` folder:

```console
python3 llms_consumer_crawler.py https://developer.cybersource.com/llms.txt \
  -o cybersource_docs \
  --same-domain-only \
  --drop-query \
  --max-depth 1000000 \
  --max-files 1000000 \
  --delay 1.0 \
  --skip-path-infix /content/cybsdeveloper2021/amer/en/
```

This resulted in 326 Markdown documents.

## Preparing the Q&A pairs

We can't just use the raw documents to fine-tune a model and hope for the best. Instead, a better approach is to generate question-and-answer pairs from them because: a) it splits the content into discrete, digestible pieces of information, and b) it mimics the format in which end users will actually query the model.

So instead of training the model on raw documents such as: `For business to business customers, Level II and Level III processing can provide lower interchange rates in exchange for providing more information during a transaction....`, we convert the text into pairs such as: `Q`: *What benefit does Level II and Level III processing provide for business to business customers?* `A`: *For business to business customers, Level II and Level III processing can provide lower interchange rates in exchange for providing more information during a transaction.*

Historically processing these documents in such a way would take a lot of time, money, and human capital. However, quite a lot has changed in [the last three years](https://www.ovidiudan.com/2025/11/30/three-years-chatgpt.html). Instead, we will use a local LLM model to process each document and extract these pairs.

Here is the hardware and software combination that we are using for this entire experiment:
```
CPU: AMD Ryzen 9 7950X3D 16-Core Processor
GPU: Dual NVIDIA RTX Pro 6000 (each at 96 GB VRAM)
RAM: 192 GB DDR5 5200
OS: Ubuntu 24.04
vLLM: 0.12.0
Python: 3.12
```

First, we start the teacher model using vLLM. I chose to run a [quantized version of Minimax M2](https://huggingface.co/QuantTrio/MiniMax-M2-AWQ) for this part of the task, as it is among the best open weight models available today.

```console
vllm serve \
    /models/awq/QuantTrio-MiniMax-M2-AWQ \
    --served-model-name MiniMax-M2-AWQ \
    --max-num-seqs 30 \
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

I attempted to use the [synthetic-data-kit](https://github.com/meta-llama/synthetic-data-kit) from Meta for all of 10 minutes until I realized for some strange reason it does not support ingesting Markdown files, so again I just Opus 4.5 to write my own quick version of something similar.

<details markdown="1">
<summary markdown="span">File `qa_generator.py` (click to expand)</summary>

```python
#!/usr/bin/env python3
"""
QA Pairs Generator using VLLM
Processes documents and generates Q&A training data using a local VLLM instance.
Supports parallel processing for high-throughput VLLM servers.
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import aiohttp
import requests


def load_prompt_template(prompt_file: str) -> str:
    """Load the prompt template from file."""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read()


def find_documents(folder: str, extensions: tuple = ('.md', '.txt', '.text')) -> list[Path]:
    """Recursively find all documents with specified extensions."""
    folder_path = Path(folder)
    documents = []
    for ext in extensions:
        documents.extend(folder_path.rglob(f'*{ext}'))
    return sorted(documents)


def read_document(file_path: Path) -> str:
    """Read document content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return ""


def build_prompt(template: str, document_text: str, n_max: int, n_per_category_max: int) -> str:
    """Build the final prompt by substituting placeholders."""
    prompt = template.replace('{text_of_full_document}', document_text)
    prompt = prompt.replace('{N_MAX}', str(n_max))
    prompt = prompt.replace('{N_PER_CATEGORY_MAX}', str(n_per_category_max))
    return prompt


def extract_json_from_response(response_text: str) -> Optional[list]:
    """Extract and validate JSON array from LLM response."""
    # Try to find JSON array in the response
    # First, try to parse the entire response as JSON
    try:
        data = json.loads(response_text.strip())
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON block in markdown code blocks
    patterns = [
        r'```json\s*([\s\S]*?)\s*```',  # ```json ... ```
        r'```\s*([\s\S]*?)\s*```',       # ``` ... ```
        r'\[\s*\[[\s\S]*\]\s*\]',        # Raw JSON array of arrays
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response_text, re.MULTILINE)
        for match in matches:
            try:
                # Handle the case where match is the full pattern match (for last pattern)
                text_to_parse = match if isinstance(match, str) else match[0]
                data = json.loads(text_to_parse.strip())
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                continue
    
    # Try to find the outermost [ ... ] in the response
    try:
        start_idx = response_text.find('[')
        if start_idx != -1:
            # Find matching closing bracket
            depth = 0
            for i, char in enumerate(response_text[start_idx:], start=start_idx):
                if char == '[':
                    depth += 1
                elif char == ']':
                    depth -= 1
                    if depth == 0:
                        json_str = response_text[start_idx:i+1]
                        data = json.loads(json_str)
                        if isinstance(data, list):
                            return data
                        break
    except json.JSONDecodeError:
        pass
    
    return None


def validate_message(message: dict) -> bool:
    """Validate that a message conforms to the expected schema."""
    if not isinstance(message, dict):
        return False
    
    # Required fields
    if 'role' not in message or 'content' not in message:
        return False
    
    # Role must be 'user' or 'assistant'
    if message['role'] not in ('user', 'assistant'):
        return False
    
    # Content must be a non-empty string
    if not isinstance(message['content'], str) or not message['content'].strip():
        return False
    
    # thinking field: must be null for user, string or null for assistant
    if 'thinking' in message:
        thinking = message['thinking']
        if message['role'] == 'user':
            if thinking is not None:
                return False
        else:  # assistant
            if thinking is not None and not isinstance(thinking, str):
                return False
    
    return True


def validate_conversation(conversation: list) -> bool:
    """Validate that a conversation conforms to the expected schema."""
    if not isinstance(conversation, list):
        return False
    
    # Must have at least 2 messages (user + assistant)
    if len(conversation) < 2:
        return False
    
    # Validate each message
    for message in conversation:
        if not validate_message(message):
            return False
    
    # First message should be from user
    if conversation[0].get('role') != 'user':
        return False
    
    # Should have at least one assistant response
    has_assistant = any(m.get('role') == 'assistant' for m in conversation)
    if not has_assistant:
        return False
    
    return True


def call_vllm(
    prompt: str,
    vllm_url: str,
    model: str,
    max_tokens: int = 16384,
    temperature: float = 0.7
) -> Optional[str]:
    """Call the VLLM Chat Completions API and return the response (sync version for model detection)."""
    endpoint = f"{vllm_url}/v1/chat/completions"
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    try:
        response = requests.post(endpoint, json=payload, timeout=600)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"VLLM API error: {e}", file=sys.stderr)
        return None
    except (KeyError, IndexError) as e:
        print(f"Error parsing VLLM response: {e}", file=sys.stderr)
        return None


async def call_vllm_async(
    session: aiohttp.ClientSession,
    prompt: str,
    vllm_url: str,
    model: str,
    max_tokens: int = 16384,
    temperature: float = 0.7
) -> Optional[str]:
    """Call the VLLM Chat Completions API asynchronously and return the response."""
    endpoint = f"{vllm_url}/v1/chat/completions"
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    try:
        async with session.post(endpoint, json=payload, timeout=aiohttp.ClientTimeout(total=600)) as response:
            response.raise_for_status()
            result = await response.json()
            return result['choices'][0]['message']['content']
    except aiohttp.ClientError as e:
        print(f"VLLM API error: {e}", file=sys.stderr)
        return None
    except (KeyError, IndexError) as e:
        print(f"Error parsing VLLM response: {e}", file=sys.stderr)
        return None
    except asyncio.TimeoutError:
        print(f"VLLM API timeout", file=sys.stderr)
        return None


def get_available_models(vllm_url: str) -> list[str]:
    """Get list of available models from VLLM."""
    try:
        response = requests.get(f"{vllm_url}/v1/models", timeout=10)
        response.raise_for_status()
        models = response.json()
        return [m['id'] for m in models.get('data', [])]
    except Exception as e:
        print(f"Could not fetch models: {e}", file=sys.stderr)
        return []


async def process_document(
    doc_path: Path,
    doc_index: int,
    total_docs: int,
    prompt_template: str,
    session: aiohttp.ClientSession,
    args,
    model: str,
    file_lock: asyncio.Lock,
    outfile
) -> tuple[int, bool]:
    """
    Process a single document asynchronously.
    Returns (qa_count, success).
    """
    print(f"[{doc_index}/{total_docs}] Processing: {doc_path}")
    
    # Read document
    doc_content = read_document(doc_path)
    if not doc_content.strip():
        print(f"  [{doc_index}] Skipping empty document")
        return 0, True
    
    # Build prompt
    prompt = build_prompt(
        prompt_template,
        doc_content,
        args.n_max,
        args.n_per_category_max
    )
    
    if args.verbose:
        print(f"  [{doc_index}] Prompt length: {len(prompt)} chars")
    
    # Retry loop for documents that yield zero valid results
    doc_qa_count = 0
    for attempt in range(1, args.max_retries + 1):
        if attempt > 1:
            print(f"  [{doc_index}] Retry {attempt}/{args.max_retries}...")
        
        # Call VLLM
        response = await call_vllm_async(
            session,
            prompt,
            args.vllm_url,
            model,
            args.max_tokens,
            args.temperature
        )
        
        if response is None:
            print(f"  [{doc_index}] Failed to get response from VLLM")
            if attempt == args.max_retries:
                return 0, False
            continue
        
        if args.verbose:
            print(f"  [{doc_index}] Response length: {len(response)} chars")
        
        # Extract and validate JSON
        qa_data = extract_json_from_response(response)
        
        if qa_data is None:
            print(f"  [{doc_index}] Failed to extract valid JSON from response")
            if args.verbose:
                print(f"  [{doc_index}] Response preview: {response[:500]}...")
            if attempt == args.max_retries:
                return 0, False
            continue
        
        if not qa_data:
            print(f"  [{doc_index}] Empty Q&A array (document may be too sparse)")
            return 0, True  # Empty array is valid, no retry needed
        
        # Collect valid conversations
        valid_conversations = []
        invalid_count = 0
        for conversation in qa_data:
            try:
                # Validate conversation structure against schema
                if validate_conversation(conversation):
                    valid_conversations.append(conversation)
                else:
                    invalid_count += 1
                    print(f"  [{doc_index}] [INVALID] Skipping malformed conversation: {json.dumps(conversation, ensure_ascii=False)[:300]}...")
            except Exception as e:
                invalid_count += 1
                print(f"  [{doc_index}] [ERROR] Error processing conversation: {e}")
        
        if invalid_count > 0:
            print(f"  [{doc_index}] Skipped {invalid_count} invalid conversations")
        
        # If we got at least one valid result, success - no retry needed
        if valid_conversations:
            doc_qa_count = len(valid_conversations)
            print(f"  [{doc_index}] Generated {doc_qa_count} valid Q&A conversations")
            
            # Write results immediately with lock to prevent interleaving
            async with file_lock:
                for conv in valid_conversations:
                    json_line = json.dumps(conv, ensure_ascii=False)
                    outfile.write(json_line + '\n')
                outfile.flush()
            
            return doc_qa_count, True
        else:
            # All items were invalid, retry if attempts remain
            print(f"  [{doc_index}] All {len(qa_data)} conversations were invalid")
            if attempt == args.max_retries:
                print(f"  [{doc_index}] Giving up after {args.max_retries} attempts")
                return 0, False
    
    return 0, False


async def process_documents_parallel(
    documents: list[Path],
    prompt_template: str,
    args,
    model: str,
    output_file: str
) -> tuple[int, int, int]:
    """
    Process all documents in parallel with limited concurrency.
    Results are written to file incrementally as each document completes.
    Returns (processed_count, failed_count, total_qa_pairs).
    """
    file_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(args.concurrency)
    
    # Check if file exists before opening
    if os.path.exists(output_file):
        print(f"Warning: Output file '{output_file}' already exists. Appending to it.")
    
    # Open file for the duration of processing
    with open(output_file, 'a', encoding='utf-8') as outfile:
        async def process_with_semaphore(doc_path, doc_index):
            async with semaphore:
                return await process_document(
                    doc_path,
                    doc_index,
                    len(documents),
                    prompt_template,
                    session,
                    args,
                    model,
                    file_lock,
                    outfile
                )
        
        connector = aiohttp.TCPConnector(limit=args.concurrency + 5)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                process_with_semaphore(doc_path, i)
                for i, doc_path in enumerate(documents, 1)
            ]
            
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Tally results
    processed = 0
    failed = 0
    total_qa_pairs = 0
    
    for result in task_results:
        if isinstance(result, Exception):
            print(f"Task exception: {result}", file=sys.stderr)
            failed += 1
        else:
            qa_count, success = result
            total_qa_pairs += qa_count
            if success:
                processed += 1
            else:
                failed += 1
    
    return processed, failed, total_qa_pairs


def main():
    parser = argparse.ArgumentParser(
        description='Generate Q&A pairs from documents using VLLM',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'input_folder',
        help='Folder containing documents to process'
    )
    parser.add_argument(
        '-o', '--output',
        default='output.txt',
        help='Output file path'
    )
    parser.add_argument(
        '-p', '--prompt-file',
        default='qa_pairs_prompt.txt',
        help='Path to the prompt template file'
    )
    parser.add_argument(
        '--vllm-url',
        default='http://localhost:8000',
        help='VLLM server URL'
    )
    parser.add_argument(
        '-m', '--model',
        default=None,
        help='Model name (auto-detected if not specified)'
    )
    parser.add_argument(
        '--n-max',
        type=int,
        default=100,
        help='Maximum total conversations per document'
    )
    parser.add_argument(
        '--n-per-category-max',
        type=int,
        default=20,
        help='Maximum conversations per category'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=16384,
        help='Maximum tokens in LLM response'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='Maximum retries per document if no valid Q&A pairs are generated'
    )
    parser.add_argument(
        '-c', '--concurrency',
        type=int,
        default=20,
        help='Number of documents to process in parallel'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate input folder
    if not os.path.isdir(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Load prompt template
    if not os.path.isfile(args.prompt_file):
        print(f"Error: Prompt file '{args.prompt_file}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    prompt_template = load_prompt_template(args.prompt_file)
    
    # Auto-detect model if not specified
    model = args.model
    if not model:
        models = get_available_models(args.vllm_url)
        if models:
            model = models[0]
            print(f"Auto-detected model: {model}")
        else:
            print("Error: Could not auto-detect model. Please specify with -m/--model", file=sys.stderr)
            sys.exit(1)
    
    # Find documents
    extensions = ('.md', '.txt', '.text')
    documents = find_documents(args.input_folder, extensions)
    
    if not documents:
        print(f"No documents found with extensions {extensions} in '{args.input_folder}'")
        sys.exit(0)
    
    print(f"Found {len(documents)} documents to process")
    print(f"Concurrency: {args.concurrency}")
    
    # Process documents in parallel
    processed, failed, total_qa_pairs = asyncio.run(
        process_documents_parallel(documents, prompt_template, args, model, args.output)
    )
    
    # Summary
    print("\n" + "="*50)
    print("Summary:")
    print(f"  Documents processed: {processed}")
    print(f"  Documents failed: {failed}")
    print(f"  Total Q&A conversations: {total_qa_pairs}")
    print(f"  Output written to: {args.output}")


if __name__ == '__main__':
    main()
```

</details>

Here is the prompt file `qa_pairs_prompt.txt` that I wrote with GPT 5.2 Thinking and which tells the teacher model how to extract the pairs:

```text
You are a synthetic data generator for supervised fine-tuning of a Cybersource API integration assistant.

CONTEXT
You are generating training data for an LLM that will help software developers integrate with Cybersource payment APIs. The trained model should be able to:
- Explain Cybersource concepts, endpoints, fields, and flows
- Provide working code examples (cURL, JSON request bodies, SDK snippets)
- Guide developers through integration steps
- Troubleshoot common errors and edge cases

GOAL
Generate chat-style Q&A training conversations grounded ONLY in the provided Cybersource documentation. The dataset should teach:
1) Basic factual knowledge (coverage-first)
2) Practical implementation with code/JSON examples
3) Multi-step reasoning and troubleshooting

OUTPUT (JSON ONLY)
Return a JSON array of conversations. Each conversation is a JSON array of messages:

[
  [
    { "role": "user", "thinking": null, "content": "..." },
    { "role": "assistant", "thinking": "...", "content": "..." }
  ],
  ...
]

For multi-turn conversations:
[
  [
    { "role": "user", "thinking": null, "content": "..." },
    { "role": "assistant", "thinking": "...", "content": "..." },
    { "role": "user", "thinking": null, "content": "..." },
    { "role": "assistant", "thinking": "...", "content": "..." }
  ]
]

LANGUAGE
English only.

GROUNDING (STRICT)
- Use ONLY information present in the document.
- Do NOT use external knowledge, assumptions, or speculation.
- If the document does not contain enough information to confidently answer a question, do NOT generate that Q&A.
- Prefer fewer, higher-quality examples over guessing.
- If the whole document is too sparse or irrelevant, return [].

SELF-CONTAINED QUESTIONS (CRITICAL)
- Every user question must stand alone without referring to "this document", "shown here", "above/below", "the text", etc.
- Always include "Cybersource" context in questions when relevant (e.g., "In the Cybersource Payments API...", "When using Cybersource Token Management Service...", "For Cybersource REST API authentication...")
- Include specific context such as:
  - The Cybersource product/service name (Payments, TMS, Webhooks, Decision Manager, Unified Checkout, etc.)
  - The endpoint path and HTTP method if applicable
  - Field names, header names, or parameter names
  - Error codes or status values being discussed

CATEGORY RULE (CRITICAL)
Only generate Q&A for categories that the document actually contains enough information to support.
- If a category is not supported, generate 0 examples for it.
- Output may be empty: [].

POSSIBLE CATEGORIES (GENERATE ONLY IF SUPPORTED)

A) BASIC FACTS (definitions + surface facts)
   - Definitions of Cybersource terms, meaning of fields, what a parameter represents
   - Product/service overviews and capabilities
   Example question styles:
   - "What is Cybersource [product/feature]?"
   - "In the Cybersource REST API, what does the [field] field represent?"

B) API SHAPE (endpoints, methods, URLs, request/response structure)
   - Endpoint + method + full URL (e.g., POST https://apitest.cybersource.com/pts/v2/payments)
   - Required vs optional request fields
   - Response structure and fields
   Example question styles:
   - "What is the Cybersource API endpoint for [action]?"
   - "What HTTP method does Cybersource use for [operation]?"
   - "What are the required fields for a Cybersource [operation] request?"

C) CODE & REQUEST EXAMPLES (CRITICAL FOR DEVELOPER TRAINING)
   - Generate Q&A that includes actual JSON request/response bodies from the document
   - Include cURL commands if present in the document
   - Show complete, working examples that developers can adapt
   Example question styles:
   - "Show me a sample JSON request body for a Cybersource [operation]"
   - "What does a Cybersource [operation] API response look like?"
   - "How do I structure a Cybersource [operation] request with [specific fields]?"
   
   IMPORTANT: When the document contains JSON examples, include them in your answers formatted as code blocks.

D) DATA MODELS / SCHEMAS
   - Field constraints, enums, nesting, types, validation rules
   - Object structures (e.g., orderInformation, paymentInformation, processingInformation)
   Example question styles:
   - "What fields are nested under [object] in Cybersource [API]?"
   - "What are the allowed values for [field] in Cybersource?"

E) AUTH & SECURITY
   - HTTP Signature authentication
   - JWT authentication
   - API key types (shared secret, P12 certificates)
   - Required headers (v-c-merchant-id, Date, Digest, Signature)
   - Digital signature keys for webhooks
   Example question styles:
   - "How do I authenticate requests to the Cybersource REST API?"
   - "What headers are required for Cybersource API authentication?"
   - "How do I generate the Signature header for Cybersource?"

F) ERROR HANDLING
   - HTTP status codes (201, 400, 502)
   - Error reasons and status values (AUTHORIZED, DECLINED, INVALID_REQUEST, etc.)
   - Retry guidance
   Example question styles:
   - "What does HTTP status [code] mean in Cybersource API responses?"
   - "How do I handle a Cybersource [error type] error?"
   - "What should I do when Cybersource returns [status/reason]?"

G) PROCEDURES / HOW-TO FLOWS
   - Step-by-step integration flows
   - Setup procedures (sandbox creation, key generation)
   - Multi-step processes (authorization → capture → settlement)
   Example question styles:
   - "How do I [accomplish task] with Cybersource?"
   - "What are the steps to integrate Cybersource [product]?"
   - "How do I set up [feature] in Cybersource?"

H) WEBHOOKS / EVENTS
   - Event types (Network Token Events, Invoicing, Fraud Management, Recurring Billing, etc.)
   - Webhook payload structure
   - Digital signature validation
   - Subscription management
   Example question styles:
   - "What webhook events does Cybersource support for [product]?"
   - "How do I validate a Cybersource webhook notification?"
   - "What fields are included in a Cybersource [event type] webhook payload?"

I) EDGE CASES / CONSTRAINTS
   - Limits, timeouts, special cases
   - Processor-specific behaviors
   - Sandbox vs production differences (apitest.cybersource.com vs api.cybersource.com)
   Example question styles:
   - "What are the limits for Cybersource [feature]?"
   - "What constraints apply to [field/operation] in Cybersource?"

J) COMPLEX REASONING / TROUBLESHOOTING
   - Multi-step questions combining 2+ facts from the document
   - Choosing the right endpoint + required fields + interpreting errors
   - Integration decision-making
   Example question styles:
   - "I'm getting [error] when calling Cybersource [endpoint]. What could be wrong?"
   - "When should I use [option A] vs [option B] in Cybersource?"
   - "How do I combine [feature A] with [feature B] in Cybersource?"

MULTI-TURN CONVERSATIONS (ENCOURAGED)
Generate some 2-4 turn conversations that mirror real developer interactions:
- Turn 1: Conceptual question ("What is Cybersource TMS?")
- Turn 2: Implementation question ("How do I create a payment instrument token?")
- Turn 3: Code request ("Show me the request JSON")
- Turn 4: Follow-up detail ("What if I need to include billing address?")

HOW MANY TO GENERATE
- Target up to {N_MAX} total conversations for this document.
- For each supported category, generate between 1 and {N_PER_CATEGORY_MAX} examples.
- Prioritize categories C (Code Examples), B (API Shape), and G (Procedures) for developer training.
- It is acceptable to generate 0 examples for many categories if the document is limited.

COVERAGE & DEDUPLICATION
- Prioritize breadth: cover as many distinct endpoints, fields, headers, objects, and flows as the document provides.
- Avoid duplicates: do not ask the same fact more than twice unless the document is very short.
- Avoid vague questions; be specific and grounded.

ASSISTANT "thinking" FIELD RULES
- Purpose: a short, instructional reasoning outline (NOT private deliberation).
- BASIC / API SHAPE / SCHEMAS: 1–3 short steps.
- CODE EXAMPLES / PROCEDURES / ERRORS / COMPLEX: 3–7 short steps.
- Do NOT include meta commentary (no "I'm an AI", no tool/browsing talk, no policy references).

ASSISTANT "content" FIELD RULES
- Concise, precise, and grounded in the document.
- Use Cybersource terminology exactly as it appears in the document.
- Do not invent endpoints, fields, headers, constraints, or behaviors not present in the document.
- Format JSON and code examples properly using markdown code blocks with syntax highlighting.
- When providing JSON examples, ensure they are valid JSON.
- Always mention "Cybersource" when it helps clarify context.

VALIDATION CHECKLIST (DO THIS BEFORE OUTPUT)
1) Identify which categories A–J are supported by the document.
2) If none are supported, return [].
3) For each Q&A:
   - The question is self-contained and mentions Cybersource where appropriate.
   - The answer is fully supported by the document.
   - No invented facts.
   - JSON/code examples are properly formatted if included.
4) Ensure diversity and coverage; remove near-duplicates.

EXAMPLE OUTPUT (ILLUSTRATIVE ONLY; ADAPT TO YOUR DOCUMENT'S ACTUAL CONTENT)

[
  [
    { "role": "user", "thinking": null, "content": "What is the Cybersource REST API endpoint for processing a payment authorization?" },
    { "role": "assistant", "thinking": "1) Identify the payments endpoint from the doc.\n2) Note the HTTP method.\n3) Provide the full URL.", "content": "The Cybersource endpoint for processing a payment authorization is:\n\n**POST** `https://apitest.cybersource.com/pts/v2/payments`\n\nFor production, replace `apitest` with `api`." }
  ],
  [
    { "role": "user", "thinking": null, "content": "What products and events can I subscribe to for Cybersource webhooks?" },
    { "role": "assistant", "thinking": "1) List the webhook-enabled products.\n2) Note that each product has specific event types.", "content": "Cybersource webhooks support notifications for several products:\n\n- **Network Token Events** - Token lifecycle updates\n- **Invoicing** - Invoice creation, sending, payment events\n- **Fraud Management Essentials and Decision Manager** - Fraud screening results\n- **Recurring Billing** - Subscription and payment plan events\n\nYou can discover the specific event types available for your account using the GET `/notification-subscriptions/v1/products/{organizationId}` endpoint." }
  ],
  [
    { "role": "user", "thinking": null, "content": "Show me an example Cybersource webhook payload for an invoicing event." },
    { "role": "assistant", "thinking": "1) Find the webhook payload example in the doc.\n2) Identify the event type.\n3) Format the JSON properly.", "content": "Here's an example Cybersource webhook payload for the `invoicing.customer.invoice.send` event:\n\n```json\n{\n  \"eventType\": \"invoicing.customer.invoice.send\",\n  \"webhookId\": \"0fa698ae-7419-78c4-e063-9c588e0a497e\",\n  \"productId\": \"customerInvoicing\",\n  \"organizationId\": \"your_org_id\",\n  \"eventDate\": \"2024-07-10T21:11:38\",\n  \"retryNumber\": 0,\n  \"payload\": {\n    \"data\": {\n      \"invoiceNumber\": \"2\",\n      \"invoiceBalance\": \"1.00\",\n      \"currency\": \"USD\",\n      \"dueDate\": \"2024-07-10\",\n      \"payerName\": \"Jane Doe\",\n      \"emailTo\": \"customer@example.com\"\n    },\n    \"organizationId\": \"your_org_id\"\n  }\n}\n```\n\nKey fields include `eventType` to identify the event, `payload.data` containing invoice details, and `retryNumber` indicating delivery attempts." }
  ],
  [
    { "role": "user", "thinking": null, "content": "How do I create a Cybersource sandbox account for testing?" },
    { "role": "assistant", "thinking": "1) Outline the signup steps.\n2) Emphasize saving the keys.\n3) Mention the confirmation email.", "content": "To create a Cybersource sandbox account:\n\n1. Navigate to the sandbox creation page at developer.cybersource.com\n2. Complete the registration form with your details\n3. Click **Submit** to create your account\n4. **Important**: Save the API keys displayed after signup—you'll need the Organization ID, Key, and Shared Secret for authentication\n5. Check your email for login instructions (check spam if not received within 2 minutes)\n\nOnce registered, access your account through the CyberSource Business Center." },
    { "role": "user", "thinking": null, "content": "After creating the sandbox, how do I authenticate my Cybersource API requests?" },
    { "role": "assistant", "thinking": "1) Explain where to enter credentials.\n2) List the three credential values.\n3) Mention the alternative key generation method.", "content": "To authenticate Cybersource API requests with your sandbox credentials:\n\n1. Go to the Developer Center API Reference Assets\n2. Navigate to **APIs and Endpoints**\n3. Enter your credentials:\n   - **Organization ID**\n   - **Key** (API Key)\n   - **Shared Secret Key**\n4. Click **Update Credentials**\n\nAlternatively, generate new keys through the Business Center Key Management module.\n\nThese credentials are used to construct the HTTP Signature authentication headers required for Cybersource REST API calls." }
  ],
  [
    { "role": "user", "thinking": null, "content": "What does HTTP status code 201 mean in Cybersource API responses?" },
    { "role": "assistant", "thinking": "1) Explain what 201 indicates.\n2) Note that 201 can still mean declined.\n3) Clarify the transaction resource was created.", "content": "HTTP status code **201** in Cybersource API responses means that a transaction resource ID was successfully created.\n\n**Important**: A 201 status does not necessarily mean the transaction was approved. You may receive 201 for transactions that are:\n- Under review\n- Declined\n- Failed\n\nAlways check the `status` field in the response body (e.g., `AUTHORIZED`, `DECLINED`, `PENDING_REVIEW`) to determine the actual transaction outcome." }
  ]
]

DOCUMENT
{text_of_full_document}
```

I then ran it like so:

```python
uv venv --python 3.12 --seed
source .venv/bin/activate
python3 qa_generator.py cybersource_docs -o output.txt -v
```

This resulted in about 2K+ Q&A pairs, but I actually re-ran this script multiple times until I got 11K+ items. The reason for that is two-fold: 1) More samples yield better training and 2) Since LLMs are non-deterministic, every time I ran this script the resulting pairs were slightly different (different wording).

<img width="1000" height="548" alt="image" src="https://github.com/user-attachments/assets/47c773ef-d91c-4e6b-af3f-0ba2ce86b213" />

Here is one sample row:

```json
[
  {
    "role": "user",
    "thinking": null,
    "content": "What payment card information is required for Cybersource Visa Data Only?"
  },
  {
    "role": "assistant",
    "thinking": "1) List the required card fields\n2) Note the data types and format requirements\n3) Include example values from the document\n4) Mention card type detection in responses",
    "content": "The required `paymentInformation.card` fields for Cybersource Visa Data Only requests are:\n\n**Card Details:**\n- `expirationMonth`: Two-digit month (e.g., `\"05\"`)\n- `expirationYear`: Four-digit year (e.g., `\"2029\"`)\n- `number`: Full card number (masked in examples as `\"4XXXXXXXXXXX2X24\"`)\n\n**Example:**\n```json\n\"paymentInformation\": {\n    \"card\": {\n        \"expirationMonth\": \"05\",\n        \"expirationYear\": \"2029\",\n        \"number\": \"4XXXXXXXXXXX2X24\"\n    }\n}\n```\n\n**Response includes additional card information:**\n- `paymentInformation.card.bin`: Card BIN (e.g., `\"400000\"`)\n- `paymentInformation.card.type`: Card type (e.g., `\"VISA\"`)\n- `paymentAccountInformation.card.type`: Account type (e.g., `\"001\"`)\n\nThe card number should be the full PAN; in examples it's masked for security."
  }
]
```

## Fine-tuning gpt-oss-20b

Now for the main event. I used the most excellent [Unsloth fine-tuning library](https://unsloth.ai/) via their [Docker container](https://docs.unsloth.ai/get-started/install-and-update/docker) and [Jupyter notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks) to adjust the model. The code below is based on their [gpt-oss-20b fine-tuning notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-Fine-tuning.ipynb), with modifications I will explain which help with knowledge injection.

First, let's load the original model:

```python
from unsloth import FastLanguageModel
import torch
max_seq_length = 131072
dtype = None

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gpt-oss-20b",
    dtype = dtype, # None for auto detection
    max_seq_length = 131072, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
)
```

Then add LoRA adapter for parameter efficient fine-tuning:

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 256,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = True,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
```

It took me about three tries to get reasonable training results. Above I want to point out a few changes from the original notebook that Gemini 3.0 Pro and I came up with:

* `r = 128`: The [original notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-Fine-tuning.ipynb) has this `Rank` set to `8`, but that is too low for facts. Instead, I raised the number to `128`, which leans more towards the knowledge injection side. By increasing the rank, we are creating a larger "adapter" matrix. This gives the gradient descent process more room to encode the specific relationships and facts found in our 11k+ Q&A pairs.
* `lora_alpha = 256`: I increased `lora_alpha` from `16` to `256`, which is 2x the value of `r`. I did this to maintain the strength of the model updates. In LoRA, the "intensity" of the learning signal is determined by the ratio between Alpha and Rank. If we increase the Rank (r) but forget to increase Alpha (lora_alpha), the training essentially slows to a crawl.
* `use_rslora = True`: When we increase `r`, standard LoRA scales the gradients in a way that can sometimes be unstable. RSLoRA (Rank-Stabilized LoRA) fixes the scaling factor mathematically. It is generally recommended when `r` is large.

Next, we read the Q&A pairs and format them according to the [OpenAI Harmony format](https://cookbook.openai.com/articles/openai-harmony):

```python
def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

import json
from datasets import Dataset

# Load dataset - set thinking to empty string instead of removing it
data = []
with open("/workspace/work/cybersource/output.txt", "r") as f:
    for line in f:
        line = line.strip()
        if line:
            messages = json.loads(line)
            cleaned_messages = []
            for msg in messages:
                cleaned_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                    "thinking": msg.get("thinking") or ""  # Empty string if None/null
                })
            data.append({"messages": cleaned_messages})

dataset = Dataset.from_list(data)

print("Sample:")
print(dataset[0]["messages"])

# Apply formatting
def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)
```

Now the dataset is properly formatted using Harmony:

```python
print(dataset[0]['text'])
```

```text
<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-12-22

Reasoning: medium

# Valid channels: analysis, commentary, final. Channel must be included for every message.
Calls to these tools must go to the commentary channel: 'functions'.<|end|><|start|>user<|message|>What is the data type and length constraint for the mobilePhone field in Cybersource buyerInformation?<|end|><|start|>assistant<|channel|>analysis<|message|>1) Find the specifications section in the document.
2) Identify the data type and length values.
3) Present the information clearly.<|end|><|start|>assistant<|channel|>final<|message|>According to the Cybersource field specifications:

- **Data Type:** Integer
- **Data Length:** 25 characters maximum

This applies to the `buyerInformation.mobilePhone` field in the Cybersource REST API.<|return|>
```

Next, we configure the training run:

```python
from trl import SFTConfig, SFTTrainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        per_device_train_batch_size = 32,
        gradient_accumulation_steps = 1,
        dataloader_num_workers = 8,       # Pre-loads data so GPU doesn't wait on CPU
        warmup_ratio = 0.1,
        num_train_epochs = 3, # Set this for 1 full training run.
        learning_rate = 2e-4,
        logging_steps = 5,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use TrackIO/WandB etc
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(), # Enable BF16 (Blackwell loves BF16)
    ),
)
```

Here again I want to point out a few changes from the original notebook:

* `per_device_train_batch_size = 32`: Raised this from `1` to `32`. High physical batch size to eat up VRAM/Compute. I can do this because I am using a GPU with 96 GB VRAM.
* `gradient_accumulation_steps = 1`: Decreased this from `4` to `1`. We want an Effective Batch Size (EBS) of roughly 32 to 64. This is the "Goldilocks zone" for fine-tuning: noisy enough to learn well, stable enough to converge. So we wamt `per_device_train_batch_size` multiplied by `gradient_accumulation_steps` to be `32`.
* `dataloader_num_workers = 8`: Pre-loads data so GPU doesn't wait on CPU. Blackwell is fast; give it data faster!
* `warmup_ratio = 0.1`: Uses 10% of total steps (safer than fixed steps). The original notebook had `warmup_steps = 5`
* `num_train_epochs = 3`: Pretty standard number of epochs to train for
* `logging_steps = 5`: Log training loss every 5 steps
* `fp16 = not torch.cuda.is_bf16_supported()` and `bf16 = torch.cuda.is_bf16_supported()`: Enabled BF16 (Blackwell loves BF16)

In the original notebook there was a `train_on_completions` method to only train on the assistant outputs and ignore the loss on the user's inputs. This presumably helps increase accuracy of finetunes and lower loss as well. However, the resulting model when using it had lower quality, although I did not dig too much into it. I decided to skip it, so I train on the entire Chain-Of-Thought, as well as user input.

Now let's run the training!

```python
trainer_stats = trainer.train()
```

<img width="1000" height="617" alt="image" src="https://github.com/user-attachments/assets/187d9f0f-5c7f-4565-b4f2-aea2db9d9baa" />

The process took 1,056 training steps. Here are some stats for the run which took place on a single RTX Pro 6000:

```text
2786.3157 seconds used for training.
46.44 minutes used for training.
Peak reserved memory = 67.488 GB.
Peak reserved memory for training = 48.134 GB.
Peak reserved memory % of max memory = 71.063 %.
Peak reserved memory for training % of max memory = 50.684 %.
```

Finally, we can save the model locally:

```python
model.save_pretrained("finetuned_lora_model")
```

And we can also upload it to HuggingFace in [mxfp4](https://huggingface.co/blog/RakshitAralimatti/learn-ai-with-me) format:

```python
model.push_to_hub_merged("zmarty/gpt-oss-20b-cybersource2", tokenizer, token = "hf_...", save_method = "mxfp4")
```

## Example

For the question below the original gpt-oss-20b model guesses and hallucinates an answer, while our fine-tuned version correctly outputs the response. Note some of the internal monologue of the original model: ***I'm not sure of exact endpoint path.*** ... ***But we need to provide a precise answer.*** ... ***The docs might say*** ...

`User question:` What is the Cybersource API endpoint for retrieving batch submission detail reports?

### <u>INCORRECT</u> answer from the unmodified gpt-oss-20b model

<details markdown="1">
<summary markdown="span">Reasoning (click to expand)</summary>
We need to answer: "What is the Cybersource API endpoint for retrieving batch submission detail reports?" Likely this refers to the API used to get batch submission details (like batch details, maybe GET /batchDetails). I'm not sure of exact endpoint path. Let’s search memory: Cybersource has an API for "batch details" maybe /batchDetails? According to documentation, there is "Batch details" endpoint: GET /batchDetails?batchId=... Or maybe /batchDetails?batchId=... Actually, the "Batch Detail Report" is part of the "Batch Management API". The endpoint might be GET /reports/batchDetails. Let's recall. There's "Batch Detail Report" in the "Reports" API. The endpoint: GET /reports/batchDetail?batchId=... or /batchDetails?batchId=... Let’s search memory: I recall that Cybersource has a "Batch" API: POST /batches to create batch, GET /batches/{batchId} to get batch, GET /batches to list. But for "batch details", there is "Batch Details Report" that provides information about each transaction in a batch. The endpoint may be: GET /batchDetails?batchId=... Actually, reading docs: "Batch details report" can be retrieved via the "Batch Details" endpoint: GET /reports/batchDetails?batchId=... Also can specify fields like startTime, endTime, status, etc.

But we need to provide a precise answer. The question likely expects: The endpoint is: GET /reports/batchDetail?batchId=... Let's confirm: In Cybersource API documentation, there's an endpoint: GET /batchDetails?batchId=... Let's search memory: I saw something like "/batchDetails?batchId=..." used to retrieve details of a particular batch submission. There's also "Batch List" endpoint: GET /batchList?startTime=... etc. But the question is "retrieving batch submission detail reports". That suggests maybe "batchDetails" endpoint. So answer: GET https://api.cybersource.com/v1/batchDetails?batchId={id} or /reports/batchDetails. But better to be specific: The endpoint for retrieving batch submission detail reports is: GET /reports/batchDetails. Let's check typical Cybersource API base: https://api.cybersource.com/v1. The docs might say: GET /reports/batchDetails?batchId=... So I'd answer: Use the GET /reports/batchDetails endpoint, e.g., https://api.cybersource.com/v1/reports/batchDetails?batchId=<batchId>. If you need a specific endpoint, that's it. Also mention that you need to include query param "batchId" and maybe "pageIndex", "pageSize". So answer accordingly.

Thus: The Cybersource API endpoint is: GET /reports/batchDetails?batchId={batchId}. Provide example.``
</details>

`Answer:` In the Cybersource REST APIs the “Batch‑Detail Report” is exposed through the **Batch Details** endpoint.

To pull the report for a specific batch you call:

```markdown
GET https://api.cybersource.com/v1/reports/batchDetails?batchId={batchId}
```

* **Method** – GET
* **Base URL** – `https://api.cybersource.com/v1`
* **Endpoint** – `/reports/batchDetails`
* **Required Query Parameter** – `batchId` (the ID of the batch you submitted)  \n*
* **Optional Parameters** – pagination (`pageIndex`, `pageSize`), filtering (`status`, `startTime`, `endTime`), etc., depending on your needs.

You authenticate the request with your standard Cybersource OAuth 2.0 or JWT credentials, then parse the JSON response which contains the transaction‑level detail for that batch.

### <u>CORRECT</u> answer from our fine-tuned model gpt-oss-20b model-cybersource2

<details markdown="1">
<summary markdown="span">Reasoning: (click to expand)</summary>

1) Identify the endpoint path
2) Note the HTTP method
3) Provide the complete URL structure
4) Mention the required parameters

</details>

`Answer:` 

The Cybersource API endpoint for retrieving batch submission detail reports is:

**GET** `https://<url_prefix>/pts/v1/transaction-batch-details/{id}`

Where `<url_prefix>` is the server hostname specified in your API call (e.g., `apitest.cybersource.com` or `api.cybersource.com`), and `{id}` is the unique ID assigned to your batch upload file.

This endpoint retrieves the detailed processing status report for your specific batch upload file.

### Answer from training data

<details markdown="1">
<summary markdown="span">Reasoning: (click to expand)</summary>

1) Identify the HTTP method.
2) Provide the endpoint structure.
3) Explain the URL components.

</details>

`Answer:` 

To retrieve a batch submission detail report, send an HTTP **GET** request to:

```
https://<url_prefix>/pts/v1/transaction-batch-details/{id}
```

Where:
- `<url_prefix>` is the server domain (see supported values below)
- `{id}` is the unique ID assigned to your batch upload file

## What's Next?

Next, I plan to further evaluate the fine-tuned model against the original. I am especially interested in determining if there is any degradation in the capabilities of the model. I plan to run the typical barrage of LLM tests in [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). Furthermore, I am interested in trying out [continued pretraining](https://docs.unsloth.ai/basics/continued-pretraining), which while heavier than fine-tuning should in theory allow feeding the model raw documents. 