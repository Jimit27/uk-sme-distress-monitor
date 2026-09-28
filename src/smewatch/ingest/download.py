"""Resumable streaming downloads from the Companies House bulk data site."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)

CHUNK = 1 << 20  # 1 MiB
HEADERS = {"User-Agent": "smewatch/0.1 (+https://github.com/Jimit27/uk-sme-distress-monitor)"}


def download(url: str, dest: Path, retries: int = 5, timeout: int = 60) -> Path:
    """Download ``url`` to ``dest``, resuming a partial file if one exists.

    Returns the destination path. Skips the download when the file is already
    complete (server size matches local size).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    try:
        head = requests.head(url, timeout=timeout, allow_redirects=True, headers=HEADERS)
        if head.status_code == 404:
            head.raise_for_status()
        total = int(head.headers.get("Content-Length", 0)) if head.ok else 0
    except requests.ConnectionError:
        total = 0
    if dest.exists() and total and dest.stat().st_size == total:
        log.info("already downloaded: %s", dest.name)
        return dest

    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = dict(HEADERS, Range=f"bytes={have}-") if have else dict(HEADERS)
        try:
            with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
                if r.status_code == 416:  # already complete
                    break
                if r.status_code == 404:
                    r.raise_for_status()  # not retryable
                r.raise_for_status()
                mode = "ab" if have and r.status_code == 206 else "wb"
                with open(part, mode) as fh:
                    for chunk in r.iter_content(CHUNK):
                        fh.write(chunk)
            if not total or part.stat().st_size >= total:
                break
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                raise
            wait = 5 * attempt
            log.warning("download error (%s), retry %d/%d in %ds", exc, attempt, retries, wait)
            time.sleep(wait)
        except requests.RequestException as exc:
            wait = 5 * attempt
            log.warning("download error (%s), retry %d/%d in %ds", exc, attempt, retries, wait)
            time.sleep(wait)
    else:
        raise RuntimeError(f"failed to download {url}")

    part.replace(dest)
    log.info("downloaded %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest
