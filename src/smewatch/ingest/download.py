"""Resumable streaming downloads from the Companies House bulk data site."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)

CHUNK = 1 << 20  # 1 MiB


def download(url: str, dest: Path, retries: int = 5, timeout: int = 60) -> Path:
    """Download ``url`` to ``dest``, resuming a partial file if one exists.

    Returns the destination path. Skips the download when the file is already
    complete (server size matches local size).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    head = requests.head(url, timeout=timeout, allow_redirects=True)
    head.raise_for_status()
    total = int(head.headers.get("Content-Length", 0))
    if dest.exists() and total and dest.stat().st_size == total:
        log.info("already downloaded: %s", dest.name)
        return dest

    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
                if r.status_code == 416:  # already complete
                    break
                r.raise_for_status()
                mode = "ab" if have and r.status_code == 206 else "wb"
                with open(part, mode) as fh:
                    for chunk in r.iter_content(CHUNK):
                        fh.write(chunk)
            if not total or part.stat().st_size >= total:
                break
        except requests.RequestException as exc:
            wait = 5 * attempt
            log.warning("download error (%s), retry %d/%d in %ds", exc, attempt, retries, wait)
            time.sleep(wait)
    else:
        raise RuntimeError(f"failed to download {url}")

    part.replace(dest)
    log.info("downloaded %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest
