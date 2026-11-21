"""Download the files of a Zenodo record, including a restricted one, with the owner's personal access token.

    python zenodo_data.py 17102801 [target_folder]        (prints what the record holds and downloads it)

The token (Zenodo -> Account -> Applications -> Personal access tokens; give it the narrowest scope the page offers,
it only has to let the account read the record) is read from the env var
ZENODO_TOKEN, or from the file /content/zenodo_token that the pipeline uploads to a Colab session.  It must belong
to an account that may see the record (a creator, or someone the creators approved).  Public records need no token.

Archives (.zip, .tar, .tar.gz) are unpacked next to the download.  Only the standard library is used, so it runs
unchanged on Colab, the Jetson and the Mac.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

API = "https://zenodo.org/api/records"
IMAGES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def token() -> str:
    t = os.environ.get("ZENODO_TOKEN", "").strip()
    if not t and Path("/content/zenodo_token").is_file():
        t = Path("/content/zenodo_token").read_text().strip()
    return t


def _get(url: str, tok: str):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"} if tok else {})
    return urllib.request.urlopen(req, timeout=120)


def record(record_id: str, tok: str = "") -> dict:
    """The record's metadata and file list; says plainly why a restricted record cannot be read."""
    try:
        with _get(f"{API}/{record_id}", tok) as r:
            data = json.load(r)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise PermissionError(f"Zenodo refused the token for record {record_id} (HTTP {exc.code}): it must belong to an "
                                  "account that may see the record (and a scope that allows reading it)") from exc
        raise
    files = data.get("files") or []
    if not files and data.get("metadata", {}).get("access_right") == "restricted":
        raise PermissionError(f"record {record_id} is restricted and shows no files"
                              + ("" if tok else ": set ZENODO_TOKEN") + (": this token does not give access to them" if tok else ""))
    return data


def download(record_id: str, target: str | Path, tok: str | None = None) -> list[Path]:
    """Download every file of the record into `target`, verify its checksum, unpack archives; returns the files."""
    tok = token() if tok is None else tok
    data = record(record_id, tok)
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for f in data["files"]:
        dest = target / f["key"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = (f.get("links") or {}).get("self") or f"{API}/{record_id}/files/{f['key']}/content"
        with _get(url, tok) as r, dest.open("wb") as fh:
            shutil.copyfileobj(r, fh)
        algo, _, expected = (f.get("checksum") or "").partition(":")
        if algo == "md5" and expected and hashlib.md5(dest.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"checksum mismatch for {f['key']}")
        out.append(dest)
        _unpack(dest)
    return out


def _unpack(path: Path) -> None:
    name = path.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            z.extractall(path.with_suffix(""))
    elif name.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(path) as t:
            safe = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}  # refuse paths that escape the folder
            t.extractall(path.parent / path.name.split(".")[0], **safe)


def describe(folder: str | Path) -> dict:
    """What an unpacked record holds: image counts per sub-folder, to see whether it is an ImageFolder tree."""
    counts: dict[str, int] = {}
    for p in Path(folder).rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGES:
            key = str(p.parent.relative_to(folder))
            counts[key] = counts.get(key, 0) + 1
    return counts


if __name__ == "__main__":
    rid = sys.argv[1] if len(sys.argv) > 1 else "17102801"
    dest = Path(sys.argv[2] if len(sys.argv) > 2 else f"zenodo_{rid}")
    try:
        meta = record(rid, token())["metadata"]
        print(f"{meta.get('title')}  ({meta.get('access_right')}, {(meta.get('license') or {}).get('id')})")
        files = download(rid, dest)
    except PermissionError as exc:
        sys.exit(f"cannot read the files: {exc}")
    print("files:", [(p.name, round(p.stat().st_size / 1e6, 2)) for p in files])
    print("images per folder:", describe(dest))
