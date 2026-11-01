"""Download public pipe-inspection videos (and the files that come with them) from figshare, with an MD5 check.

usage: python -I scripts/download_data.py --list [DATASET]
       python -I scripts/download_data.py DATASET OUT_DIR [FILE ...]
  DATASET  sheffield-concrete5m  Sheffield Minicam, 300 mm concrete pipe, 5 m, 5 runs            (CC BY 4.0)
           sheffield-mix15m      Sheffield Minicam, 15 m mixed network, 15 runs                  (CC BY 4.0)
           sheffield-sewer       real sewers Pipe1-3 (Edwards et al. 2023) + checkerboard video   (CC BY-NC-SA 4.0,
                                 + per-frame ground-truth distances                                non-commercial only)
  FILE     file names inside the data set (default: all of them). Names, sizes and MD5 come from the figshare API
           at run time, so --list shows exactly what can be fetched.
A file already in OUT_DIR with the right MD5 is skipped; an incomplete one is resumed (HTTP range request).
Put every data set in a folder of its own (downloaded files are data; scripts stay in the repository).
"""
import hashlib
import json
import os
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

DATASETS = {   # key: (figshare article id, DOI, licence, what it is)
    "sheffield-concrete5m": (28046066, "10.15131/shef.data.28046066", "CC BY 4.0",
                             "Sheffield Minicam, 300 mm concrete pipe, 5 m, 5 inspection runs (720x576 PAL, 25 fps)"),
    "sheffield-mix15m": (28047074, "10.15131/shef.data.28047074", "CC BY 4.0",
                         "Sheffield Minicam, 15 m network (concrete, clay, plastic, 2 manholes), 15 runs"),
    "sheffield-sewer": (21198070, "10.15131/shef.data.21198070", "CC BY-NC-SA 4.0 (non-commercial)",
                        "real buried sewers Pipe1-3 (Edwards et al. 2023), checkerboard calibration video, "
                        "per-frame ground-truth distances"),
}
API = "https://api.figshare.com/v2/articles/{}"


def files_of(key):
    with urllib.request.urlopen(API.format(DATASETS[key][0]), timeout=60) as r:
        return json.load(r)["files"]


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url, out, size):
    """download url to out, resuming a partial file, with a progress line every ~5 s"""
    have = os.path.getsize(out) if os.path.exists(out) else 0
    if have > size:                              # longer than the file on the server: start again
        os.remove(out)
        have = 0
    if have == size:
        return
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    t0, last, got = time.time(), time.time(), have
    with urllib.request.urlopen(req, timeout=120) as r, open(out, "ab" if have else "wb") as f:
        if have and r.status != 206:             # the server ignored the range: rewrite from the start
            f.seek(0), f.truncate()
            got = 0
        for block in iter(lambda: r.read(1 << 20), b""):
            f.write(block)
            got += len(block)
            if time.time() - last > 5:
                rate = (got - have) / max(time.time() - t0, 1e-6) / 1e6
                print(f"    {got / 1e6:8.1f} / {size / 1e6:.1f} MB  ({rate:.1f} MB/s)")
                last = time.time()


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        sys.exit(__doc__)
    if args[0] == "--list":
        for key in args[1:] or DATASETS:
            art, doi, lic, what = DATASETS[key]
            print(f"\n{key}: {what}\n  DOI {doi}, licence {lic}")
            for f in files_of(key):
                print(f"  {f['size'] / 1e6:9.1f} MB  {f['computed_md5']}  {f['name']}")
        return
    key, out_dir, want = args[0], args[1], args[2:]
    if key not in DATASETS:
        sys.exit(f"unknown data set {key}; known: {', '.join(DATASETS)}")
    files = files_of(key)
    names = {f["name"]: f for f in files}
    missing = [n for n in want if n not in names]
    if missing:
        sys.exit(f"not in {key}: {missing}; available: {sorted(names)}")
    todo = [names[n] for n in want] if want else files
    os.makedirs(out_dir, exist_ok=True)
    art, doi, lic, what = DATASETS[key]
    print(f"{key}: {what}; DOI {doi}; licence {lic}")
    for f in todo:
        out = os.path.join(out_dir, f["name"])
        if os.path.exists(out) and os.path.getsize(out) == f["size"] and md5_of(out) == f["computed_md5"]:
            print(f"  ok (already there)  {f['name']}")
            continue
        print(f"  downloading {f['name']} ({f['size'] / 1e6:.1f} MB) from {f['download_url']}")
        fetch(f["download_url"], out, f["size"])
        got = md5_of(out)
        if got != f["computed_md5"]:
            sys.exit(f"MD5 mismatch for {out}: {got}, expected {f['computed_md5']} (delete the file and run again)")
        print(f"  ok, MD5 {got}  {f['name']}")
    json.dump(dict(dataset=key, doi=doi, licence=lic, article=art, files=todo),
              open(os.path.join(out_dir, "_download.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
