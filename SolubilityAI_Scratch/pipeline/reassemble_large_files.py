"""Rebuild files larger than the 20 MB transfer limit from their .partNN chunks.

Run once after copying this folder:  python reassemble_large_files.py
Each rebuilt file is checked against the SHA-256 in large_files_manifest.json.  The rebuilt
files are byte-identical to the originals (standard joblib / npz files); the parts can then be
deleted.
"""
import hashlib, json
from pathlib import Path
R = Path(__file__).resolve().parent
man = json.load(open(R / "large_files_manifest.json"))
for rel, info in man.items():
    out = R / rel
    parts = [R / p for p in info["parts"]]
    with open(out, "wb") as f:
        for p in parts:
            f.write(p.read_bytes())
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    print(("OK  " if h == info["sha256"] else "BAD ") + rel)
    assert h == info["sha256"], rel
