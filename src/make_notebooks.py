"""Colab 用に、コードセルのある原稿を book/_build/html/_sources/ へ .ipynb として書き出す。"""
import shutil
from pathlib import Path

import jupytext

BOOK = Path(__file__).resolve().parent.parent / "book"
OUT = BOOK / "_build/html/_sources"

for src in sorted(BOOK.rglob("*")):
    if "_build" in src.parts or src.suffix not in (".md", ".ipynb"):
        continue
    dest = OUT / src.relative_to(BOOK).with_suffix(".ipynb")
    if src.suffix == ".md" and "```{code-cell}" not in src.read_text():
        continue
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".md":
        jupytext.write(jupytext.read(src), dest)
    else:
        shutil.copy(src, dest)
    print(dest.relative_to(OUT))
