"""ビルド済み HTML サイトから EPUB を生成する。

Jupyter Book 2 (mystmd) は EPUB を直接出力できないため、`jupyter book build --html`
で生成した HTML の本文だけを取り出し、pandoc で epub3 に変換する。

    (cd book && uv run jupyter book build --html --execute --strict)
    uv run python src/build_epub.py

出力: book/exports/book.epub
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from html import escape
from pathlib import Path

import yaml
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
BOOK = ROOT / "book"
SITE = BOOK / "_build" / "html"
OUT = BOOK / "exports" / "book.epub"
CSS = Path(__file__).resolve().parent / "epub.css"

# 本文以外の要素（ナビゲーション、コピーボタン、数式の二重表示など）を落とす。
DROP_SELECTORS = [
    "div.hidden",
    "nav",
    "button",
    "svg",
    "script",
    "style",
    "a.myst-fm-edit-link",
    "div.myst-fm-downloads-dropdown",
    "div.myst-fm-block-header",
    "div.myst-footer-links",
    "div.myst-jp-nb-block-spinner",
    "div.myst-jp-nb-block > div.sticky",
    "span.katex-html",
    "div.block.my-10",
]


def toc_files(node: list[dict]) -> list[str]:
    """myst.yml の project.toc を文書順のファイル一覧に平坦化する。"""
    files: list[str] = []
    for item in node or []:
        if "file" in item:
            files.append(item["file"])
        files.extend(toc_files(item.get("children", [])))
    return files


def html_path(source: str) -> Path:
    stem = Path(source).with_suffix("")
    path = SITE / stem / "index.html"
    if path.exists():
        return path
    # 先頭ページ（intro.md など）はサイトのルート index.html になる
    return SITE / stem.parent / "index.html"


def clean_article(html: str, prefix: str) -> BeautifulSoup | None:
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article")
    if article is None:
        return None
    for selector in DROP_SELECTORS:
        for el in article.select(selector):
            el.decompose()
    # コードブロックの行番号用 <code> を落とす
    for code in article.find_all("code"):
        if code.find(class_="react-syntax-highlighter-line-number"):
            code.decompose()
    # 見出しに付いている「¶」リンクを落とす
    for heading in article.find_all(re.compile(r"^h[1-6]$")):
        for a in heading.find_all("a"):
            a.decompose()
    # ページタイトルの h1 をトップレベルに出し、章分割できるようにする
    for el in article.select("div.myst-fm-block"):
        el.unwrap()
    # KaTeX の入れ子を外して数式だけ残す
    for cls in ("katex-mathml", "katex"):
        for el in article.select(f"span.{cls}"):
            el.unwrap()
    for el in article.select("span.katex-display"):
        el.name = "div"
    # pandoc の TeX パーサは \displaystyle を解釈できず数式全体を落とす。
    # 表示には MathML が使われるので annotation 側だけ整える。
    for annotation in article.find_all("annotation"):
        if annotation.get("encoding") == "application/x-tex" and annotation.string:
            annotation.string = annotation.string.replace("\\displaystyle", "").replace(
                "\\\\", " \\quad "
            )
    # id はページをまたいで重複するので、ページ接頭辞を付けて一意にする
    for el in article.find_all(id=True):
        el["id"] = prefix + el["id"]
    for link in article.find_all("a", href=True):
        href = link["href"]
        if href.startswith("#"):
            link["href"] = "#" + prefix + href[1:]
        elif href.startswith(("http://", "https://", "mailto:")):
            continue
        else:
            # サイト内の絶対パスは EPUB 内で解決できないのでリンクを外す
            link.unwrap()
    # 画像を実ファイルに解決する（pandoc に埋め込ませる）
    for img in article.find_all("img"):
        src = img.get("src", "")
        path = SITE / src.lstrip("/")
        if path.exists():
            img["src"] = str(path)
        if not img.get("alt"):
            img["alt"] = ""
    return article


def build_html() -> tuple[Path, dict]:
    config = yaml.safe_load((BOOK / "myst.yml").read_text())
    sources = toc_files(config["project"]["toc"])

    parts: list[str] = []
    missing: list[str] = []
    for index, source in enumerate(sources):
        path = html_path(source)
        if not path.exists():
            missing.append(source)
            continue
        article = clean_article(path.read_text(), f"pg{index:03d}-")
        if article is None:
            missing.append(source)
            continue
        # 見出しをトップレベルに置かないと pandoc が章分割できない
        parts.extend(str(child) for child in article.contents)

    if missing:
        sys.exit(
            "HTML が見つかりません: "
            + ", ".join(missing)
            + "\n(cd book && uv run jupyter book build --html --execute --strict) を先に実行してください。"
        )

    project = config["project"]
    title = escape(project.get("title", "book"))
    html = f"""<!DOCTYPE html>
<html lang="ja">
<head><meta charset="utf-8"><title>{title}</title></head>
<body>{"\n".join(parts)}</body>
</html>"""

    fd, name = tempfile.mkstemp(suffix=".html")
    with open(fd, "w") as handle:
        handle.write(html)
    return Path(name), project


def main() -> None:
    if not SITE.exists():
        sys.exit(f"{SITE} がありません。先に HTML をビルドしてください。")

    tmp_html, project = build_html()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    authors = [a["name"] for a in project.get("authors", []) if "name" in a]

    cmd = [
        "pandoc",
        str(tmp_html),
        "-f",
        "html",
        "-t",
        "epub3",
        "-o",
        str(OUT),
        "--toc",
        "--toc-depth=1",
        "--split-level=1",
        f"--resource-path={SITE}",
        f"--css={CSS}",
        "--metadata",
        f"title={project.get('title', 'book')}",
        "--metadata",
        "lang=ja",
    ]
    for author in authors:
        cmd += ["--metadata", f"author={author}"]
    cover = BOOK / "logo.png"
    if cover.exists():
        cmd += [f"--epub-cover-image={cover}"]

    try:
        subprocess.run(cmd, check=True)
    finally:
        tmp_html.unlink(missing_ok=True)
    size = OUT.stat().st_size / 1024
    print(f"✅ {OUT.relative_to(ROOT)} ({size:.0f} KB)")


if __name__ == "__main__":
    main()
