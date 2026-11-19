# Pythonで学ぶ！データサイエンス講座

統計の基礎から機械学習、テキスト分析、グラフ、強化学習までを、Pythonのコードを動かしながら学びます。

**公開サイト: https://matt76k.github.io/ds-seminar/**

## 内容

| 章 | タイトル | 章 | タイトル |
|---|---|---|---|
| 1 | 記述統計 | 9 | サポートベクトルマシン |
| 2 | データ処理と可視化 | 10 | 次元削減 |
| 3 | 確率 | 11 | テキストデータの分析 |
| 4 | 推計統計学 | 12 | テキストマイニングの応用 |
| 5 | 回帰分析 | 13 | グラフ理論の基礎 |
| 6 | 決定木 | 14 | グラフの応用 |
| 7 | クラスタリング | 15 | 強化学習 |
| 8 | ガウス混合モデル | | |

## コードの実行

各ページ上部の「Open in Colab」ボタンから、そのページをGoogle Colabでノートブックとして開けます。
必要なライブラリのインストールとデータの取得は、ノートブック先頭の隠しセルで自動的に行われます。

## ローカルでのビルド

[uv](https://docs.astral.sh/uv/) と Node.js（Jupyter Book 2 が使用）が必要です。

```sh
uv sync
cd book
uv run jupyter book build --html --execute --strict   # 出力: book/_build/html
```

- セルの実行結果は `book/_build/execute` にキャッシュされ、変更したページだけが再実行されます。
- `--strict` により、セルの例外や警告があるとビルドが失敗します。
- 描画の段階でまれに `ECONNRESET` で失敗することがあります。その場合は再実行してください。

## EPUB の作成

Jupyter Book 2（mystmd）は EPUB を直接出力できないため、ビルド済みの HTML から本文を取り出して [pandoc](https://pandoc.org/) で epub3 に変換します。

```sh
(cd book && uv run jupyter book build --html --execute --strict)
uv run python src/build_epub.py   # 出力: book/exports/book.epub
```

- サイトの HTML を入力にするので、先に HTML をビルドしておきます（コードの実行結果や図もそのまま入ります）。
- 章の分割は `book/myst.yml` の目次順です。目次には h1（章・節）だけを載せています。
- 表紙は `book/logo.png` を使います。差し替えたいときは `src/build_epub.py` の `cover` を変更してください。
- スタイルは `src/epub.css` です。

## 構成

- `book/myst.yml`: 目次と設定。新しいページはここの `toc` に追加します。
- `book/chapterNN/`: 各章の原稿（MyST Markdown + jupytext）と、その章で使うデータ。
- `book/japanese-index.mjs`: 索引ページ（五十音順）を作るプラグイン。
- `src/make_notebooks.py`: Colab用に原稿を `.ipynb` に変換するスクリプト（CIで実行）。

## 公開

`main` へのpushとPRではビルドの検証だけを行います。
`v*` タグをpushすると、GitHub Actions がビルドして `gh-pages` ブランチに公開します。

```sh
git tag v0.0.15
git push origin v0.0.15
```

## データの出典

- `book/chapter03/Davis.csv`: R の `carData` パッケージ（GPL-2 以上）の `Davis` データ。詳細は [book/chapter03/README.md](book/chapter03/README.md)。
