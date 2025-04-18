---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
推計統計学
==========
```{epigraph}
All models are wrong, some are useful.

-- George E. P. Box
```
これまで、得られたデータの特徴である平均や標準偏差などを求める方法について学びました。
これは記述統計の一部です。

しかし、実際の問題では、すべてのデータを取得するには時間やコスト面で困難な場合があります。
そこで、データの一部のみを取得し、そこから全体の特性を**推定**する方法が必要となります。
また、その推定が正しいかどうかを**検定**する方法も必要です。
これが推論統計学の基本的な考え方です。

推論統計学では、以下の重要な概念があります。

- **母集団と標本**
   - **母集団**：推測したい対象となるすべてのデータ
   - **標本**：母集団から取得した一部のデータ
   - **抽出**：母集団から標本を取り出す過程

- **確率変数と実現値**
   - 実際に観測されたデータ $x_1, x_2, \ldots, x_n$ は、$n$個の確率変数 $X_1, X_2, \ldots, X_n$ の実現値です。ここで、$n$ を標本のサイズ（大きさ）といいます。

- **母数**
   - 母集団の平均（**母平均**）や分散（**母分散**）といった、母集団の特性を表す定数を**母数**といいます。

- **統計的推論**
   - 選んだ標本が母集団の特性を適切に反映している場合、その標本から得られた結果を用いて母集団全体の特性を推測できます。このプロセスを統計的推論と呼びます。

-  **点推定と区間推定**
   - **点推定**：標本から計算された統計量を用いて、母数の値を1つの値で推定すること。
   - **区間推定**：母数が含まれると考えられる区間を推定すること。

-  **信頼区間**
   - 区間推定において、ある確率（**信頼度**）で母数を含む区間のこと。
   - 例えば、95%信頼区間は、同じ方法で多数回推定を行った場合、その95%が真の母数を含むことを意味します。

例えば、ある会社の全製品の平均寿命（母平均）を推定する場合を考えます。
100個の製品をランダムに選んで寿命を測定したとします。このとき、以下の点を考慮します。

- 100個という標本サイズが十分かどうか
- 選んだ標本が母集団を適切に代表しているか
- 標本平均を用いて母平均の点推定を行う
- 標本の情報を用いて母平均の95%信頼区間を計算する

例えば、標本平均が1000時間で、95%信頼区間が[980時間, 1020時間]と計算されたとします。
この結果は、真の平均寿命が980時間から1020時間の間にある確率が95%であると解釈できます。

推論統計学は、このような限られたデータから母集団全体について信頼性の高い結論を導き出すための手法を提供します。
点推定だけでなく区間推定を行うことで、推定の不確実性も考慮に入れた、より慎重な判断が可能になります。