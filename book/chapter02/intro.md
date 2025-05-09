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
データ処理と可視化
===============
```{epigraph}
百聞は一見に如かず

-- 漢書
```

データを分析する前に、以下の処理が必要になります。

1. 取得
   - データを分析するには、まず分析対象となるデータが必要です。
  
2. 保存
   - 取得したデータをどのように保存するかが重要です。
   - 独自の形式で保存すると、他の人が利用しにくく、再利用が困難になります。
   - そのため、データを保存するための一般的な形式がいくつか存在します。

3. データクリーニング
   - 生のデータには欠損値や分析に悪影響を与える外れ値などが存在します。これらの問題に対処するため、データの整形が必要です。

4. データ変換   
   - データを分析しやすくするために、正規化・標準化（スケーリング）やエンコーディング（カテゴリ変数の数値化）などの処理を行います。

データの準備ができたら、平均や分散などの基本統計量を確認し、そのデータの特徴を把握します。
しかし、特徴量の把握だけでなく、データから有意義な情報を引き出すには、データの可視化が非常に有効です。

本章では、データを処理していく過程を通じて、Pythonを使用したデータ処理と可視化の方法について学びます。