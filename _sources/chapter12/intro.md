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
テキストマイニングの応用
====================
SNSの投稿、ニュース記事、商品レビュー、メール、ウェブサイトのコンテンツなど、テキストデータは私たちの周りに溢れています。
これらの膨大なテキストデータから価値ある情報や洞察を引き出すことは、現代のビジネスや研究において極めて重要になっています。

テキストマイニングとは、自然言語処理（NLP）と機械学習の技術を用いて、
構造化されていないテキストデータから有用なパターンや情報を発見するプロセスです。
人間が読むことを前提に書かれたテキストを、コンピュータが分析可能な形に変換し、そこから意味のある知見を抽出します。

テキストマイニングの応用範囲は非常に広く、以下のような場面で活用されています：

- ビジネスインテリジェンス: 顧客の声や市場動向の分析
- ヘルスケア: 医療記録からの症例分析や疾病パターンの発見
- 学術研究: 大量の論文から研究トレンドの把握
- マーケティング: 消費者感情の分析やブランド評判のモニタリング
- 政治・社会分析: 世論調査や政策反応の分析
- セキュリティ: 不審なコミュニケーションパターンの検出

この章では、テキストマイニングの主要な応用技術として、テキストのベクトル化、感情分析、トピックモデリングを取り上げます。
これらの技術は、人間の言語という複雑なデータから、機械が理解・分析可能な形式への変換と、そこからの有益な情報抽出の基盤となるものです。

テキストのベクトル化では、言葉をどのように数値表現に変換するかを学びます。
感情分析では、テキストから書き手の感情や態度を自動的に検出する方法を探ります。
トピックモデリングでは、大量の文書から自動的に主題やテーマを抽出する技術を見ていきます。

## テキストのベクトル化

テキストデータは、そのままでは機械学習モデルに入力できません。
機械学習モデルは数値データを扱うため、テキストを数値表現（ベクトル）に変換する必要があります。
この過程は**テキストのベクトル化**または**特徴量抽出**と呼ばれます。

ここではBoW（Bag of Words）とTF-IDFについて説明します。

### Bag of Words (BoW)

BoWは最も基本的なテキスト表現方法で、単語の出現回数のみを考慮します。

```{code-cell}
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sudachipy import tokenizer, dictionary
from tabulate import tabulate 

texts = [
    "機械学習は人工知能の一種です。機械学習ではデータから学習します。",
    "深層学習は機械学習の一種で、ニューラルネットワークを使います。",
    "自然言語処理は機械学習を使ったテキスト分析技術です。",
]

def tokenize_text(text):
    tokenizer_obj = dictionary.Dictionary().create()
    mode = tokenizer.Tokenizer.SplitMode.C  # 分割モードC（最も長い単位）
    tokens = []
    for token in tokenizer_obj.tokenize(text, mode):
        if token.part_of_speech()[0] in ['名詞']: # 名詞のみ
            tokens.append(token.dictionary_form())
    return " ".join(tokens)

tokenized_texts = [tokenize_text(text) for text in texts]
print("トークン化されたテキスト:")
for i, text in enumerate(tokenized_texts):
    print(f"文書{i+1}: {text}")

# 1. Bag of Words（BoW）
bow_vectorizer = CountVectorizer()
X_bow = bow_vectorizer.fit_transform(tokenized_texts)

bow_df = pd.DataFrame(
    X_bow.toarray(), 
    columns=bow_vectorizer.get_feature_names_out(),
    index=["文書1", "文書2", "文書3"]
)
print("Bag of Words 表現:")
print(tabulate(bow_df, headers='keys', tablefmt='simple', showindex=True))
```

**特徴:** 
- 「機械学習」は3つの文書全てに出現するため、全ての行で値が入ります
- 単純に出現回数をカウントするため、文書1では「機械学習」が2回出現し、値が2になっています
- 単語の重要度は考慮されず、どの単語も同等に扱われます

### TF-IDF (Term Frequency-Inverse Document Frequency)

TF-IDFは単語の出現頻度だけでなく、その単語がどれだけ重要かも考慮します。

```{code-cell}
# TF-IDFモデルの作成
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(tokenized_texts)

tfidf_df = pd.DataFrame(
    X_tfidf.toarray(), 
    columns=tfidf_vectorizer.get_feature_names_out(),
    index=["文書1", "文書2", "文書3"]
)
print(tabulate(tfidf_df, headers='keys', tablefmt='plain'))
```

**特徴:**

- 「機械学習」は全ての文書に出現するため、IDF値は低めになりますが、文書1では出現頻度(TF)が高いため、結果的にTF-IDF値も他の文書より高くなっています
- 「深層学習」や「自然言語処理」のような特定の文書だけに出現する単語は高いTF-IDF値を持ちます


## 感情分析

多くの時間をネットで過ごしていると、さまざまなテキストに出会います。
その中でもネガティブなテキストは私たちの気分や感情状態に深刻な影響を及ぼします。

ある研究では以下のような結果が出ています。
- ネガティブな情報を多く閲覧するほど、不安・抑うつ、社会的回避や強迫的行動が顕著に表れる
- 気分が落ち込んでいると、無意識にさらにネガティブな情報を選択する傾向がある

このような悪循環を避けるためには、できる限りネガティブな情報との接触を減らすことが望ましいでしょう。
そのためには、文章から感情を事前に予測する必要があります。

機械学習では感情分析という技術があり、テキストに含まれる感情や意見を自動的に判別します。

感情分析は以下の分野でも使われています。

- 商品レビューの評判分析
- SNS上での企業や製品の評判監視
- カスタマーサポートの自動化
- マーケティング戦略の立案

ここでは最もシンプルな感情分析を実装してみましょう。
その方法とは文章を単語に分割し、文章中にあるポジティブ単語とネガティブな単語の出現回数から
その文章のポジティブ度とネガティブ度を計算するというものです。

```{code-cell}
import pandas as pd
from sudachipy import tokenizer, dictionary
from tabulate import tabulate 

def tokenize(text):
    tokenizer_obj = dictionary.Dictionary().create()
    mode = tokenizer.Tokenizer.SplitMode.C
    morphemes = tokenizer_obj.tokenize(text, mode)
    tokens = [m.dictionary_form() for m in morphemes]
    return tokens

# サンプルテキスト
texts = [
    "このレストランの料理はとても美味しかった。また行きたい。",
    "サービスが遅く、値段も高すぎる。二度と利用しない。",
    "製品の品質は普通だが、配送が早くて助かった。"
]

# ポジティブな単語とネガティブな単語のリスト
positive_words = ["美味しい", "良い", "素晴らしい", "助かる", "早い", "行く"]
negative_words = ["遅い", "高い", "悪い", "ひどい"]

results = []
for text in texts:
    tokens = tokenize(text)

    total_tokens = len(tokens)
    
    pos_matches = [word for word in tokens if word in positive_words]
    neg_matches = [word for word in tokens if word in negative_words]

    pos_score = len(pos_matches) / total_tokens if total_tokens > 0 else 0
    neg_score = len(neg_matches) / total_tokens if total_tokens > 0 else 0
        
    compound = pos_score - neg_score
    
    results.append({
        'テキスト': text,
        'ポジティブ': pos_score,
        'ネガティブ': neg_score,
        '複合スコア': compound
    })

df = pd.DataFrame(results)
print(tabulate(df, headers='keys', tablefmt='plain', showindex=False))
```

このコードは基本的ですが、以下のように発展できます。

- より豊富な感情辞書の使用
- 文脈や否定表現（「美味しくない」など）の考慮
- 感情の強さや複雑な感情の分析

## トピックモデリング

テキストデータを分析する際、感情分析だけでなく「どのような話題について書かれているか」を把握することも重要です。
トピックモデリングはテキスト集合から自動的に潜在的なトピックを抽出する技術です。
ここで潜在的とは、トピックが明示的に示されていないということです。

トピックモデリングを使うと、以下のようなことが可能になります。

- 大量の文書からの主要テーマの自動抽出
- 類似したコンテンツのグループ化
- 時間経過による話題の変化の追跡
- 文書の要約や分類の自動化

トピックモデリングで最も広く使われているのがLDA（Latent Dirichlet Allocation）です。
LDAは各文書が複数のトピックの混合であり、各トピックは単語の確率分布であると仮定します。

```{code-cell}
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sudachipy import tokenizer, dictionary

# 文書のサンプル
documents = [
    "私はラーメンが好きで、週末には新しい店を開拓しています。特に味噌ラーメンが好きです。",
    "昨日の野球の試合は接戦でした。9回裏にホームランが出て逆転勝ちしました。",
    "新しいスマートフォンを購入しました。カメラの性能が素晴らしく、バッテリーの持ちも良いです。",
    "東京では桜が満開です。花見客で公園は賑わっています。",
    "プログラミングの勉強を始めました。Pythonは初心者にも扱いやすい言語だと思います。",
    "今日の料理教室ではラーメンの作り方を習いました。スープの取り方が難しかったです。",
    "プロ野球の開幕戦を見に行きました。球場の雰囲気は最高でした。",
    "カメラの新機種が発売されました。高画質で夜景の撮影も綺麗に撮れるそうです。"
]

# 形態素解析で単語に分割
def tokenize_ja(text):
    tokenizer_obj = dictionary.Dictionary().create()
    mode = tokenizer.Tokenizer.SplitMode.C
    tokens = []
    for token in tokenizer_obj.tokenize(text, mode):
        if token.part_of_speech()[0] in ['名詞', '動詞', '形容詞']:  # 主要な品詞のみ抽出
            tokens.append(token.dictionary_form())
    return ' '.join(tokens)

# 単語分割を適用
processed_docs = [tokenize_ja(doc) for doc in documents]

# BoWモデルの作成
vectorizer = CountVectorizer(max_df=0.95, min_df=1)
X = vectorizer.fit_transform(processed_docs)

# トピック数を3に設定してLDAを実行
n_topics = 3
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(X)

# 各トピックの特徴的な単語を表示
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    print(f"トピック #{topic_idx+1}:")
    print(" ".join([feature_names[i] for i in topic.argsort()[:-6:-1]]))
    print()

# 各文書のトピック分布を表示
doc_topics = lda.transform(X)
for i, doc in enumerate(documents):
    print(f"文書 {i+1}:")
    print(doc[:50] + "...")  # 最初の50文字だけ表示
    topic_dist = [f"トピック{j+1}: {prob:.2f}" for j, prob in enumerate(doc_topics[i])]
    print(", ".join(topic_dist))
    print()
```

プログラムでは8つの文書から3つのトピックを抽出しています。
人間が指定した内容ではなく、プログラムが自動的にトピックを抽出しているので、直感とは異なる結果になっているかもしれません。

