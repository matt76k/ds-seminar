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
再帰関数
=======

再帰関数は初心者にとっては難しい概念ですが、
ステップバイステップで理解すれば、それほど難しくはありません。

まず、関数から関数を呼べることを確認します。

```{code-cell}
def neko():
    print('ネコ')

def koneko():
    print('コネコ')

def main():
    neko()
    neko()
    koneko()

main()
```

では、mainから`neko`だけを呼び出して、同じ出力を得るにはどうしたらよいでしょうか。

```{code-cell}
def main():
    neko() # mainからは`neko`だけを呼び出す

main()
```

解答の1つを示します。

```{code-cell}
def neko():
    print('ネコ')
    print('ネコ')
    koneko()

def main():
    neko()

main()
```

これは1つの例であり、他にも様々な方法が考えつくでしょう。
ここで学びたいのは、関数から関数を呼べるということです。

次に、1つの関数で同じような出力をする関数を作ります。

```{code-cell}
def neko_or_koneko(b):
    if b == 0:
        print('コネコ')
    else:
        print('ネコ')

neko_or_koneko(2)
neko_or_koneko(1)
neko_or_koneko(0)
```

この関数は、0を渡したら'コネコ'を、0以外は'ネコ'を表示します。
このように与える引数で関数の動作を変えることができます。

これを`for`文で書くとどうなるでしょうか？
前のコードを見て、与えてる引数の値がカウントダウンされていることに気づきましたか？

```{code-cell}
for i in range(2, -1, -1):
    neko_or_koneko(i)
```

前のプログラムがスッキリしましたね。
0が終わりを意味しているわけです。
カウントダウンという考えではなくカウントアップにしてしまうと、
どこで終わるかの数値を指定しないといけなくなるため、関数に余分な引数を与える必要が出てきてしまいます。

さらに`for`文まで無くしてみましょう。
そのためには、`neko_or_koneko`の中で`neko_or_koneko`を呼び出すようにします。
このように、関数の中で自分自身を呼び出すことを再帰呼び出しといい、そのような関数を**再帰関数**といいます。

```{code-cell}
def neko_or_koneko(b):
    if b == 0:
        print('コネコ')
    else:
        print('ネコ')
        neko_or_koneko(b - 1)

neko_or_koneko(2)
```

再帰関数の基本的な構造は、以下の通りです。

```python
def func(n):
    if n == 0: # 終了
        return val
    else: # 継続
        #ここで何かして
        func(n - 1) # 自分自身を呼び出す
```

データ構造やアルゴリズムによっては、再帰関数が最も自然な表現方法になります。

たとえば、[フィボナッチ数列](https://ja.wikipedia.org/wiki/%E3%83%95%E3%82%A3%E3%83%9C%E3%83%8A%E3%83%83%E3%83%81%E6%95%B0)です。

```{code-cell}
def fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n - 1) + fib(n - 2)

fib_list = [fib(i) for i in range(10)]
print(fib_list)
```

