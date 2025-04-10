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
強化学習
=======
```{epigraph}
兵に常勢なく水に常形なし

-- 孫子
```

私たちは日常生活で様々な状況に直面し、行動を選択します。
その結果に基づいて次の行動を調整していきます。  
「この行動は良い結果をもたらした」「あの選択は失敗だった」という経験から学習するプロセスは、まさに**強化学習**の考え方そのものです。

強化学習とは、**環境との相互作用**を通じて、**報酬を最大化**するように行動を学習する機械学習の一分野です。
エージェントは行動を選択し、環境から状態と報酬を受け取り、これらの経験に基づいて方策を改善していきます。

強化学習の応用例は多岐にわたります。

- ゲームAIの開発
- ロボット制御
- 自動運転技術
- 推薦システム
- トレーディングアルゴリズム

## 強化学習の基本

強化学習には以下の基本要素があります：

1. **エージェント**：学習し行動する主体
2. **環境**：エージェントが相互作用する外部世界
3. **状態 (S)**：環境の現在の状況
4. **行動 (A)**：エージェントが選択できる選択肢
5. **報酬 (R)**：行動の結果として環境から得られる数値
6. **方策 (Policy)**：各状態でどの行動を選ぶかを決める戦略

強化学習の目標は、将来の報酬の総和（累積報酬）を最大化する最適な方策を見つけることです。

## Q-Learningとは

Q-Learningは、環境と相互作用しながら最適な決定を行うようにエージェントを訓練するためのモデルフリー強化学習アルゴリズムです。
Q-Learningの「Q」は「Quality（質）」を表し、特定の状態で特定の行動をとることの価値（Q値）を学習します。

**モデルフリー**とは、環境のモデル（状態遷移確率や報酬の仕組み）を事前に知らなくても、試行錯誤を通じて学習できるという意味です。

### Q値とQテーブル

Q値は特定の状態で特定の行動をとることの期待報酬を表します。
これらの値はQテーブルに格納され、エージェントの記憶構造として機能します。

Qテーブルの構造：

- 行は状態を表す
- 列は可能な行動を表す
- 各セルには、その状態-行動ペアのQ値が含まれる

エージェントが環境を探索し、相互作用から学習するにつれて、Qテーブルを更新し、意思決定能力を徐々に向上させていきます。

### Q-Learningの更新式

Q-learningの核心は、時間差分（Temporal Difference）更新式です。

```{math}
:label: td-eq
Q(S,A) \leftarrow Q(S,A) + \alpha (R + \gamma \max_{A'} Q(S',A') - Q(S,A))
```

ここで：
- $Q(S,A)$ は状態Sで行動Aをとる現在のQ値
- $\alpha$ （アルファ）は学習率（0～1）で、新しい情報をどの程度Q値に反映させるかを決定します
- $R$ は状態Sで行動Aをとった際に得られる報酬
- $\gamma$ （ガンマ）は割引率（0～1）で、即時報酬と将来の報酬のバランスを取る
- $S'$ はエージェントが次に移動する状態
- $A'$ は状態$S'$での最良の次の行動
- $\max_{A'} Q(S',A')$ は次の状態での最大Q値

この式は、「現在の見積もり」と「新しい情報に基づく見積もり」の差を取り、その一部を現在の見積もりに加えることで、エージェントが徐々にQ値を洗練させる助けとなります。

### 探索と活用のバランス

Q-learningの重要な側面は、探索（より良い戦略を発見するために新しい行動を試す）と活用（既知の良い戦略を使用する）のバランスを取ることです。これは通常、$\epsilon$-greedy方策によって実現されます。

- **活用**：確率$1-\epsilon$で、エージェントは最高のQ値を持つ行動を選択
- **探索**：確率$\epsilon$で、エージェントはランダムな行動を選択

時間の経過とともに、$\epsilon$の値は通常減少し、エージェントが環境についてより多くを学ぶにつれて、探索から活用への移行を可能にします。

## Q-Learningアルゴリズムの実装

簡単な迷路問題でQ-Learningを実装してみましょう。
4×4のグリッド上で、スタートから目標地点まで最短経路を学習させます。

### 迷路環境の準備

まずは迷路環境を定義します。この環境では

- 0は通路
- 1は壁
- 2はゴール地点

となります。

エージェントはスタート地点からゴールまでを目指します。

```{code-cell}
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap

class MazeEnvironment:
    def __init__(self, start=(2, 0)):
        # 0: 通路, 1: 壁, 2: ゴール
        self.maze = np.array([
            [0, 0, 0, 0],
            [0, 1, 0, 1],
            [0, 0, 0, 0],
            [1, 1, 0, 2]
        ])
        self.start_state = start  # スタート位置
        self.current_state = self.start_state
        self.goal_state = (3, 3)  # ゴール位置
        self.actions = ["up", "right", "down", "left"]
        
    def reset(self):
        self.current_state = self.start_state
        return self.current_state
    
    def step(self, action):
        i, j = self.current_state
        
        # 行動に基づいて次の状態を計算
        if action == "up" and i > 0:
            next_state = (i-1, j)
        elif action == "right" and j < 3:
            next_state = (i, j+1)
        elif action == "down" and i < 3:
            next_state = (i+1, j)
        elif action == "left" and j > 0:
            next_state = (i, j-1)
        else:
            next_state = (i, j)
            
        # 壁に当たった場合は移動しない
        next_i, next_j = next_state
        if self.maze[next_i, next_j] == 1:
            next_state = (i, j)
            
        self.current_state = next_state
        
        if next_state == self.goal_state:
            reward = 100  # ゴールに到達した場合の報酬
            done = True
        else:
            reward = -1  # 各ステップのペナルティ
            done = False
            
        return next_state, reward, done
```

この環境クラスでは、エージェントが取れる行動（上、右、下、左）と、それに対する状態遷移、報酬の計算を定義しています。
ゴールに到達すると大きな報酬（+100）を得られますが、それ以外の各ステップでは小さなペナルティ（-1）を受けるため、
エージェントには最短経路を見つける動機が生まれます。

### 環境の可視化

迷路や学習状況を視覚化するための関数を追加します。

```{code-cell}
def render(maze, q_table=None):
    # 迷路の可視化
    plt.figure(figsize=(6, 6))
    
    # 色分けの設定
    colors = ListedColormap(['white', 'black', 'green', 'red'])
    plt.imshow(maze.maze, cmap=colors)
    
    # グリッド線の表示
    plt.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    plt.xticks(np.arange(-.5, 4, 1), [])
    plt.yticks(np.arange(-.5, 4, 1), [])
    
    # 現在位置の表示
    i, j = maze.current_state
    plt.plot(j, i, 'bo', markersize=15)
    
    plt.title("Maze Environment")
    plt.show()
```

この関数は迷路の状態を視覚的に表示します。

実際に迷路を可視化してみましょう。
このプログラムを実行すると、迷路の初期状態が表示されます。

```{code-cell}
env = MazeEnvironment()
render(env)
```

青い点がエージェントの位置になります。
次に、このエージェントを上に動かし、その結果を見てみましょう。

```{code-cell}
env.step("up")
render(env)
```

エージェントが上に動いた結果を確認できましたね。

### Q学習アルゴリズムの実装

Q学習のアルゴリズムを実装します。

q_tableは、状態と行動のペアをキーとして、Q値を格納する辞書になります。
また、初期値は全て0にしておきます。

```{code-cell}
q_table = {}
# Q-tableの初期化
for i in range(4):
    for j in range(4):
        for a in ["up", "right", "down", "left"]:
            q_table[((i, j), a)] = 0
```

Q-Learningのアルゴリズムを実装します。

```{code-cell}
---
mystnb:
  number_source_lines: true
---
def q_learning(env, q_table, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, num_episodes=1000):
    actions = ["up", "right", "down", "left"]
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            # 行動選択（ε-greedy方策）
            if random.uniform(0, 1) < epsilon:
                action = random.choice(actions)
            else:
                q_values = [q_table[(state, a)] for a in actions]
                action = actions[np.argmax(q_values)]
                        
            # 行動を実行し、次の状態と報酬を観測
            next_state, reward, done = env.step(action)
            
            # Q値の更新
            old_q_value = q_table[(state, action)]
            next_max_q = max([q_table[(next_state, a)] for a in actions])
            
            new_q_value = old_q_value + learning_rate * (reward + discount_factor * next_max_q - old_q_value)
            q_table[(state, action)] = new_q_value
            
            # 状態を更新
            state = next_state
            steps += 1
            
            if steps > 100:  # 無限ループ防止
                break
                
        epsilon = max(epsilon_min, epsilon * epsilon_decay)        
```

`q_learning`関数は、Q-Learningのアルゴリズムを実装しています。
引数は

- `env`: 環境
- `q_table`: Qテーブル
- `learning_rate`: 学習率
- `discount_factor`: 割引率
- `epsilon`: ε-greedyパラメータ
- `epsilon_decay`: εの減衰率
- `epsilon_min`: εの最小値
- `num_episodes`: エピソード数

を受け取ります。
環境とQテーブルは必須で、それ以外はデフォルト値を設定しています。
学習率0.1、割引率0.95を使用しています。
学習率が小さいほど慎重に学習し、大きいほど新しい情報を強く反映します。
割引率が大きいほど将来の報酬を重視します。

21-25行目は、Q値の更新式{eq}`td-eq`に対応しています。

では実際に学習して、start地点のQ値を見てみましょう。

```{code-cell}
env = MazeEnvironment()
q_learning(env, q_table)
for a in env.actions:
    print(f"{a}: {q_table[(env.start_state, a)]}")
```

Q値の結果から、start地点からは右に行くと良さそうですね。

### 学習した方策

学習したQ値を使って、エージェントに最適な経路をたどらせてみましょう。

```{code-cell}
state = env.reset()
path = [state]
done = False

render(env)

while not done and len(path) < 20:
    q_values = [q_table[(state, a)] for a in env.actions]
    action = env.actions[np.argmax(q_values)]

    next_state, _, done = env.step(action)

    path.append(next_state)

    render(env)

    state = next_state
```

ちゃんとゴールに到達していますね。

- Q-tableの初期値を`random`にしてみましょう。
- 各マスでQ値が大きいほうに矢印を表示してみましょう。