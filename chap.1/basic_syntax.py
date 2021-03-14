import numpy as np
import torch

"""
Tensor の作成
"""
t = torch.tensor([[1, 2], [3, 4]])
# 0 から 9 までの数値で初期化された1次元の Tensor
t = torch.arange(0, 10)
# すべての値が 0 の 5行 × 10列 の Tensor を作成
t = torch.zeros(5, 10)
# 乱数の 5行 × 10列 の Tensor を作成
t = torch.randn(5, 10)

# Tensor の形状を取得
t.size()

"""
インデクシング操作
"""
t = torch.tensor([[1, 2, 3], [4, 5, 6]])
# スカラーの添字で指定
print(t[0, 2])
# スライス指定
print(t[:, 2])
print(t[:, :2])
# 添字のリストで指定
print(t[:, [1, 2]])
# マスク配列を使用して3よりも大きい部分のみ選択
print(t[t > 3])
# 指定要素の値の変更
t[0, 1] = 100
print(t)
# スライスを用いた一括代入
t[:, 1] = 200
print(t)
# マスク配列を使用して特定条件の要素のみ置換
t[t > 10] = 20
print(t)

"""
GPU関連
"""
# GPU が使えるかのチェック
if torch.cuda.is_available():
    # GPU に Tensor を作成
    t = torch.tensor([[1, 2], [3, 4]], device = "cuda:0")
    # 作成した Tensor を GPU に転送
    t = torch.zeros(100, 10).to("cuda:0")
else:
    print("Can't use GPU")

"""
numpy への変換
"""
# ndarray への変換
t = torch.tensor([[1, 2], [3, 4]])
x = t.numpy()
## GPU上の Tensor は，一度CPUの Tensor に変換する必要がある
#t = torch.tensor([[1, 2], [3, 4]], device = "cuda:0")
#x = t.to("cpu").numpy()