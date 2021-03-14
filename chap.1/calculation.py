import torch

"""
四則演算
"""
# ベクトル の設定
v1 = torch.tensor([[1, 2, 3]])
v2 = torch.tensor([[10, 20, 30]])
# ベクトル 演算
print(v1 + 10)
print(v1 * 100)
print(v2 - v1)
# 行列 の設定
m1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
m2 = torch.tensor([[10, 20, 30], [40, 50, 60]])
# 行列 の演算
print(m1 + 10)
print(m1 * 2)
print(m1 + m1)
print(m1 * m1)  #（注）行列の各要素ごとの乗算
# ベクトル と 行列 の演算（ブロードキャストが働く）
print(m1 + v1)
print()

"""
インデクシング操作
"""
m1 = torch.tensor([[1, 2], [3, 4]])
m2 = torch.tensor([[10, 20, 30], [40, 50, 60]])
# 4×1 に見せる
print(m1.view(4, 1))
# -1は残りの次元を表している．以下の例では 4 に自動変換される
print(m1.view(1, -1))
# 転置
print(m2.t())
# 結合
print(torch.cat([m1, m2], dim = 1))
# 次元の入れ替え
m3 = torch.randn(3,5,7)
print(m3.transpose(0,2).size())
print()

"""
数学関数
"""
x = torch.randn(2, 5)
print(x)
# 絶対値
print(torch.abs(x))
# 平均値計算
print(torch.mean(x))
print(x.mean())
# 値のみの取り出し
print(x.mean().item())
# スライスの平均値
print(x.mean(0))    # 行方向に集計し，列ごとに平均値計算
print(x.mean(1))    # 列ごとに計算し，行ごとに平均値計算
print()

"""
線形代数演算子
"""
v = torch.randn(3)
m = torch.randn(5, 3)
print(v)
print(m)
# 内積
print(torch.dot(v, v))
print(torch.mv(m, v))
print(torch.mm(m, m.t()))   #線形代数で用いられる行列の掛け算（m @ m.t()でも行える）
# 特異値分解
u, s, v = torch.svd(m)