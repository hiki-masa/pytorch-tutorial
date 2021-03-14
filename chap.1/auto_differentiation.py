import torch

# 関数
def f(x):
    return (2 * x**3) + (4 * x**2) + (5 * x) + 6

# requires_grad = True で微分の変数の設定
x = torch.tensor(0.5, requires_grad = True)
# 計算することで計算グラフの作成を行う
y = f(x)
# 微分の実行
y.backward()
# 勾配の計算
g = x.grad

print(g)