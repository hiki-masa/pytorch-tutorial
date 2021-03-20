import torch
import matplotlib.pyplot as plt

# 関数
def f(x):
    return (2 * x**3) - (3 * x**2) - (12 * x) + 3

# requires_grad = True で微分の変数の設定
x = torch.arange(-4, 4, 0.01, requires_grad = True)
# 計算することで計算グラフの作成を行う
y = f(x)
# 微分の実行
y.backward(gradient = torch.ones_like(y))   # 複数のスカラー値の場合，勾配を格納する場所が必要なので，テンソルを仮生成
# 勾配の計算
g = x.grad

# グラフの作成・設定
plt.plot(x.detach().numpy(), y.detach().numpy(), label = "f(x)")
plt.plot(x.detach().numpy(), g.detach().numpy(), label = "f'(x)")
plt.xlim(-4, 4)
plt.ylim(-30, 30)
plt.grid()
plt.legend()
plt.show()