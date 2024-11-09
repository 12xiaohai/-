import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 例如：SimHei 或者其他支持中文的字体

# 数据
data = np.array([[0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8],
                 [19.21, 18.15, 15.36, 14.10, 12.89, 9.32, 7.45, 5.24, 3.01]])

# 定义符号变量
t, a = sp.symbols('t a')

# 生成矩阵 A 和向量 B
Tnx = np.vstack([np.ones(len(data[0, :])), data[0, :]])  # Tnx is [1; t]
A = np.zeros((2, 2))
B = np.zeros(2)

# 计算矩阵 A 和 B
for i in range(2):
    for j in range(2):
        A[i, j] = np.sum(Tnx[i, :] * Tnx[j, :])

yy = np.log(data[1, :])
for i in range(2):
    B[i] = np.sum(yy * Tnx[i, :])

# 求解 a 和 b
alp = np.linalg.inv(A).dot(B)

# 得到拟合的对数形式表达式
S_log = alp[0] + alp[1] * t  # 对数形式

# 将 S_log 转换为一个数值函数
S_log_func = sp.lambdify(t, S_log, 'numpy')  # 数值函数 S_log_func(t)
S_func = lambda t_val: np.exp(S_log_func(t_val))  # 原始浓度形式的数值函数

# 输出拟合的公式，采用指数形式 a * e^(b * t)
fit_expr = f'C(t) = {np.exp(alp[0]):.3f} * e^({alp[1]:.3f} * t)'  # 采用 e 的形式输出

# 数值拟合
t_vals = np.linspace(0, 8, 1000)
C_vals = S_func(t_vals)  # 使用 S_func 计算拟合曲线的浓度值

# 绘制图形
plt.scatter(data[0, :], data[1, :], color='black', label='数据点')
plt.plot(t_vals, C_vals, label='拟合曲线', linewidth=2)
plt.xlabel('时间 t (h)')
plt.ylabel('浓度 C (μg/mL)')
plt.legend()

# 将简化的拟合公式添加到图片上方
plt.text(0.5, 1.05, fit_expr, ha='center', va='bottom', transform=plt.gca().transAxes, fontsize=12)

plt.show()
