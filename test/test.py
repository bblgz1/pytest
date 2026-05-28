# import torch, d2l
# print("PyTorch版本:", torch.__version__)
# print("d2l版本:", d2l.__version__)
# print("GPU可用:", torch.cuda.is_available())
# import torch
# import time

# # 确保安装了GPU版本的PyTorch
# print(f"PyTorch版本: {torch.__version__}")
# print(f"CUDA可用: {torch.cuda.is_available()}")
# print(f"设备数量: {torch.cuda.device_count()}")
# print(f"设备名称: {torch.cuda.get_device_name(0)}")

# # 定义设备（非常重要！）
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"使用的设备: {device}")

# # 在CPU上运行
# start = time.time()
# a = torch.randn(10000, 10000)
# b = torch.randn(10000, 10000)
# c = torch.matmul(a, b)
# print(f"CPU时间: {time.time()-start:.2f}秒")

# # 在GPU上运行
# start = time.time()
# a = a.to(device)  # 移动到设备
# b = b.to(device)  # 移动到设备
# c = torch.matmul(a, b)
# torch.cuda.synchronize()  # 确保所有GPU操作完成
# print(f"GPU时间: {time.time()-start:.2f}秒")
import torch
# x = torch.arange(12)
# print(x)
# print(x.shape)
# print(x.numel())
# X = x.reshape(3, 4)
# print(X)
# print(torch.ones(2, 3, 4))
# print(torch.randn(3, 4))
# x = torch.tensor([1.0, 2, 4, 8])
# y = torch.tensor([2, 2, 2, 2])
#print(x + y, x - y, x * y, x / y, x ** y)  # **运算符是求幂运算
#print(torch.exp(x))
# X = torch.arange(12, dtype=torch.float32).reshape((3,4))
# Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
# print(X>Y)
# print(X.sum())
# a = torch.arange(8).reshape(2,2,2)
# b = torch.arange(4).reshape(2,2)
# print(a)
# print(b)
# print(a+b)
# 
# import os

# os.makedirs(os.path.join('..', 'data'), exist_ok=True)
# data_file = os.path.join('..', 'data', 'house_tiny.csv')
# with open(data_file, 'w') as f:
#     f.write('NumRooms,Alley,Price\n')  # 列名
#     f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
#     f.write('2,NA,106000\n')
#     f.write('4,NA,178100\n')
#     f.write('NA,NA,140000\n')
# # 如果没有安装pandas，只需取消对以下行的注释来安装pandas
# # !pip install pandas
# import pandas as pd
# data = pd.read_csv(data_file)
# print(data)
# # import six
# # print(six.__version__)
# inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
# inputs = inputs.fillna(inputs.mean())
# print(inputs)
# inputs = pd.get_dummies(inputs, dummy_na=True)
# print(inputs)
# import torch

# X = torch.tensor(inputs.to_numpy(dtype=float))
# y = torch.tensor(outputs.to_numpy(dtype=float))
# X, y
# print(X.shape)
# print(y.shape)
# print(X)
# print(y)
# import torch
# x = torch.tensor(3.0)
# y = torch.tensor(2.0)
# print(x+y)
# print(x*y)
# print(x/y)
# print(x**y)
# x = torch.arange(4)
# print(x)
# print(x[3])
# A = torch.arange(20).reshape(5,4)
# # print(A)
# # print(A.T)
# B = A+A
# # print(B)
# # print(A*B)
# print(A*A)
# x = torch.arange(4, dtype=torch.float32)
# x, x.sum()
# A_sum_axis0 = A.sum(axis=0)#axis = ? 就说明降到几维
# print(A_sum_axis0,A_sum_axis0.shape)

# a = torch.zeros(4,9)
# print(a,torch.norm(a))%matplotlib inline
# import numpy as np
# import matplotlib.pyplot as plt
# from d2l import torch as d2l
# import matplotlib as mpl
# def f(x):
#     return 3 * x ** 2 - 4 * x

# # 示例绘图
# # x = np.linspace(-5, 5, 400)
# # y = f(x)

# # plt.plot(x, y)
# # plt.xlabel("x")
# # plt.ylabel("f(x)")
# # plt.grid(True)
# # plt.show()
# # def numerical_lim(f, x, h):
# #     return (f(x + h) - f(x)) / h

# # h = 0.1
# # for i in range(5):
# #     print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
# #     h *= 0.1
# def use_svg_display():  #@save
#     #"""使用svg格式在Jupyter中显示绘图"""
#     mpl.backend_inline.set_matplotlib_formats('svg')
# def set_figsize(figsize=(3.5, 2.5)):
#     d2l.plt.rcParams['figure.figsize'] = figsize
# #@save
# def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
#     """设置matplotlib的轴"""
#     axes.set_xlabel(xlabel)
#     axes.set_ylabel(ylabel)
#     axes.set_xscale(xscale)
#     axes.set_yscale(yscale)
#     axes.set_xlim(xlim)
#     axes.set_ylim(ylim)
#     if legend:
#         axes.legend(legend)
#     axes.grid()
# #@save
# def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
#          ylim=None, xscale='linear', yscale='linear',
#          fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
#     """绘制数据点"""
#     if legend is None:
#         legend = []

#     set_figsize(figsize)
#     axes = axes if axes else d2l.plt.gca()

#     # 如果X有一个轴，输出True
#     def has_one_axis(X):
#         return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
#                 and not hasattr(X[0], "__len__"))

#     if has_one_axis(X):
#         X = [X]
#     if Y is None:
#         X, Y = [[]] * len(X), X
#     elif has_one_axis(Y):
#         Y = [Y]
#     if len(X) != len(Y):
#         X = X * len(Y)
#     axes.cla()
#     for x, y, fmt in zip(X, Y, fmts):
#         if len(x):
#             axes.plot(x, y, fmt)
#         else:
#             axes.plot(y, fmt)
#     set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
# x = np.arange(0, 3, 0.1)
# plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
# import torch

# x = torch.arange(4.0)
# # 等价于x=torch.arange(4.0,requires_grad=True)
# x.requires_grad_(True)
# # 默认值是None
# x.grad#0，1，2，3
# # 计算y的值，y是x和x的点积的两倍
# y = 2 * torch.dot(x, x)
# # 对y进行反向传播计算 = 0*0+1*1+2*2+3*3
# y.backward()
# #获取x的梯度
# x.grad
# # 打印y的值
# # print(y)
# # 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
# x.grad.zero_()#梯度清零
# # print(x.grad)
# y = x.sum()
# # print(y)
# y.backward()#梯度计算
# # print(x.grad)
# x.grad.zero_()
# y = x*x
# # print(x.grad)
# y.sum().backward()
# # print(x.grad)
# x.grad.zero_()
# y = x * x
# u = y.detach()
# # print(u)
# z = u * x

# z.sum().backward()
# # print(x.grad == u)
# x.grad.zero_()
# def f(a):
#     b = a * 2
#     while b.norm() < 1000:
#         b = b * 2
#     if b.sum() > 0:
#         c = b
#     else:
#         c = 100 * b
#     return c
# print(f(x))
# import torch
# def f(a):
#     b = a * 2
#     while b.norm() < 1000:
#         print("\n",b.norm())
#         b = b * 2
#     if b.sum() > 0:
#         c = b
#         print("C==b\n",c)
#     else:
#         c = 100 * b
#         print("c=100b\n",c)
#     return c

# a = torch.randn(size=(3,1), requires_grad=True)
# print(a.shape)
# print(a)
# d = f(a)
# # d.backward() #<====== run time error if a is vector or matrix RuntimeError: grad can be implicitly created only for scalar outputs
# d.sum().backward() #<===== this way it will work
# print(d)
# import matplotlib.pylab as plt
# from matplotlib.ticker import FuncFormatter, MultipleLocator
# import numpy as np
# import torch

# # 创建一个子图，用于绘制正弦函数及其梯度
# f, ax = plt.subplots(1)

# # 定义x轴数据范围，从-3π到3π，共100个点
# x = np.linspace(-3*np.pi, 3*np.pi, 100)

# # 将numpy数组转换为torch张量，并设置requires_grad=True以计算梯度
# x1 = torch.tensor(x, requires_grad=True)

# # 计算正弦函数值
# y1 = torch.sin(x1)

# # 计算梯度（反向传播）
# # 通过sum()将张量转换为标量，以便计算梯度
# y1.sum().backward()

# # 绘制正弦函数曲线
# ax.plot(x, np.sin(x), label='sin(x)')

# # 绘制梯度曲线
# ax.plot(x, x1.grad, label="gradient of sin(x)")

# # 添加图例
# ax.legend(loc='upper center', shadow=True)

# # 设置x轴刻度格式化器，将数值转换为π的倍数显示
# ax.xaxis.set_major_formatter(FuncFormatter(
#     lambda val, pos: '{:.0g}$\pi$'.format(val/np.pi) if val != 0 else '0'
# ))

# # 设置x轴主刻度间隔为π
# ax.xaxis.set_major_locator(MultipleLocator(base=np.pi))

# # 显示图形
# plt.show()
import torch
import random
from d2l import torch as d2l
def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声 """
    X = torch.normal(0,1,(num_examples,len(w)))
    Y = torch.matmul(X,w) + b
    Y += torch.normal(0,0.01,Y.shape)
    return X,Y.reshape((-1,1))
True_w = torch.tensor([2,-3.4])
True_b = 4.2
features,labels = synthetic_data(True_w,True_b,1000)
print('features:',features[0],'\nlabel:',labels[0])
d2l.set_figsize()
d2l.plt.scatter(features[:,1].detach().numpy(),labels.detach().numpy(),1)
def data_iter(batch_size,features,labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices = torch.tensor(
            indices[i:min(i+batch_size,num_examples)]
        )
        yield features[batch_indices],labels[batch_indices]
batch_size = 10
for X,y in data_iter(batch_size,features,labels):
    print(X,y)
    break
w = torch.normal(0,0.01,size=(2,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)
def linreg(X,w,b):
    """线性回归模型"""
    return torch.matmul(X,w) + b
def squared_loss(y_hat,y):
    """均方损失"""
    return (y_hat-y.reshape(y_hat.shape))**2/2
def sgd(params,lr,batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad/batch_size
            param.grad.zero_()
lr = 0.03#学习率
num_epochs = 3#学习次数
net = linreg
loss = squared_loss
for epoch in range(num_epochs):
    for X,y in data_iter(batch_size,features,labels):
        l = loss(net(X,w,b),y)
        l.sum().backward()
        sgd([w,b],lr,batch_size)
    with torch.no_grad():
        train_l = loss(net(features,w,b),labels)
        print(f'epoch {epoch+1}, loss {float(train_l.mean()):f}')
print(f'w: {w}\nb: {b}')
print(w,b)
