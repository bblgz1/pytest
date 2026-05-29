# '''单层线性回归'''
# import torch
# import random
# from d2l import torch as d2l

# def synthetic_data(w, b, num_examples):
#     """生成 y = Xw + b + 噪声 

#     Args:
#         w (torch.Tensor): 线性模型的权重参数
#         b (float): 线性模型的偏置参数
#         num_examples (int): 生成样本的数量

#     Returns:
#         tuple: 包含特征矩阵X和标签向量Y的元组
#     """
#     X = torch.normal(0,1,(num_examples,len(w)))
#     Y = torch.matmul(X,w) + b
#     Y += torch.normal(0,0.01,Y.shape)
#     return X,Y.reshape((-1,1))

# True_w = torch.tensor([2,-3.4])
# True_b = 4.2
# features,labels = synthetic_data(True_w,True_b,1000)
# print('features:',features[0],'\nlabel:',labels[0])
# d2l.set_figsize()
# d2l.plt.scatter(features[:,1].detach().numpy(),labels.detach().numpy(),1)

# def data_iter(batch_size,features,labels):
#     """随机批量数据迭代器

#     Args:
#         batch_size (int): 批量大小
#         features (torch.Tensor): 特征矩阵
#         labels (torch.Tensor): 标签向量

#     Yields:
#         tuple: 每次迭代返回一个批量的特征和标签
#     """
#     num_examples = len(features)
#     indices = list(range(num_examples))
#     random.shuffle(indices)
#     for i in range(0,num_examples,batch_size):
#         batch_indices = torch.tensor(
#             indices[i:min(i+batch_size,num_examples)]
#         )
#         yield features[batch_indices],labels[batch_indices]

# batch_size = 10
# for X,y in data_iter(batch_size,features,labels):
#     print(X,y)
#     break
# w = torch.normal(0,0.01,size=(2,1),requires_grad=True)
# b = torch.zeros(1,requires_grad=True)

# def linreg(X,w,b):
#     """线性回归模型

#     Args:
#         X (torch.Tensor): 输入特征
#         w (torch.Tensor): 权重参数
#         b (torch.Tensor): 偏置参数

#     Returns:
#         torch.Tensor: 线性模型预测结果
#     """
#     return torch.matmul(X,w) + b

# def squared_loss(y_hat,y):
#     """均方损失

#     Args:
#         y_hat (torch.Tensor): 模型预测值
#         y (torch.Tensor): 真实标签值

#     Returns:
#         torch.Tensor: 预测值与真实值之间的均方误差
#     """
#     return (y_hat-y.reshape(y_hat.shape))**2/2

# def sgd(params,lr,batch_size):
#     """小批量随机梯度下降

#     Args:
#         params (list): 待优化的参数列表
#         lr (float): 学习率
#         batch_size (int): 批量大小
#     """
#     with torch.no_grad():
#         for param in params:
#             param -= lr * param.grad/batch_size
#             param.grad.zero_()

# lr = 0.03#学习率
# num_epochs = 3#学习次数
# net = linreg
# loss = squared_loss

# # 模型训练过程
# for epoch in range(num_epochs):
#     for X,y in data_iter(batch_size,features,labels):
#         l = loss(net(X,w,b),y)# X和y的小批量损失
#         # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
#         # 并以此计算关于[w,b]的梯度
#         l.sum().backward()
#         sgd([w,b],lr,batch_size)# 使用参数的梯度更新参数
#     with torch.no_grad():
#         train_l = loss(net(features,w,b),labels)
#         print(f'epoch {epoch+1}, loss {float(train_l.mean()):f}')
# print(f'w: {w}\nb: {b}')
# print(w,b)
# '''线性回归简洁实现(单层线性回归)'''
# import numpy as np
# import torch
# from torch.utils import data
# from d2l import torch as d2l

# # 定义线性回归的真实参数
# true_w = torch.tensor([2,-3.4])
# true_b = 4.2

# # 生成合成数据集
# features,labels = d2l.synthetic_data(true_w,true_b,1000)


# def load_array(data_arrays,batch_size,is_train=True):
#     """将数据加载为小批量数据集

#     参数:
#         data_arrays (tuple): 包含特征和标签的张量数组
#         batch_size (int): 每个批次的样本数量
#         is_train (bool): 是否为训练模式（是否打乱数据）

#     返回:
#         DataLoader: 可迭代的小批量数据加载器
#     """
#     dataset = data.TensorDataset(*data_arrays)
#     return data.DataLoader(dataset,batch_size,shuffle=is_train)

# # 定义批量大小并创建数据加载器
# Batch_size = 10
# data_iter = load_array((features,labels),Batch_size)

# # 查看一个批次的数据（演示用）
# next(iter(data_iter))

# # 定义神经网络模型（单层线性回归）
# import torch.nn as nn#nn是神经网络缩写
# net = nn.Sequential(nn.Linear(2,1))

# # 初始化模型参数（权重和偏置）
# net[0].weight.data.normal_(0,0.01)  # 权重初始化为均值0、标准差0.01的正态分布
# net[0].bias.data.fill_(0)  # 偏置初始化为0

# # 定义损失函数（均方误差）
# loss = nn.MSELoss()#MSE损失函数

# # 定义优化器（随机梯度下降）
# trainer = torch.optim.SGD(net.parameters(),lr=0.03)#SGD优化器

# # 训练模型
# num_epochs = 3
# for epoch in range(num_epochs):
#     for X,y in data_iter:
#         l = loss(net(X),y)
#         trainer.zero_grad()
#         l.backward()
#         trainer.step()
    
#     # 每轮训练后评估损失
#     l = loss(net(features),labels)
#     print(f'epoch {epoch+1},loss {l:f}')
'''图像分类数据集(Fashion-MNIST)'''
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
d2l.use_svg_display()
# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0～1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root="../data",train=True,transform=trans,download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="../data",train=False,transform=trans,download=True)
print(len(mnist_train),len(mnist_test))
print(mnist_train[0][0].shape)
# 可视化数据集的函数
def get_fashion_mnist_labels(Labels):
    '''返回 fashion-mnist数据集的文本标签'''
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in Labels]
def show_images(imgs,num_rows,num_cols,titles=None,scale=1.5):
    '''绘制图像'''
    figsize =   (num_cols * scale, num_rows * scale)
    _,axes = d2l.plt.subplots(num_rows,num_cols,figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 将张量转为NumPy数组
            ax.imshow(img.numpy())
        else:
            # 显示图片
            ax.imshow(img)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            if titles: ax.set_title(titles[i])  
    return axes     
X,y = next(iter(data.DataLoader(mnist_train,batch_size=18)))
show_images(X.reshape(18,28,28),2,9,titles=get_fashion_mnist_labels(y))
#读取小批量数据
batch_size = 256
def get_dataloader_workers():
    num_workers = 4
    return num_workers
train_iter = data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=get_dataloader_workers())
timer = d2l.Timer()
for X,y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')
#定义load_data_fashion_mnist函数
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集,然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))
    
# '''softmax回归'''
# import torch
# from IPython import display
# from d2l import torch as d2l
# batch_size = 256
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# num_inputs = 784
# num_outputs = 10
# W = torch.normal(0,0.01,(num_inputs,num_outputs),requires_grad=True)
# b = torch.zeros(num_outputs,requires_grad=True)
# X = torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]])
# X.sum(dim=0,keepdim=True),X.sum(dim=1,keepdim=True)
# def softmax(X):
#     X_exp = torch.exp(X)
#     partition = X_exp.sum(1,keepdim=True)
#     return X_exp / partition# 这里应用了广播机制
# #验证
# X = torch.normal(0,1,(2,5))
# X_prob = softmax(X)
# print(X_prob,X_prob.sum(1))
# #softmax回归的模型
# def net(X):
#     return softmax(torch.matmul(X.reshape((-1,W.shape[0])),W)+b)
# #损失函数
# y = torch.tensor([0,2])
# y_hat = torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]])
# def cross_entropy(y_hat,y):
#     return -torch.log(y_hat[range(len(y_hat)),y])
# print(cross_entropy(y_hat,y))
# #与真实y元素进行比较
# def accuracy(y_hat,y):
#     """计算预测正确的数量"""
#     if len(y_hat.shape)>1 and y_hat.shape[1]>1:
#         y_hat = y_hat.argmax(axis=1)
#     cmp = y_hat.type(y.dtype)==y
#     return float(cmp.type(y.dtype).sum())
# print(accuracy(y_hat,y)/len(y))
# #评价函数(评估模型在数据集上的准确率)
# def evaluate_accuracy(net,data_iter):
#     """计算在指定数据集上模型的准确率(精度)"""
#     if isinstance(net,torch.nn.Module):
#         net.eval()#评估模式，这会关闭dropout
#     metric = Accumulator(2)#统计预测正确的数量和预测的总数量
#     with torch.no_grad():
#         for X,y in data_iter:
#             metric.add(accuracy(net(X),y),y.numel())
#     return metric[0]/metric[1]
# #迭代器 accumulator
# class Accumulator:  #@save
#     """在n个变量上累加"""
#     def __init__(self, n):
#         self.data = [0.0] * n

#     def add(self, *args):
#         self.data = [a + float(b) for a, b in zip(self.data, args)]

#     def reset(self):
#         self.data = [0.0] * len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]
# #softmax回归的训练
# def train_epoch_ch3(net, train_iter, loss, updater):  #@save
#     """训练模型一个迭代周期(定义见第3章)"""
#     # 将模型设置为训练模式
#     if isinstance(net, torch.nn.Module):
#         net.train()
#     # 训练损失总和、训练准确度总和、样本数
#     metric = Accumulator(3)
#     for X, y in train_iter:
#         # 计算梯度并更新参数
#         y_hat = net(X)
#         l = loss(y_hat, y)
#         if isinstance(updater, torch.optim.Optimizer):
#             # 使用PyTorch内置的优化器和损失函数
#             updater.zero_grad()
#             l.mean().backward()
#             updater.step()
#         else:
#             # 使用定制的优化器和损失函数
#             l.sum().backward()
#             updater(X.shape[0])
#         metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
#     # 返回训练损失和训练精度
#     return metric[0] / metric[2], metric[1] / metric[2]
# #animator
# class Animator:  #@save
#     """在动画中绘制数据"""
#     def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
#                  ylim=None, xscale='linear', yscale='linear',
#                  fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
#                  figsize=(3.5, 2.5)):
#         # 增量地绘制多条线
#         if legend is None:
#             legend = []
#         d2l.use_svg_display()
#         self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
#         if nrows * ncols == 1:
#             self.axes = [self.axes, ]
#         # 使用lambda函数捕获参数
#         self.config_axes = lambda: d2l.set_axes(
#             self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
#         self.X, self.Y, self.fmts = None, None, fmts

#     def add(self, x, y):
#         # 向图表中添加多个数据点
#         if not hasattr(y, "__len__"):
#             y = [y]
#         n = len(y)
#         if not hasattr(x, "__len__"):
#             x = [x] * n
#         if not self.X:
#             self.X = [[] for _ in range(n)]
#         if not self.Y:
#             self.Y = [[] for _ in range(n)]
#         for i, (a, b) in enumerate(zip(x, y)):
#             if a is not None and b is not None:
#                 self.X[i].append(a)
#                 self.Y[i].append(b)
#         self.axes[0].cla()
#         for x, y, fmt in zip(self.X, self.Y, self.fmts):
#             self.axes[0].plot(x, y, fmt)
#         self.config_axes()
#         display.display(self.fig)
#         display.clear_output(wait=True)
# #训练函数
# def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
#     animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
#                         legend=['train loss', 'train acc', 'test acc'])
#     for epoch in range(num_epochs):
#         train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
#         test_acc = evaluate_accuracy(net, test_iter)
#         animator.add(epoch + 1, train_metrics + (test_acc,))
#     print('loss {:.3f}, train acc {:.3f}, test acc {:.3f}'.format(
#         train_metrics[0], train_metrics[1], test_acc))
# #训练一个epoch
# lr = 0.1
# def updater(batch_size): return d2l.sgd([W,b],lr,batch_size)
# num_epochs = 10
# train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
# #预测
# def predict_ch3(net, test_iter, n=6):  #@save
#     """预测标签(定义见第3章)"""
#     for X, y in test_iter:
#         break
#     trues = d2l.get_fashion_mnist_labels(y)
#     preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
#     titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
#     d2l.show_images(
#         X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

# predict_ch3(net, test_iter)
'''softmax回归简介实现'''
# import torch
# from torch import nn
# from d2l import torch as d2l

# batch_size = 256
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# #初始化模型参数
# # PyTorch不会隐式地调整输入的形状。因此，
# # 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
# net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

# def init_weights(m):
#     if type(m) == nn.Linear:
#         nn.init.normal_(m.weight, std=0.01)

# net.apply(init_weights)
# loss = nn.CrossEntropyLoss(reduction='none')
# #优化算法
# trainer = torch.optim.SGD(net.parameters(), lr=0.1)
# #训练
# num_epochs = 10
# d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
# print('训练完成')
# for X,y in test_iter:
#     print(f'测试集准确率{torch.sum(torch.argmax(net(X),dim=1)==y)/y.shape[0]}')
#     break
'''多层感知机实现'''
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# import torch
# from torch import nn
# from d2l import torch as d2l
# batch_size = 256
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# num_inputs, num_outputs, num_hiddens = 784, 10, 256
# W1 = nn.Parameter(torch.randn(
#     num_inputs, num_hiddens, requires_grad=True))
# b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
# W2 = nn.Parameter(torch.randn(
#     num_hiddens, num_outputs, requires_grad=True))
# b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
# params = [W1, b1, W2, b2]
# def relu(X):#激活函数
#     a = torch.zeros_like(X)
#     return torch.max(X,a)
# def net(X):#模型
#     X = X.reshape((-1, num_inputs))
#     H = relu(X@W1 + b1)
#     return (H@W2 + b2)
# loss = nn.CrossEntropyLoss(reduction='none')
# num_epochs, lr = 10, 0.1
# updater = torch.optim.SGD(params, lr=lr)
# d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
'''多层感知机简洁实现'''
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 训练模型并记录指标
train_losses = []
train_accs = []
test_accs = []

# 自定义训练函数，记录训练过程中的指标
def train_and_eval(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        # 训练一个epoch
        train_metrics = d2l.train_epoch_ch3(net, train_iter, loss, updater)
        # 计算测试集准确率
        test_acc = d2l.evaluate_accuracy(net, test_iter)
        
        # 记录指标
        train_losses.append(train_metrics[0])
        train_accs.append(train_metrics[1])
        test_accs.append(test_acc)
        
        print(f'epoch {epoch + 1}, loss {train_metrics[0]:.3f}, '
              f'train acc {train_metrics[1]:.3f}, test acc {test_acc:.3f}')

# 训练模型
train_and_eval(net, train_iter, test_iter, loss, num_epochs, trainer)

# 可视化训练过程
plt.figure(figsize=(12, 4))

# 绘制训练损失
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

# 绘制准确率
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accs, label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), test_accs, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Comparison')

# 保存图像
plt.tight_layout()
plt.savefig('outputs/training_process.png')
print("训练过程图像已保存为 'outputs/training_process.png'")


