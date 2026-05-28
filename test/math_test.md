# 数学公式渲染测试

## Sigmoid 激活函数

公式：
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

导数：
$$\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))$$

## 线性回归 — 均方误差损失

线性模型预测：
$$\hat{y} = Wx + b$$

均方误差 (MSE)：
$$J(W, b) = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2$$

## 梯度下降更新规则

权重更新：
$$W := W - \alpha \frac{\partial J}{\partial W}$$

偏置更新：
$$b := b - \alpha \frac{\partial J}{\partial b}$$

## 多层感知机 (MLP)

单隐藏层前向传播：
$$H = \sigma(X W^{[1]} + b^{[1]})$$

输出层：
$$O = H W^{[2]} + b^{[2]}$$

## ReLU 激活函数

$$\text{ReLU}(x) = \max(0, x)$$

## Softmax 函数

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{k=1}^{K} e^{z_k}}$$
