# 学习笔记

## PyTorch 线性回归要点
1. `nn.Linear(input_dim, 1)` — 单层全连接
2. `nn.MSELoss()` — 回归任务常用损失
3. `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()` — 标准训练三步
