这三行是 SAC 的**变量替换公式（change of variables）**，计算经过 `tanh` 压缩和线性缩放后的实际动作的 log 概率。

---

## 数据流

```
x_t ∼ N(mean, std)    →    y_t = tanh(x_t)    →    action = y_t · scale + bias
  (无界高斯采样)              (压缩到 -1~1)              (映射到实际动作空间)
```

---

## 逐行讲解

### 第 1 行：高斯分布的 log 概率

```python
log_prob = normal.log_prob(x_t)   # shape: (batch, 5)
```

计算采样值 `x_t` 在原始高斯分布中的 **log 概率**（非真实动作的 log 概率）：

$$\log p(x_t) = -\frac{1}{2}\left(\frac{x_t - \mu}{\sigma}\right)^2 - \log\sigma - \frac{1}{2}\log(2\pi)$$

---

### 第 2 行：变量替换修正（Enforcing Action Bound）

```python
log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
```

这是**核心**。概率密度在变量变换时会"拉伸"或"收缩"，需要补偿：

$$\log \pi(\text{action}) = \log p(x_t) - \log\left|\det\frac{d(\text{action})}{d(x_t)}\right|$$

变换的雅可比行列式由两步组成：

| 变换 | 雅可比行列式 |
|------|-------------|
| `y_t = tanh(x_t)` | $\prod_i (1 - y_i^2)$ |
| `action = y_t · scale + bias` | $\prod_i \text{scale}_i$ |

合成：

$$\det = \prod_i \text{scale}_i \cdot (1 - y_i^2)$$

取对数就是减法：

```python
log_prob -= log(scale) + log(1 - y_t²)
```

**为什么叫 "Enforcing Action Bound"？** — 不做这个修正，计算出的 log 概率对应的是 unbounded 高斯中的值，而非实际 bounded 动作的概率，RL 训练会出错。

---

### 第 3 行：求和得到联合 log 概率

```python
log_prob = log_prob.sum(1, keepdim=True)   # shape: (batch, 1)
```

把 5 个独立动作维度的 log 概率**相加**，得到联合动作的 log 概率（假设各维度条件独立）：

$$\log \pi(a_1, a_2, a_3, a_4, a_5) = \sum_{i=1}^{5} \log \pi(a_i)$$

---

## 整体数学公式

整个过程等价于：

$$\log \pi(\text{action}) = \underbrace{\sum_{i=1}^{5} \log p\big(x_i \mid \mu_i, \sigma_i\big)}_{\text{高斯 log 概率}} \;-\; \underbrace{\sum_{i=1}^{5} \log\big(\text{scale}_i \cdot (1 - \tanh^2\!(x_i))\big)}_{\text{变量替换修正}}$$

其中 $\pi(\text{action})$ 是实际有界动作空间上的策略分布。SAC 的 actor 损失函数要用这个修正后的 log 概率来正确计算。