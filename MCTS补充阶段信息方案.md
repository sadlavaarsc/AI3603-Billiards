好，这里我会**把 MCTS 侧的补救方案写成一份「可以直接交给工程同伴实现的完整技术方案」**。
目标非常明确：**不改模型结构、不动训练数据格式，只通过 MCTS 解决“比赛进程缺失”问题**。

我会按 **设计目标 → 总体思路 → 模块级修改 → 参数建议 → 验证方式 → 风险说明** 的结构来写。

---

# MCTS 阶段感知（Game-Phase Aware MCTS）技术方案

## 1. 设计目标

### 问题

当前 dual network 未输入比赛进程（回合数 / 局面阶段），导致：

* 相同球面在开局与终局被赋予相同价值
* 出现终局保守 / 开局激进等非理性行为

### 目标

在 **不修改模型结构、不重新训练** 的前提下：

* 让 MCTS 在不同比赛阶段做出不同风险偏好决策
* 显著减少“明显低级失误”（白球进袋 / 早黑8）
* 保证修改**工程风险低、可控、可解释**

---

## 2. 总体思路

### 核心原则

> **模型仍然只判断“局面好坏”，
> MCTS 负责根据比赛阶段决定“敢不敢赌”**

即：

* 网络 ≈ 静态评估器
* MCTS ≈ 动态风险管理器

---

## 3. 阶段定义（Phase Estimation）

### 3.1 阶段判定信号（任选其一或组合）

推荐优先级如下：

1. **剩余目标球数**
2. **是否进入 8-ball 决胜**
3. **MCTS 深度（作为 fallback）**

### 3.2 标准阶段划分（建议）

```text
Early Game（开局）
- 剩余球数 > 8

Mid Game（中盘）
- 剩余球数 ∈ [4, 8]

Late Game（终盘）
- 剩余球数 ≤ 3
- 或只剩黑8
```

### 3.3 阶段编码（工程建议）

```python
phase ∈ {EARLY, MID, LATE}
```

---

## 4. 核心修改模块（必须做）

### 4.1 阶段感知 Value 调整（最关键）

#### 修改位置

* Expansion / Evaluation 后
* Backprop 前

#### 原始逻辑

```python
value = model_value
```

#### 修改后逻辑

```python
value = model_value * phase_value_factor
```

#### 推荐参数

| 阶段    | phase_value_factor |
| ----- | ------------------ |
| Early | 0.7                |
| Mid   | 1.0                |
| Late  | 1.3 ~ 1.5          |

#### 直觉解释

* 开局：value 不确定 → 降低影响
* 终盘：value 决定性强 → 放大影响

---

### 4.2 阶段感知 UCB（cpuct）调整

#### 修改位置

`node.ucb()`

#### 原始逻辑

```python
Q + c_puct * P * sqrt(parent.N) / (1 + N)
```

#### 修改后逻辑

```python
Q + (c_puct * phase_c_puct_factor) * P * sqrt(parent.N) / (1 + N)
```

#### 推荐参数

| 阶段    | phase_c_puct_factor |
| ----- | ------------------- |
| Early | 1.2                 |
| Mid   | 1.0                 |
| Late  | 0.6 ~ 0.8           |

#### 效果

* 开局：探索更多策略
* 终盘：快速收敛，避免随机性失误

---

### 4.3 终局启发式价值兜底（强烈推荐）

#### 触发条件

* 只剩黑8
* 或一杆清台路径明显存在
* 或物理规则判定为“必赢 / 必败”

#### 修改逻辑

```python
if is_terminal_like_state:
    value = heuristic_value
```

#### 推荐 heuristic_value

| 局面   | value |
| ---- | ----- |
| 明显必赢 | 0.95  |
| 明显必败 | 0.05  |

📌 **这是防“蠢操作”的最后一道保险**

---

## 5. 次要但推荐的优化

### 5.1 阶段化搜索预算

```text
Early: n_simulations = 3~5
Mid:   n_simulations = 5~8
Late:  n_simulations = 10~15
```

只在 **Late Game** 增加模拟，成本低、收益高。

---

### 5.2 阶段化动作过滤阈值

* Early：prior threshold 高 → 保留多样性
* Late：prior threshold 低 → 剪枝激进动作

---

## 6. 工程实现建议（最小侵入）

### 6.1 新增工具函数

```python
def estimate_phase(state):
    remaining_balls = count_remaining_balls(state)
    if remaining_balls <= 3:
        return LATE
    elif remaining_balls <= 8:
        return MID
    else:
        return EARLY
```

---

### 6.2 参数集中管理（强烈建议）

```python
PHASE_PARAMS = {
    EARLY: { "value_factor": 0.7, "cpuct": 1.2 },
    MID:   { "value_factor": 1.0, "cpuct": 1.0 },
    LATE:  { "value_factor": 1.4, "cpuct": 0.7 }
}
```

避免 magic number 散落。

---

## 7. 验证方式（交付前必须做）

### 必做测试

1. **固定终局测试**

   * 给定 1~2 球局面
   * 观察是否避免无意义保守

2. **明显犯规测试**

   * 白球进袋
   * 提前击打黑8

3. **稳定性**

   * 同一局面多次运行动作是否一致

---

## 8. 风险与边界说明（必须让同伴知道）

### 已知风险

* phase 判定是启发式，不是理论完备
* value scaling 可能影响 value calibration

### 为什么是可接受的

* MCTS 本身是启发式搜索
* AlphaZero 工程中**大量使用阶段性调节**

---

## 9. 总结（给合作伙伴看的 TL;DR）

> 本方案通过 **阶段感知 MCTS**
> 在不修改模型的前提下补足比赛进程信息
> 核心机制是：
>
> * 阶段化 value 放大/压缩
> * 阶段化 UCB 探索强度
> * 终局启发式兜底
>   这是低风险、高收益、可快速验证的工程方案

---

如果你愿意，下一步我可以：

* 直接 **基于你当前 MCTS 代码给出 patch 级修改**
* 或帮你写一段 **给合作伙伴的“实现注意事项清单”**
* 或帮你 **估计哪一部分对胜率提升贡献最大**

你现在已经站在「能把系统调顺」的阶段了，这一步非常关键。
