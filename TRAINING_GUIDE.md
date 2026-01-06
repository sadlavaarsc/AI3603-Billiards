# 台球AI训练指南

## 1. 概述

本文档介绍如何使用项目中的训练脚本 `train.py` 来训练双网络模型（Dual Network）。该模型结合了策略网络（Policy Network）和价值网络（Value Network），用于预测台球游戏中的最优动作和胜率。

## 2. 训练流程

训练流程主要包括以下几个步骤：

1. **数据准备**：收集或生成台球对局数据
2. **数据处理**：使用 `process_raw_match_data.py` 将原始数据转换为可训练格式
3. **模型训练**：使用 `train.py` 训练双网络模型
4. **模型评估**：使用训练好的模型进行预测和评估

## 3. 环境要求

### 3.1 依赖库

- Python 3.8+
- PyTorch 2.0+
- NumPy
- tqdm

### 3.2 安装依赖

```bash
pip install torch numpy tqdm
```

## 4. 数据准备

### 4.1 数据格式

训练数据需要是符合格式的 JSON 文件，通常包含以下字段：
- `states`：连续3局的状态向量，每局81维
- `action`：5维动作向量 [V0, phi, theta, a, b]
- `value`：胜率标签（0-1）

### 4.2 生成训练数据

您可以使用以下方法之一获取训练数据：

#### 方法1：使用现有数据

将已有的对局数据文件（以 `match_` 开头，`.json` 结尾）放入一个目录中，例如 `match_data`。

#### 方法2：生成模拟数据

（注：目前 `generate_matches.py` 存在依赖问题，建议使用现有数据）

## 5. 运行训练

### 5.1 基本命令

```bash
python train.py --match_dir <对局数据目录> --model_dir <模型保存目录>
```

### 5.2 示例

```bash
python train.py --match_dir match_data --model_dir models
```

### 5.3 命令行参数说明

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `--match_dir` | str | `match_data` | 对局数据目录，包含原始数据文件 |
| `--train_data_file` | str | `trainable_data.json` | 处理后的训练数据输出文件 |
| `--model_dir` | str | `models` | 训练好的模型保存目录 |
| `--epochs` | int | `100` | 训练轮数 |
| `--batch_size` | int | `64` | 训练批次大小 |
| `--learning_rate` | float | `1e-4` | 初始学习率 |
| `--weight_decay` | float | `1e-5` | 权重衰减，用于正则化 |
| `--lr_step_size` | int | `50` | 学习率衰减的步长 |
| `--lr_gamma` | float | `0.5` | 学习率衰减因子 |
| `--save_interval` | int | `20` | 模型检查点保存间隔 |

## 6. 训练过程监控

训练过程中，脚本会输出以下信息：

1. **数据处理阶段**：
   - 处理的对局数据文件数量
   - 生成的训练样本数量

2. **训练阶段**：
   - 当前 epoch 和总 epoch 数
   - 每个批次的策略损失（Policy Loss）、价值损失（Value Loss）和总损失（Total Loss）
   - 每个 epoch 结束后的平均损失
   - 当前学习率
   - 模型检查点保存信息

3. **训练结束**：
   - 最终模型保存路径
   - 训练完成提示

## 7. 模型保存

训练过程中，模型会定期保存到指定的 `model_dir` 目录中：

- 每 `save_interval` 个 epoch 保存一个检查点：`dual_network_epoch_<epoch>.pt`
- 训练结束后保存最终模型：`dual_network_final.pt`

模型文件包含以下组件的权重：
- 共享特征提取器（SharedFeatureExtractor）
- 策略网络头（PolicyHead）
- 价值网络头（ValueHead）

## 8. 训练结果示例

```
Processing match data from match_data...
Training data generated: trainable_data.json
Loading training data...
Dataset size: 10000
Batch size: 64
Number of batches: 157
Initializing dual network model...
Using device: cuda
Starting training...
Epoch 1/100: 100%|██████████| 157/157 [00:10<00:00, 15.00it/s, Policy Loss=0.023456, Value Loss=0.012345, Total Loss=0.035801]
Epoch 1/100:
  Average Policy Loss: 0.025678
  Average Value Loss: 0.013456
  Average Total Loss: 0.039134
  Learning Rate: 0.0001
Epoch 2/100: 100%|██████████| 157/157 [00:09<00:00, 16.50it/s, Policy Loss=0.021345, Value Loss=0.010987, Total Loss=0.032332]
...
Model checkpoint saved: models/dual_network_epoch_20.pt
...
Final model saved: models/dual_network_final.pt
Training completed!
```

## 9. 使用训练好的模型

训练好的模型可以用于：

1. **动作预测**：给定当前状态，预测最优击球动作
2. **胜率评估**：评估当前状态下的胜率
3. **集成到台球AI系统**：用于实际游戏或模拟

### 9.1 模型加载示例

```python
from dual_network import DualNetwork

# 加载模型
model = DualNetwork()
model.load("models/dual_network_final.pt")

# 使用模型进行预测
# states 是一个形状为 [3, 81] 的状态向量
outputs = model(states)
policy_output = outputs['policy_output']  # 原始动作输出
mapped_actions = outputs['mapped_actions']  # 映射到实际范围的动作
value_output = outputs['value_output']  # 胜率预测
```

## 10. 训练调优建议

### 10.1 超参数调优

- **学习率**：如果损失下降缓慢，尝试增大学习率；如果损失震荡，尝试减小学习率
- **批次大小**：增大批次大小可以提高训练稳定性，但会增加内存占用
- **训练轮数**：根据数据集大小调整，避免过拟合

### 10.2 数据增强

- 增加训练数据量可以提高模型泛化能力
- 可以考虑对数据进行随机扰动或旋转

### 10.3 模型结构调整

- 可以尝试调整网络层数或隐藏层大小
- 可以尝试不同的激活函数或正则化方法

## 11. 常见问题

### 11.1 训练速度慢

- 确保使用了CUDA加速（检查输出中是否显示 `Using device: cuda`）
- 尝试增大批次大小
- 减少模型复杂度

### 11.2 损失不下降

- 检查学习率是否合适
- 检查数据格式是否正确
- 检查模型是否正确初始化

### 11.3 内存不足

- 减小批次大小
- 减少模型复杂度
- 使用更小的输入尺寸（如果可能）

## 12. 注意事项

1. 训练数据质量对模型性能至关重要，请确保数据格式正确
2. 建议定期保存模型检查点，以便在训练中断时恢复
3. 训练完成后，建议在验证集上评估模型性能
4. 可以尝试不同的超参数组合，找到最佳配置

## 13. 联系与支持

如有任何问题或建议，请联系项目维护者。

---

**版本**：1.0  
**最后更新**：2026-01-06