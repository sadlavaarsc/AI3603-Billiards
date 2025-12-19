# AI3603-Billiards
AI3603课程台球大作业

## 关键文件说明

| 文件 | 作用 | 在最终测试中是否可修改 |
|------|------|-----------|
| `poolenv.py` | 台球环境（游戏规则） | ❌ 不可修改 |
| `agent.py` | Agent 定义（在 `NewAgent` 中实现你的算法） | ✅ 可修改 `NewAgent` |
| `evaluate.py` | 评估脚本（运行对战） | ✅ 可修改 `agent_b` |
| `PROJECT_GUIDE.md` | 项目详细指南 | 📖 参考文档 |
| `GAME_RULES.md` | 游戏规则说明 | 📖 参考文档 |
| `generate_matches.py` | 自对弈对局数据生成脚本 | ✅ 可修改 |
| `process_match_data.py` | 对局数据处理脚本 | ✅ 可修改 |
| `main.py` | 数据生成主控制脚本 | ✅ 可修改 |

对作业内容的视频说明：
说明.mp4：https://pan.sjtu.edu.cn/web/share/da9459405eac6252d01c249c3bcb989f
供大家参考，以文字说明为准。

---

## 训练数据生成系统

本项目提供了一个完整的数据生成系统，用于通过贝叶斯AI自对弈生成台球AI训练数据，包括行为网络和价值网络的训练数据。

### 系统架构

系统由以下几个主要组件组成：

1. **generate_matches.py** - 自对弈对局数据生成脚本
   - 调用贝叶斯AI (BasicAgent) 进行自对弈
   - 记录完整的对局过程，包括状态、动作和结果
   - 支持参数控制对局数量和环境设置

2. **process_match_data.py** - 对局数据处理脚本
   - 加载原始对局数据
   - 转换为神经网络的输入特征（56维特征向量）
   - 生成行为网络和价值网络的训练数据

3. **main.py** - 主控制脚本
   - 整合数据生成和处理的完整流程
   - 提供灵活的参数配置选项
   - 支持测试模式和环境验证

### 使用方法

#### 1. 基本使用 - 完整流程

执行完整的数据生成和处理流程：

```bash
python main.py --num_matches 100 --enable_noise
```

#### 2. 测试模式

使用少量数据快速测试整个流程：

```bash
python main.py --test_mode
```

#### 3. 分步执行

仅生成对局数据：

```bash
python main.py --num_matches 50 --skip_processing
```

仅处理现有对局数据：

```bash
python main.py --skip_generation
```

#### 4. 自定义目录

指定自定义的数据保存目录：

```bash
python main.py --num_matches 100 \
    --match_dir "my_match_data" \
    --behavior_dir "my_training_data/behavior" \
    --value_dir "my_training_data/value"
```

### 参数说明

主要参数：
- `--num_matches` - 生成的对局数量（默认：10）
- `--test_mode` - 启用测试模式，使用少量数据快速测试
- `--enable_noise` - 启用动作噪声（默认启用）
- `--max_hit_count` - 每局最大击球次数（默认：200）
- `--skip_generation` - 跳过对局数据生成，直接处理现有数据
- `--skip_processing` - 跳过数据处理，只生成对局数据

### 数据格式

生成的数据包括：
- 对局数据 (match_*.json) - 完整的比赛记录
- 行为网络训练数据 (behavior_*.json) - 状态-动作对
- 价值网络训练数据 (value_*.json) - 状态-胜率-步数预测

每个状态使用56维特征向量表示，包含球的位置、速度和游戏状态等信息。