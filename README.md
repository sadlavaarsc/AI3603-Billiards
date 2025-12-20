# 台球AI训练数据生成系统

本系统用于生成台球AI的训练数据，包括对局数据生成和训练数据处理两大模块。

## 功能概述

1. **对局数据生成**：通过模拟台球对局，生成包含球的状态和击球动作的对局数据文件
2. **训练数据处理**：将对局数据转换为神经网络训练所需的行为网络数据和价值网络数据

## 并行化支持

系统已支持异步并行生成和处理对局数据，适用于超算平台等多核心环境。通过设置不同脚本实例的ID范围，可以避免多核心并行时的文件读写冲突。

## 文件结构

- `generate_train_data.py`: 主流程脚本，协调对局数据生成和训练数据处理
- `generate_matches.py`: 对局数据生成脚本
- `process_match_data.py`: 训练数据处理脚本

## 使用说明

### 1. 主流程脚本 (generate_train_data.py)

```bash
python generate_train_data.py [参数]
```

**参数说明**：

- `--num_matches`: 生成的对局数量 (默认: 1000)
- `--match_dir`: 对局数据输出目录 (默认: ./match_data)
- `--behavior_output_dir`: 行为网络数据输出目录 (默认: ./training_data/behavior)
- `--value_output_dir`: 价值网络数据输出目录 (默认: ./training_data/value)
- `--start_id`: 对局起始ID (默认: 0)
- `--max_hit_count`: 每局最大击球次数 (默认: 50)
- `--verbose`: 是否显示详细输出 (默认: False)

### 2. 对局数据生成脚本 (generate_matches.py)

```bash
python generate_matches.py [参数]
```

**参数说明**：

- `--num_matches`: 生成的对局数量 (默认: 1000)
- `--output_dir`: 对局数据输出目录 (默认: ./match_data)
- `--start_id`: 对局起始ID (默认: 0)
- `--max_hit_count`: 每局最大击球次数 (默认: 50)
- `--verbose`: 是否显示详细输出 (默认: False)

### 3. 训练数据处理脚本 (process_match_data.py)

```bash
python process_match_data.py [参数]
```

**参数说明**：

- `--match_dir`: 对局数据目录 (默认: ./match_data)
- `--behavior_output_dir`: 行为网络数据输出目录 (默认: ./training_data/behavior)
- `--value_output_dir`: 价值网络数据输出目录 (默认: ./training_data/value)
- `--start_id`: 起始ID，用于文件名标识 (默认: None)
- `--end_id`: 结束ID，用于文件名标识 (默认: None)

## 并行处理示例

假设我们需要在25个核心上生成和处理100,000个对局数据，每个核心处理4,000个对局。

### 核心1:
```bash
python generate_train_data.py --num_matches 4000 --start_id 0
```

### 核心2:
```bash
python generate_train_data.py --num_matches 4000 --start_id 4000
```

### 核心3:
```bash
python generate_train_data.py --num_matches 4000 --start_id 8000
```

### ...以此类推

### 核心25:
```bash
python generate_train_data.py --num_matches 4000 --start_id 96000
```

## 注意事项

1. 每个核心生成的对局数据文件使用ID命名，如`match_000001.json`，确保不会与其他核心生成的文件冲突
2. 训练数据输出文件包含ID范围标识，如`behavior_network_data_0_3999_20231027_153045.json`
3. 确保输出目录已存在或可由脚本创建
4. 在超算平台上运行时，请根据平台特性调整命令执行方式
5. 处理完成后，可以将所有核心生成的训练数据合并用于模型训练

## 文件命名规则

1. **对局数据文件**：`match_[ID].json`，其中ID为6位数字，如`match_000123.json`
2. **训练数据文件**：
   - 行为网络：`behavior_network_data_[start_id]_[end_id]_[timestamp].json`
   - 价值网络：`value_network_data_[start_id]_[end_id]_[timestamp].json`

## 常见问题

1. **文件冲突**：确保每个并行实例使用不同的`start_id`参数值
2. **内存占用**：对于大量对局数据，可能需要增加内存或减小单次处理的对局数量
3. **磁盘空间**：生成大量对局数据时，请确保有足够的磁盘空间

## 版本历史

- 1.0: 初始版本
- 1.1: 添加并行化支持，允许多核心同时生成和处理数据