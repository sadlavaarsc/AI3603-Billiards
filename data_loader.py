import numpy as np
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader


class BilliardsDataset(Dataset):
    """
    台球数据集：处理连续三局状态向量的数据集类
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = self._load_data()
    
    def _load_data(self):
        """
        加载数据目录中的所有行为和价值数据文件
        """
        all_data = []
        
        # 遍历数据目录
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.data_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_data.extend(data)
                        else:
                            all_data.append(data)
                except Exception as e:
                    print(f"Error loading file {filename}: {e}")
        
        return all_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        获取单个数据项
        """
        item = self.data[idx]
        
        # 提取状态、动作和价值标签
        # 注意：这里假设数据格式包含连续三局的状态
        states = np.array(item['states'], dtype=np.float32)
        policy_targets = np.array(item['action'], dtype=np.float32) if 'action' in item else None
        value_targets = np.array([item['value']], dtype=np.float32) if 'value' in item else None
        
        # 应用变换
        if self.transform:
            states = self.transform(states)
        
        return states, policy_targets, value_targets


class StatePreprocessor:
    """
    状态预处理器：处理连续三局状态向量的归一化和格式化
    """
    def __init__(self, table_width=2.845, table_length=1.4225):
        """
        参数：
            table_width: 标准球桌宽度（米）
            table_length: 标准球桌长度（米）
        """
        self.table_width = table_width
        self.table_length = table_length
    
    def normalize_state(self, state):
        """
        归一化单个局的状态向量
        
        参数：
            state: 81维状态向量
        
        返回：
            归一化后的状态向量
        """
        state = state.copy()
        
        # 球坐标归一化（0-63维：16球×4特征）
        # x坐标（0, 4, 8, ..., 60）归一化到0-1
        for i in range(0, 64, 4):
            if state[i] != -1:  # 如果不是进袋状态
                state[i] = state[i] / self.table_width
        
        # y坐标（1, 5, 9, ..., 61）归一化到0-1
        for i in range(1, 64, 4):
            if state[i] != -1:  # 如果不是进袋状态
                state[i] = state[i] / self.table_length
        
        # z坐标（2, 6, 10, ..., 62）归一化（相对球直径）
        ball_radius = 0.0285  # 标准台球半径（米）
        for i in range(2, 64, 4):
            if state[i] != -1:  # 如果不是进袋状态
                state[i] = state[i] / (2 * ball_radius)  # 归一化到-1到1
        
        # 进袋标志（3, 7, 11, ..., 63）保持不变（0或1）
        
        # 球桌尺寸归一化（64-65维）
        state[64] = state[64] / self.table_width  # 宽度归一化
        state[65] = state[65] / self.table_length  # 长度归一化
        
        # 目标球特征（66-80维）保持不变（二进制值）
        
        return state
    
    def process_three_game_states(self, three_game_states):
        """
        处理连续三局的状态向量
        
        参数：
            three_game_states: 形状为[3, 81]的数组，表示连续三局的状态
        
        返回：
            处理后的三维状态数组
        """
        # 确保输入形状正确
        if three_game_states.shape != (3, 81):
            raise ValueError(f"Expected shape (3, 81), got {three_game_states.shape}")
        
        # 对每一局进行归一化
        processed_states = np.zeros_like(three_game_states)
        for i in range(3):
            processed_states[i] = self.normalize_state(three_game_states[i])
        
        return processed_states
    
    def __call__(self, states):
        """
        支持直接调用进行处理
        """
        return self.process_three_game_states(states)


def create_combined_dataset(behavior_data_dir, value_data_dir, preprocessor=None):
    """
    创建组合数据集，合并行为和价值网络的数据
    
    参数：
        behavior_data_dir: 行为数据目录
        value_data_dir: 价值数据目录
        preprocessor: 预处理器实例
    
    返回：
        组合后的数据集
    """
    # 加载行为数据
    behavior_data = []
    for filename in os.listdir(behavior_data_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(behavior_data_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        behavior_data.extend(data)
                    else:
                        behavior_data.append(data)
            except Exception as e:
                print(f"Error loading behavior file {filename}: {e}")
    
    # 加载价值数据
    value_data = []
    for filename in os.listdir(value_data_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(value_data_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        value_data.extend(data)
                    else:
                        value_data.append(data)
            except Exception as e:
                print(f"Error loading value file {filename}: {e}")
    
    # 创建状态ID到价值标签的映射
    state_to_value = {}
    for item in value_data:
        if 'states' in item and 'value' in item:
            # 使用状态向量的哈希作为键（简化处理）
            state_key = hash(np.array(item['states']).tobytes())
            state_to_value[state_key] = item['value']
    
    # 合并数据集
    combined_data = []
    for item in behavior_data:
        if 'states' in item and 'action' in item:
            combined_item = {
                'states': item['states'],
                'action': item['action']
            }
            
            # 添加价值标签（如果存在）
            state_key = hash(np.array(item['states']).tobytes())
            if state_key in state_to_value:
                combined_item['value'] = state_to_value[state_key]
            else:
                # 如果没有对应的价值标签，可以设置默认值或跳过
                combined_item['value'] = 0.5  # 默认胜率为50%
            
            combined_data.append(combined_item)
    
    # 保存合并后的数据
    output_file = os.path.join(os.path.dirname(behavior_data_dir), 'combined_data.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)
    
    print(f"Combined dataset saved to {output_file}")
    print(f"Total samples: {len(combined_data)}")
    
    return combined_data


def create_dataloader(data, batch_size=32, shuffle=True, preprocessor=None):
    """
    创建数据加载器
    
    参数：
        data: 包含states, action, value字段的数据列表
        batch_size: 批次大小
        shuffle: 是否打乱数据
        preprocessor: 预处理器实例
    
    返回：
        DataLoader实例
    """
    class CombinedDataset(Dataset):
        def __init__(self, data_list, preprocessor=None):
            self.data = data_list
            self.preprocessor = preprocessor
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            item = self.data[idx]
            states = np.array(item['states'], dtype=np.float32)
            
            # 应用预处理
            if self.preprocessor:
                states = self.preprocessor(states)
            
            policy_targets = np.array(item['action'], dtype=np.float32)
            value_targets = np.array([item['value']], dtype=np.float32)
            
            return states, policy_targets, value_targets
    
    dataset = CombinedDataset(data, preprocessor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )


def generate_sample_data(output_dir='sample_data', num_samples=100):
    """
    生成样本数据用于测试
    
    参数：
        output_dir: 输出目录
        num_samples: 样本数量
    """
    # 创建输出目录
    behavior_dir = os.path.join(output_dir, 'behavior')
    value_dir = os.path.join(output_dir, 'value')
    os.makedirs(behavior_dir, exist_ok=True)
    os.makedirs(value_dir, exist_ok=True)
    
    # 生成行为数据
    behavior_data = []
    for i in range(num_samples):
        # 生成连续三局的状态向量
        three_game_states = np.random.rand(3, 81)
        
        # 模拟球桌尺寸（归一化到0.8-1.2倍标准尺寸）
        for game in range(3):
            three_game_states[game, 64] = 2.845 * np.random.uniform(0.8, 1.2)  # 宽度
            three_game_states[game, 65] = 1.4225 * np.random.uniform(0.8, 1.2)  # 长度
        
        # 生成动作向量
        action = np.random.rand(5)
        
        behavior_item = {
            'states': three_game_states.tolist(),
            'action': action.tolist(),
            'history_length': 3
        }
        behavior_data.append(behavior_item)
    
    # 保存行为数据
    behavior_file = os.path.join(behavior_dir, 'sample_behavior.json')
    with open(behavior_file, 'w', encoding='utf-8') as f:
        json.dump(behavior_data, f, ensure_ascii=False, indent=2)
    
    # 生成价值数据
    value_data = []
    for i in range(num_samples):
        # 复用行为数据中的状态
        if i < len(behavior_data):
            states = behavior_data[i]['states']
        else:
            states = np.random.rand(3, 81).tolist()
        
        # 生成胜负标签（0或1）
        value = np.random.randint(0, 2)
        
        value_item = {
            'states': states,
            'value': value,
            'history_length': 3
        }
        value_data.append(value_item)
    
    # 保存价值数据
    value_file = os.path.join(value_dir, 'sample_value.json')
    with open(value_file, 'w', encoding='utf-8') as f:
        json.dump(value_data, f, ensure_ascii=False, indent=2)
    
    print(f"Sample data generated in {output_dir}")
    print(f"Behavior samples: {len(behavior_data)}")
    print(f"Value samples: {len(value_data)}")


# 示例使用
if __name__ == "__main__":
    # 生成样本数据
    generate_sample_data()
    
    # 创建预处理器
    preprocessor = StatePreprocessor()
    
    # 合并数据集
    combined_data = create_combined_dataset(
        'sample_data/behavior',
        'sample_data/value',
        preprocessor
    )
    
    # 创建数据加载器
    dataloader = create_dataloader(combined_data, batch_size=8, preprocessor=preprocessor)
    
    # 测试数据加载器
    for states, policy_targets, value_targets in dataloader:
        print(f"States shape: {states.shape}")
        print(f"Policy targets shape: {policy_targets.shape}")
        print(f"Value targets shape: {value_targets.shape}")
        break  # 仅测试一个批次
