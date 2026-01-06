import json
import numpy as np
from data_loader import StatePreprocessor

def test_data_format():
    """测试生成的数据格式是否正确"""
    # 1. 加载生成的数据
    with open('trainable_data.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    print(f"Loaded {len(train_data)} samples")
    
    # 2. 测试单个样本的数据格式
    sample = train_data[0]
    print("\nTesting sample format:")
    print(f"  Sample keys: {list(sample.keys())}")
    
    # 验证样本包含所有必要的键
    assert 'states' in sample, "Sample must contain 'states' key"
    assert 'action' in sample, "Sample must contain 'action' key"
    assert 'value' in sample, "Sample must contain 'value' key"
    assert 'history_length' in sample, "Sample must contain 'history_length' key"
    
    # 3. 验证states格式
    states = sample['states']
    print(f"  States type: {type(states)}, length: {len(states)}")
    
    # 验证states是3个81维向量
    assert len(states) == 3, f"Expected 3 states, got {len(states)}"
    for i, state in enumerate(states):
        assert len(state) == 81, f"State {i+1} should be 81-dimensional, got {len(state)}"
    
    # 4. 验证action格式
    action = sample['action']
    print(f"  Action type: {type(action)}, length: {len(action)}")
    assert len(action) == 5, f"Expected action to be 5-dimensional, got {len(action)}"
    
    # 5. 验证value格式
    value = sample['value']
    print(f"  Value type: {type(value)}, value: {value}")
    assert isinstance(value, (int, float)), f"Value should be a number, got {type(value)}"
    assert 0.0 <= value <= 1.0, f"Value should be between 0.0 and 1.0, got {value}"
    
    # 6. 验证history_length
    history_length = sample['history_length']
    print(f"  History length: {history_length}")
    assert history_length == 3, f"Expected history_length to be 3, got {history_length}"
    
    # 7. 测试预处理器
    print("\nTesting StatePreprocessor:")
    preprocessor = StatePreprocessor()
    
    # 将states转换为numpy数组
    states_np = np.array(states, dtype=np.float32)
    print(f"  Original states shape: {states_np.shape}")
    
    # 应用预处理器
    processed_states = preprocessor(states_np)
    print(f"  Processed states shape: {processed_states.shape}")
    assert processed_states.shape == (3, 81), f"Expected processed states shape (3, 81), got {processed_states.shape}"
    
    # 验证处理后的数据值范围
    print(f"  Min value: {processed_states.min()}, Max value: {processed_states.max()}")
    
    print("\n✅ All tests passed! Generated data is in the correct format!")

if __name__ == "__main__":
    test_data_format()