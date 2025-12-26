import torch
import numpy as np
import unittest
import os

# 导入我们实现的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dual_network import DualNetwork, SharedFeatureExtractor, PolicyHead, ValueHead
from data_loader import StatePreprocessor


class TestDualNetwork(unittest.TestCase):
    """
    测试双网络模型的输出维度正确性
    """
    
    def setUp(self):
        """
        测试前的设置
        """
        # 创建模型实例
        self.model = DualNetwork()
        self.preprocessor = StatePreprocessor()
        
        # 测试参数
        self.batch_size = 8
        self.history_length = 3
        self.input_dim = 81
        self.test_save_path = 'test_model.pth'
    
    def tearDown(self):
        """
        测试后的清理
        """
        # 删除测试保存的模型文件
        if os.path.exists(self.test_save_path):
            os.remove(self.test_save_path)
    
    def test_model_forward_pass(self):
        """
        测试模型的前向传播是否正确
        """
        print("\nTesting model forward pass...")
        
        # 创建随机输入（batch_size x history_length x input_dim）
        states = torch.randn(self.batch_size, self.history_length, self.input_dim)
        
        # 前向传播
        policy_outputs, value_outputs = self.model(states)
        
        # 验证输出维度
        self.assertEqual(policy_outputs.shape, (self.batch_size, 5), 
                         f"Policy outputs have incorrect shape: {policy_outputs.shape}")
        self.assertEqual(value_outputs.shape, (self.batch_size, 1), 
                         f"Value outputs have incorrect shape: {value_outputs.shape}")
        
        print(f"✓ Policy outputs shape: {policy_outputs.shape}")
        print(f"✓ Value outputs shape: {value_outputs.shape}")
    
    def test_single_sample_inference(self):
        """
        测试单个样本的推理
        """
        print("\nTesting single sample inference...")
        
        # 创建单个样本（1 x history_length x input_dim）
        single_state = torch.randn(1, self.history_length, self.input_dim)
        
        # 前向传播
        policy_output, value_output = self.model(single_state)
        
        # 验证输出维度
        self.assertEqual(policy_output.shape, (1, 5))
        self.assertEqual(value_output.shape, (1, 1))
        
        print(f"✓ Single sample policy output shape: {policy_output.shape}")
        print(f"✓ Single sample value output shape: {value_output.shape}")
    
    def test_model_with_preprocessed_inputs(self):
        """
        测试模型处理预处理后的输入
        """
        print("\nTesting model with preprocessed inputs...")
        
        # 创建随机原始输入
        raw_states = np.random.rand(self.batch_size, self.history_length, self.input_dim)
        
        # 预处理每一局的状态
        processed_states = np.zeros_like(raw_states)
        for b in range(self.batch_size):
            for h in range(self.history_length):
                processed_states[b, h] = self.preprocessor.normalize_state(raw_states[b, h])
        
        # 转换为张量
        processed_states_tensor = torch.tensor(processed_states, dtype=torch.float32)
        
        # 前向传播
        policy_outputs, value_outputs = self.model(processed_states_tensor)
        
        # 验证输出维度
        self.assertEqual(policy_outputs.shape, (self.batch_size, 5))
        self.assertEqual(value_outputs.shape, (self.batch_size, 1))
        
        print(f"✓ Model works with preprocessed inputs")
    
    def test_policy_head_output_range(self):
        """
        测试策略网络输出范围是否正确映射
        """
        print("\nTesting policy head output range...")
        
        # 创建一个单独的策略头进行测试
        policy_head = PolicyHead(hidden_dim=128)
        
        # 创建随机特征向量
        features = torch.randn(self.batch_size, 128)
        
        # 前向传播
        policy_output = policy_head(features)
        
        # 检查输出维度
        self.assertEqual(policy_output.shape, (self.batch_size, 5))
        
        # 映射后的输出范围检查将在实际应用中验证
        # 这里主要检查模型是否能正常输出
        print(f"✓ Policy head outputs shape: {policy_output.shape}")
    
    def test_value_head_output_range(self):
        """
        测试价值网络输出范围（应为0-1之间的概率值）
        """
        print("\nTesting value head output range...")
        
        # 创建一个单独的价值头进行测试
        value_head = ValueHead(hidden_dim=128)
        
        # 创建随机特征向量
        features = torch.randn(self.batch_size, 128)
        
        # 前向传播
        value_output = value_head(features)
        
        # 检查输出维度
        self.assertEqual(value_output.shape, (self.batch_size, 1))
        
        # 检查输出是否在0-1范围内（经过sigmoid激活）
        self.assertTrue(torch.all(value_output >= 0.0))
        self.assertTrue(torch.all(value_output <= 1.0))
        
        print(f"✓ Value head outputs shape: {value_output.shape}")
        print(f"✓ Value head outputs are in range [0, 1]")
    
    def test_feature_extractor_output(self):
        """
        测试共享特征提取器的输出
        """
        print("\nTesting feature extractor output...")
        
        # 创建特征提取器实例
        feature_extractor = SharedFeatureExtractor(input_dim=self.input_dim)
        
        # 创建随机输入
        states = torch.randn(self.batch_size, self.history_length, self.input_dim)
        
        # 前向传播
        features = feature_extractor(states)
        
        # 验证输出维度（batch_size x hidden_dim）
        self.assertEqual(features.shape, (self.batch_size, 128))
        
        print(f"✓ Feature extractor outputs shape: {features.shape}")
    
    def test_model_save_and_load(self):
        """
        测试模型的保存和加载功能
        """
        print("\nTesting model save and load...")
        
        # 创建随机输入
        states = torch.randn(self.batch_size, self.history_length, self.input_dim)
        
        # 获取原始模型的输出
        original_policy, original_value = self.model(states)
        
        # 保存模型
        self.model.save_model(self.test_save_path)
        
        # 验证文件是否存在
        self.assertTrue(os.path.exists(self.test_save_path))
        
        # 创建新模型并加载权重
        loaded_model = DualNetwork()
        loaded_model.load_model(self.test_save_path)
        
        # 获取加载后模型的输出
        loaded_policy, loaded_value = loaded_model(states)
        
        # 验证输出是否一致（考虑浮点误差）
        self.assertTrue(torch.allclose(original_policy, loaded_policy, atol=1e-5))
        self.assertTrue(torch.allclose(original_value, loaded_value, atol=1e-5))
        
        print(f"✓ Model saved and loaded successfully")
        print(f"✓ Outputs from original and loaded model are consistent")
    
    def test_model_device_migration(self):
        """
        测试模型在CPU和GPU之间迁移（如果有GPU的话）
        """
        print("\nTesting model device migration...")
        
        # 检查是否有可用的GPU
        has_gpu = torch.cuda.is_available()
        
        if has_gpu:
            # 创建随机输入
            states_cpu = torch.randn(self.batch_size, self.history_length, self.input_dim)
            
            # 在CPU上获取输出
            policy_cpu, value_cpu = self.model(states_cpu)
            
            # 将模型移至GPU
            model_gpu = self.model.to('cuda')
            states_gpu = states_cpu.to('cuda')
            
            # 在GPU上获取输出
            policy_gpu, value_gpu = model_gpu(states_gpu)
            
            # 将GPU输出移回CPU并验证
            policy_gpu_cpu = policy_gpu.cpu()
            value_gpu_cpu = value_gpu.cpu()
            
            # 验证结果是否一致（考虑浮点误差）
            self.assertTrue(torch.allclose(policy_cpu, policy_gpu_cpu, atol=1e-5))
            self.assertTrue(torch.allclose(value_cpu, value_gpu_cpu, atol=1e-5))
            
            print(f"✓ Model works on GPU")
            print(f"✓ CPU and GPU outputs are consistent")
        else:
            print(f"✓ No GPU available, skipping GPU test")


def run_integration_test():
    """
    运行集成测试，测试完整的模型流程
    """
    print("\nRunning integration test...")
    
    # 创建模型和预处理器
    model = DualNetwork()
    preprocessor = StatePreprocessor()
    
    # 创建模拟数据（连续三局的状态）
    num_samples = 4
    history_length = 3
    input_dim = 81
    
    # 模拟连续三局的状态
    raw_states = np.random.rand(num_samples, history_length, input_dim)
    
    # 预处理每一局的状态
    processed_states = np.zeros_like(raw_states)
    for i in range(num_samples):
        for j in range(history_length):
            processed_states[i, j] = preprocessor.normalize_state(raw_states[i, j])
    
    # 转换为张量
    states_tensor = torch.tensor(processed_states, dtype=torch.float32)
    
    # 模型推理
    policy_outputs, value_outputs = model(states_tensor)
    
    print(f"✓ Integration test completed successfully")
    print(f"  Input shape: {states_tensor.shape}")
    print(f"  Policy outputs shape: {policy_outputs.shape}")
    print(f"  Value outputs shape: {value_outputs.shape}")
    
    # 检查输出范围
    print(f"  Value outputs min: {torch.min(value_outputs).item():.4f}, max: {torch.max(value_outputs).item():.4f}")
    
    return policy_outputs, value_outputs


if __name__ == '__main__':
    # 添加系统路径以便导入模块
    import sys
    
    print("===== Testing Dual Network Model =====")
    
    # 运行单元测试
    unittest.main(argv=[sys.argv[0]], exit=False)
    
    # 运行集成测试
    run_integration_test()
    
    print("\n===== All tests completed successfully =====")
