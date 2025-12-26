import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedFeatureExtractor(nn.Module):
    """
    共享特征提取器：提取连续三局状态向量的空间和时序特征
    输入：Batch × 3 × 81（连续三局状态向量）
    输出：Batch × 128（融合后的特征向量）
    """
    def __init__(self):
        super().__init__()
        # 空间特征提取：2层全连接网络
        self.spatial_fc1 = nn.Linear(81, 128)
        self.spatial_fc2 = nn.Linear(128, 128)
        self.layer_norm = nn.LayerNorm(128)
        
        # 时序特征融合：GRU层
        self.gru = nn.GRU(input_size=128, hidden_size=128, batch_first=True)
    
    def forward(self, x):
        # x形状: [batch_size, 3, 81]
        batch_size = x.size(0)
        
        # 空间特征提取
        spatial_features = []
        for i in range(3):  # 处理每一局
            single_game = x[:, i, :]  # 形状: [batch_size, 81]
            # 第一层全连接
            feat = F.relu(self.spatial_fc1(single_game))
            # 第二层全连接 + LayerNorm
            feat = self.spatial_fc2(feat)
            feat = self.layer_norm(F.relu(feat))
            spatial_features.append(feat)
        
        # 拼接三局空间特征：[batch_size, 3, 128]
        spatial_features = torch.stack(spatial_features, dim=1)
        
        # 时序特征融合
        # GRU输出: [batch_size, 3, 128]
        gru_out, _ = self.gru(spatial_features)
        
        # 取最后一局的特征 + 三局特征均值
        last_feat = gru_out[:, -1, :]  # 最后一局特征
        mean_feat = torch.mean(gru_out, dim=1)  # 三局平均特征
        
        # 融合特征
        fused_feat = last_feat + mean_feat  # 形状: [batch_size, 128]
        
        return fused_feat


class PolicyHead(nn.Module):
    """
    行为网络头：输出5维最优动作向量
    输入：Batch × 128（融合特征）
    输出：Batch × 5（0-1范围的动作参数）
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 5)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))  # 输出0-1范围
        return x
    
    def map_actions(self, raw_actions):
        """
        将0-1范围的原始输出映射到实际动作范围
        """
        actions = torch.zeros_like(raw_actions)
        # V0 (速度): 0.5 → 8.0
        actions[:, 0] = 0.5 + 7.5 * raw_actions[:, 0]
        # phi (水平角度): 0 → 360°
        actions[:, 1] = 360 * raw_actions[:, 1]
        # theta (垂直角度): 0 → 90°
        actions[:, 2] = 90 * raw_actions[:, 2]
        # a (x偏移): -0.5 → 0.5
        actions[:, 3] = -0.5 + 1.0 * raw_actions[:, 3]
        # b (y偏移): -0.5 → 0.5
        actions[:, 4] = -0.5 + 1.0 * raw_actions[:, 4]
        return actions


class ValueHead(nn.Module):
    """
    价值网络头：输出1维胜率
    输入：Batch × 128（融合特征）
    输出：Batch × 1（0-1范围的胜率）
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))  # 输出0-1范围的胜率
        return x


class DualNetwork(nn.Module):
    """
    双网络模型：行为网络 + 价值网络共享特征提取层
    输入：Batch × 3 × 81（连续三局状态向量）
    输出：
        - policy_output: Batch × 5（原始动作参数）
        - value_output: Batch × 1（胜率）
        - mapped_actions: Batch × 5（映射后的实际动作）
    """
    def __init__(self):
        super().__init__()
        # 共享特征提取器
        self.feature_extractor = SharedFeatureExtractor()
        # 行为网络头
        self.policy_head = PolicyHead()
        # 价值网络头
        self.value_head = ValueHead()
    
    def forward(self, x):
        # 提取共享特征
        features = self.feature_extractor(x)
        
        # 行为网络输出
        policy_output = self.policy_head(features)
        # 映射到实际动作范围
        mapped_actions = self.policy_head.map_actions(policy_output)
        
        # 价值网络输出
        value_output = self.value_head(features)
        
        return {
            'policy_output': policy_output,
            'value_output': value_output,
            'mapped_actions': mapped_actions
        }

    def save(self, path):
        """
        保存模型权重
        """
        torch.save({
            'feature_extractor': self.feature_extractor.state_dict(),
            'policy_head': self.policy_head.state_dict(),
            'value_head': self.value_head.state_dict()
        }, path)
    
    def load(self, path):
        """
        加载模型权重
        """
        checkpoint = torch.load(path)
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        self.policy_head.load_state_dict(checkpoint['policy_head'])
        self.value_head.load_state_dict(checkpoint['value_head'])
