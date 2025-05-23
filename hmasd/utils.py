import torch
import numpy as np
from collections import deque
import random

class ReplayBuffer:
    """经验回放缓冲区，用于存储和采样训练数据"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self._total_added = 0
        self._total_sampled = 0
        self._structure_validated = False
    
    def push(self, experience):
        """
        将经验存入缓冲区
        
        参数:
            experience: 经验元组，或参数列表(通过*args收集的多个参数)
        """
        # 如果传入的是多个参数，自动打包为元组
        if not isinstance(experience, tuple):
            experience = (experience,)
        
        # 记录添加计数
        if len(self.buffer) >= self.capacity:
            self._total_added += 1
        
        self.buffer.append(experience)
        
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
        self._total_added = 0
        self._total_sampled = 0
        self._structure_validated = False
    
    def sample(self, batch_size):
        """从缓冲区中随机采样一批经验"""
        sampled_batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        self._total_sampled += len(sampled_batch)
        
        # 验证样本结构
        if not self._structure_validated and sampled_batch:
            sample_structure = len(sampled_batch[0])
            print(f"缓冲区样本结构: 包含 {sample_structure} 个元素")
            self._structure_validated = True
            
        return sampled_batch
    
    def __len__(self):
        """返回缓冲区的当前大小"""
        return len(self.buffer)
    
    def get_stats(self):
        """获取缓冲区统计信息"""
        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "total_added": self._total_added,
            "total_sampled": self._total_sampled,
            "utilization": len(self.buffer) / self.capacity if self.capacity > 0 else 0
        }

class StateSkillDataset:
    """状态-技能对数据集，用于训练技能判别器"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self._total_added = 0
        self._total_sampled = 0
    
    def push(self, state, team_skill, observations, agent_skills):
        """将状态-技能对存入数据集"""
        experience = (state, team_skill, observations, agent_skills)
        
        # 记录添加计数
        if len(self.buffer) >= self.capacity:
            self._total_added += 1
            
        self.buffer.append(experience)
        
    def clear(self):
        """清空数据集"""
        self.buffer.clear()
        self._total_added = 0
        self._total_sampled = 0
    
    def sample(self, batch_size):
        """从数据集中随机采样一批数据"""
        sampled_batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        self._total_sampled += len(sampled_batch)
        return sampled_batch
    
    def __len__(self):
        """返回数据集的当前大小"""
        return len(self.buffer)
        
    def get_stats(self):
        """获取数据集统计信息"""
        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "total_added": self._total_added,
            "total_sampled": self._total_sampled,
            "utilization": len(self.buffer) / self.capacity if self.capacity > 0 else 0
        }

def compute_gae(rewards, values, next_values, dones, gamma, lam):
    """
    计算广义优势估计（GAE）
    
    参数:
        rewards: 一批奖励 [batch_size]
        values: 当前状态价值 [batch_size]
        next_values: 下一状态价值 [batch_size]
        dones: 终止标志 [batch_size]
        gamma: 折扣因子
        lam: GAE参数
        
    返回:
        advantages: 优势函数估计值 [batch_size]
        returns: 目标收益值 [batch_size]
    """
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    
    # 逆序遍历时序数据进行计算
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = next_values[t]
        else:
            next_value = values[t + 1]
            
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
    
    returns = advantages + values
    
    return advantages, returns

def compute_ppo_loss(policy, values, old_log_probs, actions, advantages, returns, 
                     clip_epsilon, entropy_coef, value_loss_coef):
    """
    计算PPO损失函数
    
    参数:
        policy: 策略分布 [batch_size, ...]
        values: 价值函数 [batch_size]
        old_log_probs: 旧策略的动作对数概率 [batch_size]
        actions: 执行的动作 [batch_size, action_dim]
        advantages: 优势函数值 [batch_size]
        returns: 目标收益值 [batch_size]
        clip_epsilon: PPO裁剪参数
        entropy_coef: 熵损失系数
        value_loss_coef: 价值损失系数
        
    返回:
        loss: 总损失值
        policy_loss: 策略损失值
        value_loss: 价值损失值
        entropy_loss: 熵损失值
    """
    # 计算当前策略对动作的对数概率
    dist = policy
    log_probs = dist.log_prob(actions)
    
    # 计算策略比率并限制在[1-epsilon, 1+epsilon]范围内
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # 计算价值损失（均方误差）
    value_loss = 0.5 * ((returns - values) ** 2).mean()
    
    # 计算熵，鼓励探索
    entropy_loss = -dist.entropy().mean()
    
    # 加权组合三个损失
    loss = policy_loss + value_loss_coef * value_loss + entropy_coef * entropy_loss
    
    return loss, policy_loss, value_loss, entropy_loss

def one_hot(indices, depth):
    """
    将索引转换为独热编码
    
    参数:
        indices: 索引张量 [batch_size]
        depth: 独热编码的维度
        
    返回:
        one_hot: 独热编码张量 [batch_size, depth]
    """
    if isinstance(indices, int):
        indices = torch.tensor([indices])
    elif isinstance(indices, list):
        indices = torch.tensor(indices)
    if indices.dim() == 0:
        indices = indices.unsqueeze(0)
    
    device = indices.device
    one_hot = torch.zeros(indices.size(0), depth, device=device)
    one_hot.scatter_(1, indices.unsqueeze(1), 1)
    
    return one_hot

def setup_optimizer(model, lr):
    """设置优化器"""
    return torch.optim.Adam(model.parameters(), lr=lr)
