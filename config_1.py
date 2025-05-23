# HMASD算法配置参数 - 基于论文《Hierarchical Multi-Agent Skill Discovery》附录E中的超参数设置

class Config:
    # 环境参数
    # 注意：实际环境中应该获取这些值
    n_agents = 10  # 无人机数量上限
    state_dim = None  # 全局状态维度（将在环境初始化时获取）
    obs_dim = None    # 单个智能体观测维度（将在环境初始化时获取）
    action_dim = 3    # 每个智能体输出3D速度向量

    # HMASD参数 - 基于论文Table 3中的3m场景
    n_Z = 3           # 团队技能数量 (论文中3m场景为3)
    n_z = 3           # 个体技能数量 (论文中3m场景为3)
    k = 10            # 技能分配间隔 (论文中3m场景为25，为适应无人机场景改为10)

    # 网络参数 - 基于论文Table 1
    hidden_size = 64         # 隐藏层大小 (论文中为64)
    embedding_dim = 64       # 嵌入维度 (与hidden_size保持一致)
    n_encoder_layers = 3     # 编码器层数
    n_decoder_layers = 3     # 解码器层数
    n_heads = 8              # 多头注意力头数
    gru_hidden_size = 64     # GRU隐藏层大小 (与hidden_size保持一致)
    lr_coordinator = 1e-4    # 技能协调器学习率 (论文Table 2中SMAC为1e-4)
    lr_discoverer = 1e-4     # 技能发现器学习率 (论文Table 2中SMAC为1e-4)
    lr_discriminator = 1e-4  # 技能判别器学习率 (论文Table 2中SMAC为1e-4)

    # PPO参数 - 基于论文Table 1
    gamma = 0.99             # 折扣因子
    gae_lambda = 0.95        # GAE参数
    clip_epsilon = 0.2       # PPO裁剪参数
    ppo_epochs = 15          # PPO迭代次数
    value_loss_coef = 1.0    # 价值损失系数 (论文中为1.0)
    max_grad_norm = 0.5      # 最大梯度范数

    # HMASD损失权重 - 基于论文Table 3中的3m场景
    # 注意：lambda_e参数已调整为100.0，与论文中一致
    lambda_e = 100.0       # 外部奖励权重 (论文中3m场景为100)
    lambda_D = 0.1           # 团队技能判别器奖励权重 (论文中3m场景为0.1)
    lambda_d = 0.5           # 个体技能判别器奖励权重 (论文中3m场景为0.5)
    lambda_h = 0.001         # 高层策略熵权重 (论文中3m场景为0.001)
    lambda_l = 0.01          # 低层策略熵权重 (论文中3m场景为0.01)

    # 训练参数 - 部分基于论文Table 1和Table 2
    buffer_size = 1024       # 经验回放缓冲区大小
    batch_size = 128         # 批处理大小
    high_level_batch_size = 128  # 高层更新的批处理大小
    num_envs = 32            # 并行环境数量 (论文中rollout threads为32)
    total_timesteps = 2e6    # 总时间步数 (论文中SMAC为2e6)
    eval_interval = 160000   # 评估间隔 (32并行环境 * 每环境5000步)
    episode_length = 5000    # 每个episode的最大长度 (基于观察到的实际行为)
    eval_episodes = 4      # 评估时的episode数量 (论文中SMAC为100)
    eval_rollout_threads = 4 # 评估时的并行线程数 (论文中SMAC为4)
    
    # 其他论文中提到的参数
    use_valuenorm = True     # 使用价值标准化
    use_orthogonal = True    # 使用正交初始化
    gain = 0.01              # 增益
    optimizer_epsilon = 1e-5 # 优化器epsilon
    weight_decay = 0         # 权重衰减
    num_mini_batch = 1       # mini batch数量
    use_huber_loss = True    # 使用Huber损失
    huber_delta = 10         # Huber delta

    def update_env_dims(self, state_dim, obs_dim):
        """更新环境维度"""
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        print(f"环境维度已更新：state_dim={state_dim}, obs_dim={obs_dim}")
