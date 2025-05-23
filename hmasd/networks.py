import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal, Categorical
import logging
from logger import main_logger

class MLP(nn.Module):
    """多层感知机"""
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2):
        super(MLP, self).__init__()
        
        layers = []
        dims = [input_dim] + [hidden_dim] * n_layers + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # 不在最后一层应用激活函数
                layers.append(nn.ReLU())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        # 确保输入是float32类型
        x = x.float()
        return self.model(x)

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        self.d_model = d_model
    
    def forward(self, x):
        # 确保输入是float32类型
        x = x.float()
        x = x + self.pe[:, :x.size(1)]
        return x

class StateEncoder(nn.Module):
    """状态编码器"""
    def __init__(self, state_dim, obs_dim, embedding_dim, n_layers, n_heads):
        super(StateEncoder, self).__init__()
        
        self.state_embedding = None  # 将在forward方法中根据实际输入维度初始化
        self.obs_embedding = None  # 将在forward方法中根据实际输入维度初始化
        self.embedding_dim = embedding_dim
        self.positional_encoding = PositionalEncoding(embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=embedding_dim * 4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)
    
    def forward(self, state, observations):
        """
        参数:
            state: 全局状态 [batch_size, state_dim]
            observations: 所有智能体观测 [batch_size, n_agents, obs_dim]
            
        返回:
            encoded_state: 编码后的状态 [batch_size, 1, embedding_dim]
            encoded_observations: 编码后的观测 [batch_size, n_agents, embedding_dim]
        """
        batch_size, n_agents, obs_dim = observations.size()
        state_dim = state.size(-1)
        
        # 确保输入是float32类型
        state = state.float()
        observations = observations.float()
        
        # 根据实际输入维度初始化嵌入层（如果尚未初始化）
        if self.state_embedding is None:
            main_logger.info(f"初始化state_embedding: 实际状态维度 = {state_dim}")
            self.state_embedding = nn.Linear(state_dim, self.embedding_dim)
            self.state_embedding = self.state_embedding.to(state.device)
            
        if self.obs_embedding is None:
            main_logger.info(f"初始化obs_embedding: 实际观测维度 = {obs_dim}")
            self.obs_embedding = nn.Linear(obs_dim, self.embedding_dim)
            self.obs_embedding = self.obs_embedding.to(observations.device)
        
        # 嵌入全局状态和局部观测
        embedded_state = self.state_embedding(state).unsqueeze(1)  # [batch_size, 1, embedding_dim]
        embedded_obs = self.obs_embedding(observations.reshape(-1, obs_dim))
        embedded_obs = embedded_obs.reshape(batch_size, n_agents, -1)  # [batch_size, n_agents, embedding_dim]
        
        # 将状态和观测拼接作为序列
        sequence = torch.cat([embedded_state, embedded_obs], dim=1)  # [batch_size, 1+n_agents, embedding_dim]
        
        # 位置编码
        sequence = self.positional_encoding(sequence)
        
        # Transformer编码器
        encoded_sequence = self.transformer_encoder(sequence)
        
        # 拆分回状态和观测
        encoded_state = encoded_sequence[:, 0:1, :]
        encoded_observations = encoded_sequence[:, 1:, :]
        
        return encoded_state, encoded_observations

class SkillDecoder(nn.Module):
    """技能解码器"""
    def __init__(self, embedding_dim, n_layers, n_heads, n_Z, n_z):
        super(SkillDecoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.Z0_embedding = nn.Embedding(1, embedding_dim)
        self.team_skill_embedding = nn.Embedding(n_Z, embedding_dim)
        self.agent_skill_embedding = nn.Embedding(n_z, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=embedding_dim * 4,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, n_layers)
        
        # 输出头
        self.team_skill_head = nn.Linear(embedding_dim, n_Z)
        self.agent_skill_head = nn.Linear(embedding_dim, n_z)
    
    def forward(self, encoded_state, encoded_observations, Z=None, z=None, step=0):
        """
        参数:
            encoded_state: 编码后的状态 [batch_size, 1, embedding_dim]
            encoded_observations: 编码后的观测 [batch_size, n_agents, embedding_dim]
            Z: 已选择的团队技能索引 [batch_size]，可选
            z: 已选择的个体技能索引列表 [batch_size, step]，可选
            step: 当前解码步骤
            
        返回:
            output: 技能分布 [batch_size, n_Z/n_z]
        """
        batch_size = encoded_state.size(0)
        device = encoded_state.device
        
        if step == 0:  # 生成团队技能Z
            # 使用特殊起始符Z0
            Z0_idx = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            decoder_input = self.Z0_embedding(Z0_idx)
            decoder_input = self.positional_encoding(decoder_input)
            
            # Transformer解码
            memory = torch.cat([encoded_state, encoded_observations], dim=1)
            decoded = self.transformer_decoder(decoder_input, memory)
            
            # 输出团队技能分布
            team_skill_logits = self.team_skill_head(decoded).squeeze(1)
            
            # 记录团队技能logits的统计信息
            with torch.no_grad():
                is_nan = torch.isnan(team_skill_logits).any().item()
                is_inf = torch.isinf(team_skill_logits).any().item()
                logits_mean = team_skill_logits.mean().item()
                logits_std = team_skill_logits.std().item()
                logits_min = team_skill_logits.min().item()
                logits_max = team_skill_logits.max().item()
                main_logger.debug(f"团队技能logits统计: 均值={logits_mean:.4f}, 标准差={logits_std:.4f}, "
                      f"最小值={logits_min:.4f}, 最大值={logits_max:.4f}, "
                      f"含NaN={is_nan}, 含Inf={is_inf}")
                
                # 如果检测到NaN或Inf，输出更详细的信息
                if is_nan or is_inf:
                    main_logger.warning("警告: 团队技能logits包含NaN或Inf值！")
                    main_logger.warning(f"logits形状: {team_skill_logits.shape}")
                    main_logger.warning(f"NaN位置: {torch.isnan(team_skill_logits).nonzero()}")
                    main_logger.warning(f"Inf位置: {torch.isinf(team_skill_logits).nonzero()}")
                    
                # 检测是否有极端值，可能导致数值不稳定
                extreme_threshold = 50.0  # 定义极端值阈值
                has_extreme = (torch.abs(team_skill_logits) > extreme_threshold).any().item()
                if has_extreme:
                    main_logger.warning(f"警告: 团队技能logits存在绝对值大于{extreme_threshold}的极端值!")
                    extreme_indices = (torch.abs(team_skill_logits) > extreme_threshold).nonzero()
                    extreme_values = team_skill_logits[extreme_indices[:, 0], extreme_indices[:, 1]]
                    main_logger.warning(f"极端值示例 (最多10个): {extreme_values[:10].tolist()}")
                    
            # 应用数值稳定性措施，裁剪极端值
            clip_threshold = 50.0  # 定义裁剪阈值
            team_skill_logits = torch.clamp(team_skill_logits, min=-clip_threshold, max=clip_threshold)
            
            return team_skill_logits
        else:  # 生成第step个智能体的个体技能zi
            # 构建已解码序列
            seq_len = step + 1  # Z0 + Z + z1 + ... + z_{step-1}
            decoder_inputs = []
            
            # 添加Z0
            Z0_idx = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            decoder_inputs.append(self.Z0_embedding(Z0_idx))
            
            # 添加Z，使用clone()创建新张量，防止原地修改导致自动求导错误
            Z_clone = Z.clone().detach()
            Z_embedded = self.team_skill_embedding(Z_clone.unsqueeze(1))
            decoder_inputs.append(Z_embedded)
            
            # 添加z1到z_{step-1}
            for i in range(step - 1):
                # 使用clone()创建新张量，防止原地修改导致自动求导错误
                z_i_clone = z[:, i].clone().detach()
                zi_embedded = self.agent_skill_embedding(z_i_clone.unsqueeze(1))
                decoder_inputs.append(zi_embedded)
            
            # 拼接所有嵌入
            decoder_input = torch.cat(decoder_inputs, dim=1)
            decoder_input = self.positional_encoding(decoder_input)
            
            # Transformer解码
            memory = torch.cat([encoded_state, encoded_observations], dim=1)
            decoded = self.transformer_decoder(decoder_input, memory)
            
            # 输出个体技能分布（仅取最后一步）
            agent_skill_logits = self.agent_skill_head(decoded[:, -1, :])
            
            # 记录个体技能logits的统计信息
            with torch.no_grad():
                is_nan = torch.isnan(agent_skill_logits).any().item()
                is_inf = torch.isinf(agent_skill_logits).any().item()
                logits_mean = agent_skill_logits.mean().item()
                logits_std = agent_skill_logits.std().item()
                logits_min = agent_skill_logits.min().item()
                logits_max = agent_skill_logits.max().item()
                main_logger.debug(f"智能体{step-1}技能logits统计: 均值={logits_mean:.4f}, 标准差={logits_std:.4f}, "
                      f"最小值={logits_min:.4f}, 最大值={logits_max:.4f}, "
                      f"含NaN={is_nan}, 含Inf={is_inf}")
                
                # 如果检测到NaN或Inf，输出更详细的信息
                if is_nan or is_inf:
                    main_logger.warning("警告: 个体技能logits包含NaN或Inf值！")
                    main_logger.warning(f"logits形状: {agent_skill_logits.shape}")
                    main_logger.warning(f"NaN位置: {torch.isnan(agent_skill_logits).nonzero()}")
                    main_logger.warning(f"Inf位置: {torch.isinf(agent_skill_logits).nonzero()}")
                    
                # 检测是否有极端值，可能导致数值不稳定
                extreme_threshold = 50.0  # 定义极端值阈值
                has_extreme = (torch.abs(agent_skill_logits) > extreme_threshold).any().item()
                if has_extreme:
                    main_logger.warning(f"警告: 个体技能logits存在绝对值大于{extreme_threshold}的极端值!")
                    extreme_indices = (torch.abs(agent_skill_logits) > extreme_threshold).nonzero()
                    extreme_values = agent_skill_logits[extreme_indices[:, 0], extreme_indices[:, 1]]
                    main_logger.warning(f"极端值示例 (最多10个): {extreme_values[:10].tolist()}")
            
            # 应用数值稳定性措施，裁剪极端值
            clip_threshold = 50.0  # 定义裁剪阈值
            agent_skill_logits = torch.clamp(agent_skill_logits, min=-clip_threshold, max=clip_threshold)
            
            return agent_skill_logits

class SkillCoordinator(nn.Module):
    """技能协调器（高层策略）"""
    def __init__(self, config):
        super(SkillCoordinator, self).__init__()
        
        self.config = config
        self.n_Z = config.n_Z
        self.n_z = config.n_z
        
        # 状态编码器
        self.state_encoder = StateEncoder(
            config.state_dim,
            config.obs_dim,
            config.embedding_dim,
            config.n_encoder_layers,
            config.n_heads
        )
        
        # 技能解码器
        self.skill_decoder = SkillDecoder(
            config.embedding_dim,
            config.n_decoder_layers,
            config.n_heads,
            config.n_Z,
            config.n_z
        )
        
        # 高层价值函数
        self.value_head_state = nn.Linear(config.embedding_dim, 1)
        self.value_heads_obs = nn.ModuleList([
            nn.Linear(config.embedding_dim, 1) for _ in range(config.n_agents)
        ])
    
    def get_value(self, state, observations):
        """获取高层价值函数值"""
        encoded_state, encoded_observations = self.state_encoder(state, observations)
        
        # 全局状态价值
        state_value = self.value_head_state(encoded_state.squeeze(1))
        
        # 每个智能体的观测价值
        agent_values = []
        for i in range(min(self.config.n_agents, encoded_observations.size(1))):
            agent_value = self.value_heads_obs[i](encoded_observations[:, i, :])
            agent_values.append(agent_value)
            
        return state_value, agent_values
    
    def forward(self, state, observations, deterministic=False):
        """
        前向传播，按顺序生成技能
        
        参数:
            state: 全局状态 [batch_size, state_dim]
            observations: 所有智能体观测 [batch_size, n_agents, obs_dim]
            deterministic: 是否使用确定性策略
            
        返回:
            Z: 团队技能索引 [batch_size]
            z: 个体技能索引 [batch_size, n_agents]
            Z_logits: 团队技能logits [batch_size, n_Z]
            z_logits: 个体技能logits列表 [n_agents个 [batch_size, n_z]]
        """
        batch_size = state.size(0)
        n_agents = observations.size(1)
        device = state.device
        
        # 确保输入是float32类型
        state = state.float()
        observations = observations.float()
        
        # 编码状态和观测
        encoded_state, encoded_observations = self.state_encoder(state, observations)
        
        # 生成团队技能Z
        Z_logits = self.skill_decoder(encoded_state, encoded_observations)
        
        # 在创建分布前检查Z_logits是否包含NaN或Inf
        with torch.no_grad():
            is_nan = torch.isnan(Z_logits).any().item()
            is_inf = torch.isinf(Z_logits).any().item()
            if is_nan or is_inf:
                main_logger.warning("警告: 在创建Categorical分布前，Z_logits包含NaN或Inf值！")
                main_logger.warning(f"Z_logits统计: 均值={Z_logits.mean().item():.4f}, 标准差={Z_logits.std().item():.4f}, "
                      f"最小值={Z_logits.min().item():.4f}, 最大值={Z_logits.max().item():.4f}")
                # 尝试修复NaN/Inf值
                Z_logits = torch.nan_to_num(Z_logits, nan=0.0, posinf=50.0, neginf=-50.0)
                main_logger.warning("已将Z_logits中的NaN和Inf值替换为有限值")
        
        try:
            Z_dist = Categorical(logits=Z_logits)
            
            if deterministic:
                Z = Z_logits.argmax(dim=-1)
            else:
                Z = Z_dist.sample()
            
            # 依次为每个智能体生成个体技能zi
            z = torch.zeros(batch_size, n_agents, dtype=torch.long, device=device)
            z_logits = []
            
            for i in range(n_agents):
                # 使用clone()创建新张量，防止原地修改导致自动求导错误
                Z_clone = Z.clone().detach()
                z_clone = z[:, :i].clone().detach() if i > 0 else None
                
                try:
                    zi_logits = self.skill_decoder(encoded_state, encoded_observations, Z_clone, z_clone, step=i+1)
                    
                    # 在创建分布前检查zi_logits是否包含NaN或Inf
                    with torch.no_grad():
                        is_nan = torch.isnan(zi_logits).any().item()
                        is_inf = torch.isinf(zi_logits).any().item()
                        if is_nan or is_inf:
                            main_logger.warning(f"警告: 在创建Categorical分布前，第{i}个智能体的zi_logits包含NaN或Inf值！")
                            main_logger.warning(f"zi_logits统计: 均值={zi_logits.mean().item():.4f}, 标准差={zi_logits.std().item():.4f}, "
                                  f"最小值={zi_logits.min().item():.4f}, 最大值={zi_logits.max().item():.4f}")
                            # 尝试修复NaN/Inf值
                            zi_logits = torch.nan_to_num(zi_logits, nan=0.0, posinf=50.0, neginf=-50.0)
                            main_logger.warning(f"已将第{i}个智能体的zi_logits中的NaN和Inf值替换为有限值")
                    
                    z_logits.append(zi_logits)
                    zi_dist = Categorical(logits=zi_logits)
                    
                    if deterministic:
                        zi = zi_logits.argmax(dim=-1)
                    else:
                        zi = zi_dist.sample()
                        
                    z[:, i] = zi
                    
                except Exception as e:
                    main_logger.error(f"在处理第{i}个智能体的zi_logits时发生错误: {e}")
                    # 如果发生错误，使用一个安全的默认值
                    safe_logits = torch.zeros((batch_size, self.n_z), device=device)
                    z_logits.append(safe_logits)
                    z[:, i] = 0  # 使用0作为默认技能索引
                    main_logger.warning(f"已为第{i}个智能体使用默认技能索引0")
                
            return Z, z, Z_logits, z_logits
            
        except Exception as e:
            main_logger.error(f"在SkillCoordinator.forward中创建Categorical分布时发生错误: {e}")
            # 返回安全的默认值
            default_Z = torch.zeros(batch_size, dtype=torch.long, device=device)
            default_z = torch.zeros(batch_size, n_agents, dtype=torch.long, device=device)
            default_Z_logits = torch.zeros((batch_size, self.n_Z), device=device)
            default_z_logits = [torch.zeros((batch_size, self.n_z), device=device) for _ in range(n_agents)]
            main_logger.warning("由于错误，返回默认值")
            return default_Z, default_z, default_Z_logits, default_z_logits

class SkillDiscoverer(nn.Module):
    """技能发现器（低层策略）"""
    def __init__(self, config, logger=None): # Add logger parameter
        super(SkillDiscoverer, self).__init__()
        
        self.config = config
        # 保存logger参数，如果为None则使用main_logger
        self.logger = logger if logger is not None else main_logger
        self.obs_dim = config.obs_dim
        self.n_z = config.n_z
        self.action_dim = config.action_dim
        self.hidden_dim = config.hidden_size
        self.gru_hidden_dim = config.gru_hidden_size
        
        # Actor网络（每个智能体共享）
        # 使用实际观测维度初始化网络，而不是配置中的obs_dim
        # 这样可以处理不同场景中不同的观测维度
        self.actor_mlp = None  # 将在forward方法中根据实际输入维度初始化
        self.actor_gru = nn.GRU(config.hidden_size, config.gru_hidden_size, batch_first=True)
        
        # 动作均值和标准差
        self.action_mean = nn.Linear(config.gru_hidden_size, config.action_dim)
        self.action_log_std = nn.Linear(config.gru_hidden_size, config.action_dim)
        
        # 重置参数
        self.actor_hidden = None
        
        # Critic网络（中心化价值函数）
        self.critic_mlp = None  # 将在get_value方法中根据实际输入维度初始化
        self.critic_gru = nn.GRU(config.hidden_size, config.gru_hidden_size, batch_first=True)
        self.value_head = nn.Linear(config.gru_hidden_size, 1)
        
        # 重置参数
        self.critic_hidden = None
    
    def init_hidden(self, batch_size=1):
        """初始化GRU隐藏状态"""
        device = next(self.parameters()).device
        self.actor_hidden = torch.zeros(1, batch_size, self.gru_hidden_dim, device=device)
        self.critic_hidden = torch.zeros(1, batch_size, self.gru_hidden_dim, device=device)
    
    def get_value(self, state, team_skill, batch_first=True):
        """获取价值函数值"""
        batch_size = state.size(0)
        
        # 确保state是float32类型
        state = state.float()
        
        if isinstance(team_skill, int) or isinstance(team_skill, torch.Tensor):
            # 将技能索引转换为独热编码
            if isinstance(team_skill, int):
                team_skill = torch.tensor([team_skill], device=state.device)
            elif team_skill.dim() == 0:  # 处理标量张量
                team_skill = team_skill.unsqueeze(0)  # 转换为一维张量
            
            # 确保是一维张量后进行独热编码
            if team_skill.dim() == 1:
                team_skill_onehot = F.one_hot(team_skill, self.config.n_Z).float()
            else:
                team_skill_onehot = team_skill.float()  # 已经是独热编码，确保是float32
        else:
            team_skill_onehot = team_skill.float()
        
        # 拼接状态和团队技能
        critic_input = torch.cat([state, team_skill_onehot], dim=-1)
        
        # 根据实际输入维度初始化critic_mlp（如果尚未初始化）
        if not hasattr(self, 'critic_mlp') or self.critic_mlp is None:
            actual_state_dim = state.size(-1)
            main_logger.info(f"初始化critic_mlp: 实际状态维度 = {actual_state_dim}, 团队技能维度 = {self.config.n_Z}")
            self.critic_mlp = MLP(actual_state_dim + self.config.n_Z, self.hidden_dim, self.hidden_dim)
            # 将critic_mlp移动到与state相同的设备上
            self.critic_mlp = self.critic_mlp.to(state.device)
        
        # 前向传播
        critic_features = self.critic_mlp(critic_input)
        
        # 确保critic_features是3D的 [batch_size, seq_len, hidden_dim]
        if critic_features.dim() == 2:
            critic_features = critic_features.unsqueeze(1)  # 添加时序维度
        
        # 初始化隐藏状态（如果需要）
        if self.critic_hidden is None or self.critic_hidden.size(1) != batch_size:
            device = critic_features.device
            self.critic_hidden = torch.zeros(1, batch_size, self.gru_hidden_dim, device=device)
            
        critic_output, self.critic_hidden = self.critic_gru(critic_features, self.critic_hidden)
        
        # 移除时序维度
        critic_output = critic_output.squeeze(1)
            
        value = self.value_head(critic_output)
        
        # 确保返回的值是float32类型
        return value.float()
    
    def forward(self, observation, agent_skill, deterministic=False):
        """
        前向传播，生成动作
        
        参数:
            observation: 智能体观测 [batch_size, obs_dim]
            agent_skill: 个体技能索引 [batch_size] 或独热编码 [batch_size, n_z]
            deterministic: 是否使用确定性策略
            
        返回:
            action: 动作 [batch_size, action_dim]
            action_logprob: 动作对数概率 [batch_size]
            action_distribution: 动作分布
        """
        batch_size = observation.size(0)
        
        # 确保observation是float32类型
        observation = observation.float()
        
        if isinstance(agent_skill, int) or isinstance(agent_skill, torch.Tensor):
            # 将技能索引转换为独热编码
            if isinstance(agent_skill, int):
                agent_skill = torch.tensor([agent_skill], device=observation.device)
            elif agent_skill.dim() == 0:  # 处理标量张量
                agent_skill = agent_skill.unsqueeze(0)  # 转换为一维张量
            
            # 确保是一维张量后进行独热编码
            if agent_skill.dim() == 1:
                agent_skill_onehot = F.one_hot(agent_skill, self.n_z).float()
            else:
                agent_skill_onehot = agent_skill.float()  # 已经是独热编码，确保是float32
        else:
            agent_skill_onehot = agent_skill.float()
        
        # 拼接观测和个体技能
        actor_input = torch.cat([observation, agent_skill_onehot], dim=-1)
        self.logger.debug(f"SkillDiscoverer.forward: actor_input shape: {actor_input.shape}, dtype: {actor_input.dtype}")
        
        # 根据实际输入维度初始化actor_mlp（如果尚未初始化）
        if self.actor_mlp is None:
            actual_obs_dim = observation.size(-1)
            print(f"初始化actor_mlp: 实际观测维度 = {actual_obs_dim}, 技能维度 = {self.n_z}")
            self.actor_mlp = MLP(actual_obs_dim + self.n_z, self.hidden_dim, self.hidden_dim)
            # 将actor_mlp移动到与observation相同的设备上
            self.actor_mlp = self.actor_mlp.to(observation.device)
        
        # 前向传播
        actor_features = self.actor_mlp(actor_input).unsqueeze(1)  # 添加时序维度
        
        # 初始化隐藏状态（如果需要）
        if self.actor_hidden is None or self.actor_hidden.size(1) != batch_size:
            device = actor_features.device
            self.actor_hidden = torch.zeros(1, batch_size, self.gru_hidden_dim, device=device)
            
        actor_output, self.actor_hidden = self.actor_gru(actor_features, self.actor_hidden)
        actor_output = actor_output.squeeze(1)  # 移除时序维度
        
        # 生成动作分布参数
        action_mean = self.action_mean(actor_output)
        action_log_std = self.action_log_std(actor_output)
        action_std = torch.exp(action_log_std)
        
        # 检查NaN或Inf并记录日志
        if torch.isnan(action_mean).any() or torch.isinf(action_mean).any():
            self.logger.warning("警告: action_mean中检测到NaN或Inf值!")
            self.logger.warning(f"action_mean统计: 形状={action_mean.shape}, 均值={action_mean.mean().item() if not torch.isnan(action_mean).all() else 'NaN'}, 标准差={action_mean.std().item() if not torch.isnan(action_mean).all() else 'NaN'}")
            # 替换NaN和Inf值
            action_mean = torch.nan_to_num(action_mean, nan=0.0, posinf=1.0, neginf=-1.0)
            self.logger.info("已将action_mean中的NaN和Inf值替换为有限值")
            
        if torch.isnan(action_std).any() or torch.isinf(action_std).any() or (action_std <= 1e-6).any():
            self.logger.warning(f"警告: action_std中检测到NaN、Inf或非常小的值!")
            self.logger.warning(f"action_std统计: 形状={action_std.shape}, 均值={action_std.mean().item() if not torch.isnan(action_std).all() else 'NaN'}, 标准差={action_std.std().item() if not torch.isnan(action_std).all() else 'NaN'}")
            # 替换NaN和Inf值
            action_std = torch.nan_to_num(action_std, nan=1.0, posinf=1.0, neginf=1.0)
            self.logger.info("已将action_std中的NaN和Inf值替换为有限值")
            
        # 添加数值稳定性处理
        # 确保action_std不会太小，避免数值问题
        action_std = torch.clamp(action_std, min=1e-6)
        
        # 创建正态分布
        try:
            action_distribution = Normal(action_mean, action_std)
        except Exception as e:
            self.logger.error(f"创建Normal分布时发生错误: {e}")
            self.logger.error(f"action_mean: {action_mean}")
            self.logger.error(f"action_std: {action_std}")
            # 使用安全的默认值
            action_mean = torch.zeros_like(action_mean)
            action_std = torch.ones_like(action_std)
            action_distribution = Normal(action_mean, action_std)
            self.logger.info("已使用安全的默认值创建Normal分布")
        
        # 采样或选择最佳动作
        if deterministic:
            action = action_mean
        else:
            action = action_distribution.sample()
        
        # 计算动作对数概率
        action_logprob = action_distribution.log_prob(action).sum(dim=-1)
        
        return action, action_logprob, action_distribution

class TeamDiscriminator(nn.Module):
    """团队技能判别器"""
    def __init__(self, config):
        super(TeamDiscriminator, self).__init__()
        
        self.model = MLP(
            input_dim=config.state_dim,
            hidden_dim=config.hidden_size,
            output_dim=config.n_Z,
            n_layers=2
        )
    
    def forward(self, state):
        """
        参数:
            state: 全局状态 [batch_size, state_dim]
            
        返回:
            logits: 团队技能logits [batch_size, n_Z]
        """
        # 确保state是float32类型
        state = state.float()
        return self.model(state)

class IndividualDiscriminator(nn.Module):
    """个体技能判别器"""
    def __init__(self, config):
        super(IndividualDiscriminator, self).__init__()
        
        self.config = config
        self.n_Z = config.n_Z
        
        self.model = MLP(
            input_dim=config.obs_dim + config.n_Z,  # 观测 + 团队技能
            hidden_dim=config.hidden_size,
            output_dim=config.n_z,
            n_layers=2
        )
    
    def forward(self, observation, team_skill):
        """
        参数:
            observation: 智能体观测 [batch_size, obs_dim]
            team_skill: 团队技能索引 [batch_size] 或独热编码 [batch_size, n_Z]
            
        返回:
            logits: 个体技能logits [batch_size, n_z]
        """
        # 确保observation是float32类型
        observation = observation.float()
        
        if isinstance(team_skill, int) or isinstance(team_skill, torch.Tensor):
            # 将技能索引转换为独热编码
            if isinstance(team_skill, int):
                team_skill = torch.tensor([team_skill], device=observation.device)
            elif team_skill.dim() == 0:  # 处理标量张量
                team_skill = team_skill.unsqueeze(0)  # 转换为一维张量
            
            # 确保是一维张量后进行独热编码
            if team_skill.dim() == 1:
                team_skill_onehot = F.one_hot(team_skill, self.config.n_Z).float()
            else:
                team_skill_onehot = team_skill.float()  # 已经是独热编码，确保是float32
        else:
            team_skill_onehot = team_skill.float()
        
        # 拼接观测和团队技能
        discriminator_input = torch.cat([observation, team_skill_onehot], dim=-1)
        
        return self.model(discriminator_input)
