import numpy as np
from envs.pettingzoo.uav_env import MultiUAVEnv

class UAVBaseStationEnv(MultiUAVEnv):
    """
    场景1：无人机均作为基站为地面用户服务
    
    特点：
    - 所有无人机直接作为基站服务地面用户
    - 不考虑回程和中继
    - 优化目标是最大化用户覆盖率和服务质量
    """
    
    def __init__(
        self,
        n_uavs=5,
        n_users=50,
        area_size=1000,
        height_range=(50, 150),
        max_speed=30,
        time_step=1.0,
        max_steps=5000,
        user_distribution="uniform",
        channel_model="free_space",
        render_mode=None,
        seed=None,
        min_sinr=0,  # 最小SINR阈值 (dB)
        max_connections=10,  # 每个无人机最大连接数
        coverage_weight=0.7,  # 覆盖率权重
        quality_weight=0.3,  # 服务质量权重
    ):
        """
        初始化UAV基站环境
        
        参数:
            n_uavs: 无人机数量
            n_users: 用户数量
            area_size: 区域大小 (m)
            height_range: 无人机高度范围 (m)
            max_speed: 最大速度 (m/s)
            time_step: 时间步长 (s)
            max_steps: 最大步数
            user_distribution: 用户分布类型 ("uniform", "cluster", "hotspot")
            channel_model: 信道模型 ("free_space", "urban", "suburban")
            render_mode: 渲染模式
            seed: 随机种子
            min_sinr: 最小SINR阈值 (dB)
            max_connections: 每个无人机最大连接数
            coverage_weight: 覆盖率权重
            quality_weight: 服务质量权重
        """
        # 调用父类初始化
        super().__init__(
            n_uavs=n_uavs,
            n_users=n_users,
            area_size=area_size,
            height_range=height_range,
            max_speed=max_speed,
            time_step=time_step,
            max_steps=max_steps,
            user_distribution=user_distribution,
            channel_model=channel_model,
            render_mode=render_mode,
            seed=seed,
        )
        
        # 场景特定参数
        self.min_sinr = min_sinr
        self.max_connections = max_connections
        self.coverage_weight = coverage_weight
        self.quality_weight = quality_weight
        
        # 场景名称
        self.metadata["name"] = "uav_base_station_v0"
    
    def _compute_reward(self):
        """
        计算奖励
        
        返回:
            reward: 全局奖励
        """
        # 基本奖励：已连接用户数（覆盖率）
        connected_users = np.sum(self.connections)
        coverage_ratio = connected_users / self.n_users
        coverage_reward = coverage_ratio
        
        # 服务质量奖励：SINR质量
        total_sinr = 0
        for i in range(self.n_uavs):
            for j in range(self.n_users):
                if self.connections[i, j]:
                    # 归一化SINR到[0,1]范围
                    normalized_sinr = np.clip((self.sinr_matrix[i, j] - self.min_sinr) / 30, 0, 1)
                    total_sinr += normalized_sinr
        
        # 平均SINR质量
        quality_reward = total_sinr / max(connected_users, 1)
        
        # 能量效率奖励：鼓励无人机降低高度（减少能耗）
        avg_height = np.mean(self.uav_positions[:, 2])
        height_range = self.height_range[1] - self.height_range[0]
        normalized_height = (avg_height - self.height_range[0]) / height_range
        energy_penalty = normalized_height * 0.1  # 高度越高，惩罚越大
        
        # 组合奖励
        reward = (
            self.coverage_weight * coverage_reward + 
            self.quality_weight * quality_reward - 
            energy_penalty
        )
        
        # 记录奖励组成
        self.reward_info = {
            "coverage_reward": coverage_reward,
            "quality_reward": quality_reward,
            "energy_penalty": energy_penalty,
            "total_reward": reward
        }
        
        return reward
    
    def _update_channel_state(self):
        """
        更新信道状态和连接
        
        在场景1中，无人机直接连接用户，不考虑回程
        """
        # 计算所有UAV-用户对的SINR
        for i in range(self.n_uavs):
            for j in range(self.n_users):
                self.sinr_matrix[i, j] = self._compute_sinr(i, j)
        
        # 更新连接（贪婪算法）
        self.connections = np.zeros((self.n_uavs, self.n_users), dtype=bool)
        
        # 按SINR降序排列所有UAV-用户对
        uav_user_pairs = []
        for i in range(self.n_uavs):
            for j in range(self.n_users):
                if self.sinr_matrix[i, j] >= self.min_sinr:
                    uav_user_pairs.append((i, j, self.sinr_matrix[i, j]))
        
        # 按SINR降序排序
        uav_user_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # 分配连接
        uav_connections = [0] * self.n_uavs  # 每个UAV的连接数
        user_connected = [False] * self.n_users  # 每个用户是否已连接
        
        for uav_idx, user_idx, sinr in uav_user_pairs:
            # 如果UAV未达到最大连接数且用户未连接
            if uav_connections[uav_idx] < self.max_connections and not user_connected[user_idx]:
                self.connections[uav_idx, user_idx] = True
                uav_connections[uav_idx] += 1
                user_connected[user_idx] = True
    
    def step(self, actions):
        """
        执行环境步骤
        
        参数:
            actions: 所有智能体的动作字典 {agent_id: action}
            
        返回:
            observations: 所有智能体的下一个观测字典
            rewards: 所有智能体的奖励字典
            terminations: 所有智能体的终止状态字典
            truncations: 所有智能体的截断状态字典
            infos: 所有智能体的信息字典
        """
        observations, rewards, terminations, truncations, infos = super().step(actions)
        
        # 添加场景特定信息到每个智能体的info中
        scenario_info = {
            "scenario": "base_station",
            "reward_info": self.reward_info if hasattr(self, "reward_info") else {},
            "coverage_ratio": np.sum(self.connections) / self.n_users,
        }
        
        for agent in self.agents:
            infos[agent].update(scenario_info)
        
        return observations, rewards, terminations, truncations, infos
    
    def _render_frame(self):
        """渲染单帧"""
        frame = super()._render_frame()
        
        # 在场景1中，我们只需要显示无人机到用户的连接
        # 父类已经实现了这个功能，所以这里不需要额外的渲染逻辑
        
        return frame
