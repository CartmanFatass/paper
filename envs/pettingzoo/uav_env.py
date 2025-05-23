import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers

class MultiUAVEnv(ParallelEnv):
    """
    多无人机基站环境的基类
    
    实现了PettingZoo的Parallel接口，提供了基本的无人机和用户模型
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "multi_uav_env_v0",
        "is_parallelizable": True,
    }
    
    def __init__(
        self,
        n_uavs=5,
        n_users=50,
        area_size=1000,  # 区域大小 (m)
        height_range=(50, 150),  # 无人机高度范围 (m)
        max_speed=30,  # 最大速度 (m/s)
        time_step=1.0,  # 时间步长 (s)
        max_steps=5000,  # 最大步数
        user_distribution="uniform",  # 用户分布类型
        channel_model="free_space",  # 信道模型
        render_mode=None,
        seed=None,
    ):
        """
        初始化多无人机基站环境
        
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
        """
        super().__init__()
        
        # 环境参数
        self.n_uavs = n_uavs
        self.n_users = n_users
        self.area_size = area_size
        self.height_range = height_range
        self.max_speed = max_speed
        self.time_step = time_step
        self.max_steps = max_steps
        self.user_distribution = user_distribution
        self.channel_model = channel_model
        self.render_mode = render_mode
        self.seed_val = seed
        
        # 初始化随机数生成器
        self.np_random = np.random.RandomState(seed)
        
        # 通信参数
        self.carrier_frequency = 2.4e9  # 载波频率 (Hz)
        self.bandwidth = 20e6  # 带宽 (Hz)
        self.tx_power = 20  # 发射功率 (dBm)
        self.noise_power = -104  # 噪声功率 (dBm)
        self.min_sinr = 0  # 最小SINR (dB)
        self.max_connections = 10  # 每个无人机最大连接数
        
        # 地面基站参数 (用于场景2)
        self.ground_bs_positions = np.array([[area_size/2, area_size/2, 30]])  # 中心位置
        self.ground_bs_tx_power = 30  # 地面基站发射功率 (dBm)
        
        # 状态和观测维度
        self.state_dim = 3 * n_uavs + 2 * n_users + 1  # UAV位置 + 用户位置 + 当前步数
        self.obs_dim = 3 + 2 * n_users + n_uavs * 3 + 1  # 自身位置 + 用户位置 + 其他UAV位置 + 当前步数
        
        # 创建智能体列表
        self.possible_agents = [f"uav_{i}" for i in range(n_uavs)]
        self.agents = self.possible_agents.copy()
        
        # 定义观测和动作空间
        self.observation_spaces = {
            agent: Dict({
                "obs": Box(low=-float('inf'), high=float('inf'), shape=(self.obs_dim,)),
                "action_mask": Box(low=0, high=1, shape=(3,))  # 3D速度控制
            }) for agent in self.possible_agents
        }
        
        self.action_spaces = {
            agent: Box(low=-1, high=1, shape=(3,))  # 归一化的3D速度控制
            for agent in self.possible_agents
        }
        
        # 环境状态
        self.uav_positions = None  # 无人机位置 [n_uavs, 3]
        self.user_positions = None  # 用户位置 [n_users, 2]
        self.connections = None  # 连接矩阵 [n_uavs, n_users]
        self.sinr_matrix = None  # SINR矩阵 [n_uavs, n_users]
        self.current_step = 0
        
        # 渲染相关
        self.viewer = None
        self.fig = None
        self.ax = None
        # 重置环境
        self.reset(seed=seed)

    # PettingZoo API methods for spaces
    def observation_space(self, agent):
        """返回指定智能体的观测空间"""
        return self.observation_spaces[agent]

    def action_space(self, agent):
        """返回指定智能体的动作空间"""
        return self.action_spaces[agent]

    def get_state_dim(self):
        """返回全局状态维度"""
        return self.state_dim
    
    def get_obs_dim(self):
        """返回观测维度"""
        return self.obs_dim
    
    def reset(self, seed=None, options=None):
        """
        重置环境
        
        返回:
            observations: 所有智能体的观测字典
            infos: 所有智能体的信息字典
        """
        if seed is not None:
            self.seed_val = seed
            self.np_random = np.random.RandomState(seed)
        
        # 重置环境状态
        self.current_step = 0
        self.agents = self.possible_agents.copy()
        
        # 初始化无人机位置
        self.uav_positions = np.zeros((self.n_uavs, 3))
        for i in range(self.n_uavs):
            self.uav_positions[i] = [
                self.np_random.uniform(0, self.area_size),
                self.np_random.uniform(0, self.area_size),
                self.np_random.uniform(*self.height_range)
            ]
        
        # 初始化用户位置
        self.user_positions = self._generate_user_positions()
        
        # 初始化连接矩阵和SINR矩阵
        self.connections = np.zeros((self.n_uavs, self.n_users), dtype=bool)
        self.sinr_matrix = np.zeros((self.n_uavs, self.n_users))
        
        # 计算初始SINR和连接
        self._update_channel_state()
        
        # 获取所有智能体的观测
        observations = {}
        infos = {}
        
        for agent_idx, agent in enumerate(self.agents):
            observations[agent] = self._get_observation(agent)
            infos[agent] = {}
        
        # 如果在渲染模式下，初始化渲染
        if self.render_mode == "human":
            self._render_frame()
        
        return observations, infos
    
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
        # 更新所有无人机位置
        for agent_idx, agent in enumerate(self.agents):
            if agent in actions:
                # 将归一化动作转换为实际速度
                velocity = actions[agent] * self.max_speed
                
                # 更新位置
                new_position = self.uav_positions[agent_idx] + velocity * self.time_step
                
                # 边界检查
                new_position[0] = np.clip(new_position[0], 0, self.area_size)
                new_position[1] = np.clip(new_position[1], 0, self.area_size)
                new_position[2] = np.clip(new_position[2], *self.height_range)
                
                # 更新位置
                self.uav_positions[agent_idx] = new_position
        
        # 更新信道状态和连接
        self._update_channel_state()
        
        # 计算奖励
        global_reward = self._compute_reward()
        
        # 更新步数
        self.current_step += 1
        
        # 检查是否达到最大步数
        done = self.current_step >= self.max_steps
        
        # 准备返回值
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        
        # 为每个智能体填充返回值
        for agent_idx, agent in enumerate(self.agents):
            observations[agent] = self._get_observation(agent)
            rewards[agent] = global_reward / self.n_uavs  # 平均分配奖励
            terminations[agent] = done
            truncations[agent] = False
            infos[agent] = {
                "connections": self.connections[agent_idx],
                "sinr_matrix": self.sinr_matrix[agent_idx],
                "served_users": np.sum(self.connections[agent_idx])
            }
        
        # 添加全局信息
        global_info = {
            "connections": self.connections,
            "sinr_matrix": self.sinr_matrix,
            "served_users": np.sum(self.connections)
        }
        
        for agent in self.agents:
            infos[agent].update({"global": global_info})
        
        # 如果在渲染模式下，更新渲染
        if self.render_mode == "human":
            self._render_frame()
        
        return observations, rewards, terminations, truncations, infos
    
    def _get_state(self):
        """
        获取全局状态
        
        返回:
            state: 全局状态向量
        """
        # 全局状态包括所有无人机位置、所有用户位置和当前步数
        uav_positions_flat = self.uav_positions.flatten()
        user_positions_flat = self.user_positions.flatten()
        step_normalized = np.array([self.current_step / self.max_steps])
        
        state = np.concatenate([uav_positions_flat, user_positions_flat, step_normalized])
        return state
    
    def _get_observation(self, agent):
        """
        获取指定智能体的观测
        
        参数:
            agent: 智能体ID
            
        返回:
            observation: 智能体的观测
        """
        agent_idx = int(agent.split("_")[1])
        
        # 自身位置
        own_position = self.uav_positions[agent_idx]
        
        # 用户位置
        user_positions_flat = self.user_positions.flatten()
        
        # 其他无人机位置
        other_uav_positions = []
        for i in range(self.n_uavs):
            if i != agent_idx:
                other_uav_positions.append(self.uav_positions[i])
            else:
                # 为了保持维度一致，添加自身位置
                other_uav_positions.append(self.uav_positions[i])
        
        other_uav_positions_flat = np.array(other_uav_positions).flatten()
        
        # 当前步数
        step_normalized = np.array([self.current_step / self.max_steps])
        
        # 组合观测
        obs = np.concatenate([own_position, user_positions_flat, other_uav_positions_flat, step_normalized])
        
        # 动作掩码（这里我们不限制动作，所以全为1）
        action_mask = np.ones(3)
        
        return {"obs": obs, "action_mask": action_mask}
    
    def _generate_user_positions(self):
        """
        生成用户位置
        
        返回:
            user_positions: 用户位置 [n_users, 2]
        """
        if self.user_distribution == "uniform":
            # 均匀分布
            user_positions = np.zeros((self.n_users, 2))
            for i in range(self.n_users):
                user_positions[i] = [
                    self.np_random.uniform(0, self.area_size),
                    self.np_random.uniform(0, self.area_size)
                ]
        
        elif self.user_distribution == "cluster":
            # 聚类分布
            n_clusters = min(5, self.n_users // 10 + 1)
            cluster_centers = np.random.uniform(0, self.area_size, (n_clusters, 2))
            cluster_std = self.area_size / 10
            
            user_positions = np.zeros((self.n_users, 2))
            users_per_cluster = self.n_users // n_clusters
            
            for i in range(n_clusters):
                start_idx = i * users_per_cluster
                end_idx = (i + 1) * users_per_cluster if i < n_clusters - 1 else self.n_users
                
                for j in range(start_idx, end_idx):
                    user_positions[j] = cluster_centers[i] + self.np_random.normal(0, cluster_std, 2)
                    # 确保在区域内
                    user_positions[j] = np.clip(user_positions[j], 0, self.area_size)
        
        elif self.user_distribution == "hotspot":
            # 热点分布
            hotspot_center = np.array([self.area_size/2, self.area_size/2])
            hotspot_radius = self.area_size / 3
            
            user_positions = np.zeros((self.n_users, 2))
            n_hotspot_users = int(self.n_users * 0.7)  # 70%的用户在热点区域
            
            # 热点区域的用户
            for i in range(n_hotspot_users):
                distance = self.np_random.uniform(0, hotspot_radius)
                angle = self.np_random.uniform(0, 2 * np.pi)
                user_positions[i] = hotspot_center + distance * np.array([np.cos(angle), np.sin(angle)])
            
            # 其余用户均匀分布
            for i in range(n_hotspot_users, self.n_users):
                user_positions[i] = [
                    self.np_random.uniform(0, self.area_size),
                    self.np_random.uniform(0, self.area_size)
                ]
        
        else:
            raise ValueError(f"未知的用户分布类型: {self.user_distribution}")
        
        return user_positions
    
    def _compute_distance(self, pos1, pos2):
        """
        计算两点之间的欧几里得距离
        
        参数:
            pos1: 位置1
            pos2: 位置2
            
        返回:
            distance: 距离
        """
        return np.sqrt(np.sum((pos1 - pos2) ** 2))
    
    def _compute_path_loss(self, uav_pos, user_pos):
        """
        计算路径损耗
        
        参数:
            uav_pos: 无人机位置 [3]
            user_pos: 用户位置 [2]
            
        返回:
            path_loss: 路径损耗 (dB)
        """
        # 检查 user_pos 是否为整数索引，如果是则获取对应的用户位置
        if isinstance(user_pos, (int, np.integer)):
            user_pos = self.user_positions[user_pos]
            
        # 计算3D距离和2D距离
        # 确保 user_pos 是二维的 (x, y)
        if len(user_pos) > 2:
            user_pos_2d = user_pos[:2]  # 如果是三维的，取前两个元素
            user_pos_3d = user_pos  # 已经是三维的
        else:
            user_pos_2d = user_pos  # 已经是二维的
            user_pos_3d = np.append(user_pos, 0)  # 假设用户在地面
            
        distance_3d = self._compute_distance(uav_pos, user_pos_3d)
        distance_2d = np.sqrt(np.sum((uav_pos[:2] - user_pos_2d) ** 2))
        height = uav_pos[2]
        
        # 计算仰角 (度)
        elevation_angle = np.degrees(np.arctan2(height, distance_2d))
        
        # 确保距离不为零，避免log10(0)错误
        safe_distance = max(distance_3d, 1e-6)  # 使用一个很小的正数代替零
        
        # 根据不同信道模型计算路径损耗
        
        # 自由空间路径损耗模型
        if self.channel_model == "free_space":
            # 自由空间路径损耗 (dB)
            wavelength = 3e8 / self.carrier_frequency
            path_loss = 20 * np.log10(safe_distance) + 20 * np.log10(4 * np.pi / wavelength)
        
        # 城市环境路径损耗模型
        elif self.channel_model == "urban":
            # 简化的城市环境路径损耗模型
            path_loss = 128.1 + 37.6 * np.log10(safe_distance / 1000)
        
        # 郊区环境路径损耗模型
        elif self.channel_model == "suburban":
            # 简化的郊区环境路径损耗模型
            path_loss = 120 + 35 * np.log10(safe_distance / 1000)
            
        # 3GPP 36.777标准的UAV信道模型 (TR 36.777)
        elif self.channel_model == "3gpp-36777":
            # 频率转换为GHz
            f_c = self.carrier_frequency / 1e9
            
            # 基于3GPP 36.777的路径损耗计算
            # 视距概率 (LoS probability)
            p_los = 1 / (1 + 5 * np.exp(-0.6 * (elevation_angle - 5)))
            
            # 视距路径损耗 (LoS path loss)
            pl_los = 28.0 + 22 * np.log10(safe_distance) + 20 * np.log10(f_c)
            
            # 非视距路径损耗 (NLoS path loss)
            pl_nlos = 22.7 + 41 * np.log10(safe_distance) + 20 * np.log10(f_c)
            
            # 阴影衰落标准差 (Shadow fading standard deviation)
            sigma_los = 4.0
            sigma_nlos = 8.0
            
            # 应用阴影衰落 (假设常规高斯分布)
            if hasattr(self, 'np_random'):
                shadow_los = self.np_random.normal(0, sigma_los)
                shadow_nlos = self.np_random.normal(0, sigma_nlos)
            else:
                shadow_los = np.random.normal(0, sigma_los)
                shadow_nlos = np.random.normal(0, sigma_nlos)
            
            # 计算总路径损耗
            path_loss_los = pl_los + shadow_los
            path_loss_nlos = pl_nlos + shadow_nlos
            
            # 根据视距概率加权平均
            path_loss = p_los * path_loss_los + (1 - p_los) * path_loss_nlos
        
        else:
            raise ValueError(f"未知的信道模型: {self.channel_model}")
            
        # 添加调试打印，用于验证路径损耗计算
        # print(f"Debug: 3D距离={distance_3d:.2f}m, 高度={height:.2f}m, 路径损耗={path_loss:.2f}dB, 信道模型={self.channel_model}")
        
        return path_loss
    
    def _compute_sinr(self, uav_idx, user_idx):
        """
        计算SINR (Signal to Interference plus Noise Ratio)
        
        参数:
            uav_idx: 无人机索引
            user_idx: 用户索引
            
        返回:
            sinr: SINR值 (dB)
        """
        # 检查索引并获取位置
        if isinstance(uav_idx, (int, np.integer)):
            uav_pos = self.uav_positions[uav_idx]
        else:
            uav_pos = uav_idx  # 假设已经是位置
            
        if isinstance(user_idx, (int, np.integer)):
            user_pos = self.user_positions[user_idx]
        else:
            user_pos = user_idx  # 假设已经是位置
        
        # 计算路径损耗
        path_loss = self._compute_path_loss(uav_pos, user_pos)
        
        # 计算接收功率 (dBm)
        rx_power = self.tx_power - path_loss
        
        # 计算干扰功率 (dBm)
        interference_power = []
        for i in range(self.n_uavs):
            if i != uav_idx:
                interferer_pos = self.uav_positions[i]
                interferer_path_loss = self._compute_path_loss(interferer_pos, user_pos)
                interferer_power = self.tx_power - interferer_path_loss
                interference_power.append(10 ** (interferer_power / 10))  # 转换为线性单位
        
        # 总干扰功率 (dBm)
        total_interference = np.sum(interference_power) if interference_power else 0
        total_interference_dbm = 10 * np.log10(total_interference) if total_interference > 0 else -float('inf')
        
        # 计算SINR (dB)
        noise_power_dbm = self.noise_power
        interference_plus_noise_dbm = 10 * np.log10(10 ** (noise_power_dbm / 10) + 10 ** (total_interference_dbm / 10)) if total_interference_dbm != -float('inf') else noise_power_dbm
        
        sinr = rx_power - interference_plus_noise_dbm
        
        return sinr
    
    def _update_channel_state(self):
        """
        更新信道状态和连接
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
    
    def _compute_reward(self):
        """
        计算奖励
        
        返回:
            reward: 全局奖励
        """
        # 基本奖励：已连接用户数
        connected_users = np.sum(self.connections)
        reward = connected_users / self.n_users
        
        # 额外奖励：SINR质量
        total_sinr = 0
        for i in range(self.n_uavs):
            for j in range(self.n_users):
                if self.connections[i, j]:
                    # 归一化SINR到[0,1]范围
                    normalized_sinr = np.clip((self.sinr_matrix[i, j] - self.min_sinr) / 30, 0, 1)
                    total_sinr += normalized_sinr
        
        # 平均SINR质量
        avg_sinr_quality = total_sinr / max(connected_users, 1)
        
        # 组合奖励
        reward = 0.7 * reward + 0.3 * avg_sinr_quality
        
        return reward
    
    def render(self):
        """
        渲染环境
        
        返回:
            frame: 渲染帧
        """
        if self.render_mode is None:
            return
        
        return self._render_frame()
    
    def _render_frame(self):
        """
        渲染单帧
        
        返回:
            frame: 渲染帧
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            print("渲染需要matplotlib库")
            return None
        
        if self.fig is None:
            self.fig = plt.figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            self.ax.clear()
        
        # 设置坐标轴
        self.ax.set_xlim(0, self.area_size)
        self.ax.set_ylim(0, self.area_size)
        self.ax.set_zlim(0, self.height_range[1] * 1.2)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title(f'多无人机基站环境 - 步数: {self.current_step}/{self.max_steps}')
        
        # 绘制用户
        user_x = self.user_positions[:, 0]
        user_y = self.user_positions[:, 1]
        user_z = np.zeros(self.n_users)  # 用户在地面
        self.ax.scatter(user_x, user_y, user_z, c='blue', marker='.', label='用户')
        
        # 绘制无人机
        for i in range(self.n_uavs):
            uav_pos = self.uav_positions[i]
            self.ax.scatter(uav_pos[0], uav_pos[1], uav_pos[2], c='red', marker='^', s=100, label=f'UAV {i}' if i == 0 else "")
            
            # 绘制连接线
            for j in range(self.n_users):
                if self.connections[i, j]:
                    user_pos = self.user_positions[j]
                    self.ax.plot([uav_pos[0], user_pos[0]], [uav_pos[1], user_pos[1]], [uav_pos[2], 0], 'g-', alpha=0.3)
        
        # 绘制地面基站（如果有）
        if hasattr(self, 'ground_bs_positions') and len(self.ground_bs_positions) > 0:
            bs_x = self.ground_bs_positions[:, 0]
            bs_y = self.ground_bs_positions[:, 1]
            bs_z = self.ground_bs_positions[:, 2]
            self.ax.scatter(bs_x, bs_y, bs_z, c='black', marker='s', s=100, label='地面基站')
        
        # 添加图例
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        # 添加统计信息
        connected_users = np.sum(self.connections)
        coverage_ratio = connected_users / self.n_users
        self.ax.text2D(0.02, 0.95, f'已连接用户: {connected_users}/{self.n_users} ({coverage_ratio:.2%})', transform=self.ax.transAxes)
        
        self.fig.canvas.draw()
        
        if self.render_mode == "human":
            plt.pause(0.01)
            return None
        
        # 返回RGB数组
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        canvas = FigureCanvasAgg(self.fig)
        canvas.draw()
        image = np.array(canvas.renderer.buffer_rgba())
        return image
    
    def close(self):
        """关闭环境"""
        if self.fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self.fig)
            self.fig = None
            self.ax = None
