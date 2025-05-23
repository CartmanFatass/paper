import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

class ParallelToArrayAdapter(gym.Env): # Inherit from gym.Env
    """
    适配器类，将PettingZoo的Parallel API环境转换为数组格式接口
    
    这个适配器将Parallel API的字典格式转换为训练脚本期望的数组格式：
    - reset() 返回 (state, observations) 而不是 (observations, infos)
    - step(actions) 接受动作数组而不是动作字典，并返回 (next_state, next_observations, reward, done, info)
      而不是 (observations, rewards, terminations, truncations, infos)
    """
    
    def __init__(self, env, seed=None): # Add seed parameter
        """
        初始化适配器
        
        参数:
            env: PettingZoo Parallel API环境实例
            seed: 随机种子 (可选)
        """
        super().__init__() # Initialize gym.Env
        self.env = env
        self.agents = self.env.possible_agents # Use possible_agents for consistency
        self.n_uavs = len(self.agents)
        self.state_dim = self.env.get_state_dim() # Use getter methods
        self.obs_dim = self.env.get_obs_dim() # Use getter methods
        self.action_dim = self.env.action_space(self.agents[0]).shape[0] # Get action dim from space

        # Define observation and action spaces for the adapted environment
        # Observation space is an array of observations for all UAVs
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.n_uavs, self.obs_dim), dtype=np.float32)
        # Action space is an array of actions for all UAVs
        self.action_space = Box(low=-1, high=1, shape=(self.n_uavs, self.action_dim), dtype=np.float32)

        # 保留环境的其他属性
        self.n_users = getattr(env, 'n_users', None) # Use getattr for safety
        self.area_size = getattr(env, 'area_size', None)
        self.height_range = getattr(env, 'height_range', None)
        self.max_speed = getattr(env, 'max_speed', None)
        self.time_step = getattr(env, 'time_step', None)
        self.max_steps = getattr(env, 'max_steps', None)
        self.user_distribution = getattr(env, 'user_distribution', None)
        self.channel_model = getattr(env, 'channel_model', None)
        self.render_mode = getattr(env, 'render_mode', None)

        # Set seed if provided
        if seed is not None:
            self.seed(seed)

        # 保留环境的方法 (Gym API expects these)
        # self.render = env.render # Render method needs adaptation for Gym API if used directly
        # self.close = env.close # Close method is standard
    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s)."""
        # Note: PettingZoo envs handle seeding internally in reset
        # We might need to pass the seed to the underlying env's reset
        # For now, just store it if needed elsewhere
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        # Pass seed to underlying PettingZoo env if its reset supports it
        # self.env.reset(seed=seed) # This might reset the env prematurely
        return [seed]

    def reset(self, seed=None, options=None):
        """
        重置环境 (符合Gymnasium API)

        返回:
            observations: 所有智能体的观测数组
            info: 包含全局状态和其他信息的字典
        """
        # Pass seed to the underlying PettingZoo environment's reset method
        if seed is not None:
            self.seed(seed) # Set seed for the adapter as well

        # 调用环境的reset方法，获取字典格式的观测和信息
        # Ensure the underlying env's reset uses the seed if provided
        observations_dict, infos_dict = self.env.reset(seed=seed, options=options)

        # 获取全局状态 (assuming the env has a _get_state method)
        state = self.env._get_state() if hasattr(self.env, '_get_state') else None

        # 将字典格式的观测转换为数组格式
        observations_array = self._dict_to_array(observations_dict)

        # Prepare info dictionary (Gym standard)
        info = {
            "state": state,
            "infos_dict": infos_dict # Original PettingZoo infos
        }

        return observations_array.astype(np.float32), info

    def step(self, actions_array):
        """
        执行环境步骤 (符合Gymnasium API)

        参数:
            actions_array: 所有智能体的动作数组 [n_uavs, action_dim]

        返回:
            next_observations: 所有智能体的下一个观测数组
            reward: 全局奖励 (或每个智能体的平均奖励)
            terminated: 是否有任何智能体终止
            truncated: 是否有任何智能体截断
            info: 包含下一状态和其他信息的字典
        """
        # 将数组格式的动作转换为字典格式
        actions_dict = self._array_to_dict(actions_array)

        # 调用环境的step方法，获取字典格式的结果
        observations_dict, rewards_dict, terminations_dict, truncations_dict, infos_dict = self.env.step(actions_dict)

        # 获取全局状态
        next_state = self.env._get_state() if hasattr(self.env, '_get_state') else None

        # 将字典格式的观测转换为数组格式
        next_observations_array = self._dict_to_array(observations_dict)

        # 计算全局奖励（所有智能体奖励的平均值） - 可以根据需要调整
        # reward = sum(rewards_dict.values()) / len(rewards_dict) if rewards_dict else 0
        # 或者使用第一个智能体的奖励作为代理，如果奖励设计是全局的
        first_agent = self.agents[0]
        reward = rewards_dict.get(first_agent, 0) # Assuming global reward reflected in each agent's reward

        # 判断是否终止或截断 (任一智能体终止或截断)
        terminated = any(terminations_dict.values())
        truncated = any(truncations_dict.values())

        # 合并信息 (Gym standard)
        info = {
            "next_state": next_state,
            "terminations_dict": terminations_dict,
            "truncations_dict": truncations_dict,
            "rewards_dict": rewards_dict,
            "infos_dict": infos_dict
        }

        # 添加场景特定信息（从第一个智能体的info中获取，如果存在）
        if first_agent in infos_dict:
            agent_info = infos_dict[first_agent]
            if "scenario" in agent_info:
                info["scenario"] = agent_info["scenario"]
            if "reward_info" in agent_info:
                info["reward_info"] = agent_info["reward_info"]
            if "coverage_ratio" in agent_info:
                info["coverage_ratio"] = agent_info["coverage_ratio"]
            if "served_users" in agent_info:
                 info["served_users"] = agent_info.get("global", {}).get("served_users", 0) # Get global served users

        return next_observations_array.astype(np.float32), float(reward), terminated, truncated, info

    def _dict_to_array(self, data_dict):
        """
        将PettingZoo字典格式的观测/动作转换为数组格式
        Assumes data_dict contains the actual observation/action under the 'obs' key if it's a Dict space,
        or is the observation/action directly if it's a Box space.
        Handles cases where agents might be missing from the dict (e.g., after termination).

        参数:
            data_dict: 字典格式的数据 {agent_id: data}

        返回:
            data_array: 数组格式的数据 [n_agents, data_dim]
        """
        data_array = []
        default_value = None # Need a default if an agent is missing

        for agent in self.agents: # Iterate through possible agents for consistent order
            agent_data = data_dict.get(agent)

            if agent_data is not None:
                 # Check if the original observation space was a Dict
                original_obs_space = self.env.observation_space(agent)
                if isinstance(original_obs_space, gym.spaces.Dict) and "obs" in original_obs_space.spaces:
                     # Extract the 'obs' part if it's a Dict space observation
                     actual_data = agent_data.get("obs") if isinstance(agent_data, dict) else agent_data # Handle potential direct obs return
                else:
                     # Assume it's already the correct data (e.g., action)
                     actual_data = agent_data

                if actual_data is not None:
                    data_array.append(actual_data)
                    if default_value is None: # Infer default value shape/type from first valid data
                         default_value = np.zeros_like(actual_data)
                elif default_value is not None:
                     data_array.append(default_value) # Use default if agent data is None but we have a default shape
                # else: Cannot determine shape yet, skip or raise error? For now, skip.

            elif default_value is not None:
                # Agent not in dict (e.g., terminated), use default value
                data_array.append(default_value)
            # else: Agent not in dict and no default value yet. This case should ideally not happen
            # if the environment correctly returns observations for all agents upon reset.
            # If it happens mid-episode, we need a strategy (e.g., zero padding).

        # Ensure all appended data has the same shape before stacking
        if not data_array:
             # Handle case where no data was collected (e.g., env terminates immediately)
             # Return an empty array with the correct shape if possible
             if default_value is not None:
                 return np.array([default_value] * len(self.agents)) # Should not happen often
             else:
                 # Cannot determine shape, return empty or raise error
                 # Let's return an empty array matching the expected space shape
                 if isinstance(data_dict, dict) and data_dict: # Check if it was observation dict
                     return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
                 else: # Assume action dict
                     return np.zeros(self.action_space.shape, dtype=self.action_space.dtype)


        # Find the maximum length among sub-arrays if shapes differ (shouldn't happen with Box spaces)
        # max_len = max(len(arr) for arr in data_array)
        # Pad arrays if necessary (again, shouldn't be needed for Box)
        # padded_array = [np.pad(arr, (0, max_len - len(arr)), 'constant') for arr in data_array]

        try:
            result_array = np.stack(data_array)
            # Ensure dtype matches the space definition
            if result_array.dtype != self.observation_space.dtype:
                 if self.observation_space.contains(result_array.astype(self.observation_space.dtype)):
                     return result_array.astype(self.observation_space.dtype)
            return result_array

        except ValueError as e:
             print(f"Error stacking array: {e}")
             print(f"Data array contents: {[arr.shape for arr in data_array]}")
             # Fallback: return zero array matching space shape
             # This might hide issues but prevents crashes
             if default_value is not None: # Check if it was observation dict based on shape
                 if default_value.shape == self.observation_space.shape[1:]:
                     return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
             return np.zeros(self.action_space.shape, dtype=self.action_space.dtype)



    def _array_to_dict(self, data_array):
        """
        将数组格式的动作/数据转换为PettingZoo字典格式

        参数:
            data_array: 数组格式的数据 [n_agents, data_dim]

        返回:
            data_dict: 字典格式的数据 {agent_id: data}
        """
        data_dict = {}
        for i, agent in enumerate(self.agents):
             if i < len(data_array): # Check bounds
                 data_dict[agent] = data_array[i]
        return data_dict

    def render(self, mode="human"):
        """Renders the environment."""
        # Use the underlying PettingZoo environment's render method
        return self.env.render() # Assuming the base env render matches Gym modes

    def close(self):
        """Closes the environment."""
        self.env.close()
