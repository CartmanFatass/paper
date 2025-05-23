import os
import time
import numpy as np
import torch
import logging
import traceback
from datetime import datetime
import multiprocessing as mp
from logger import init_multiproc_logging, get_logger, shutdown_logging, LOG_LEVELS

# 导入 Stable Baselines3 的向量化环境
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# 导入自定义的模块
from config_1 import Config
from hmasd.agent import HMASDAgent
from envs.pettingzoo.env_adapter import ParallelToArrayAdapter
from envs.pettingzoo.scenario1 import UAVBaseStationEnv
# 只导入make_env，不导入get_device和evaluate
from train_multiproc_config_1 import make_env
# 增加必要的异常类型，用于捕获
from multiprocessing.connection import EOFError, BrokenPipeError

# 测试配置类，继承自原始Config但设置更小的参数用于快速测试
class TestConfig(Config):
    """
    用于快速测试的配置类，继承自原始Config，但设置更小的参数
    """
    def __init__(self):
        super().__init__()
        
        # 减小网络大小
        self.hidden_size = 64  # 减小网络大小
        
        # 减少训练步数
        self.total_timesteps = 10000  # 总训练步数
        self.buffer_size = 256  # 经验回放缓冲区大小
        self.batch_size = 64  # 批量大小
        self.high_level_batch_size = 32  # 高层批量大小
        
        # 减小并行环境数
        self.num_envs = 2  # 测试时使用较少的并行环境
        self.eval_rollout_threads = 2  # 评估并行线程数
        
        # 增加评估频率
        self.eval_interval = 1000  # 每1000步评估一次
        self.eval_episodes = 2  # 每次评估2个episodes
        
        # 技能相关参数
        self.k = 5  # 减少技能持续时间
        self.n_Z = 2  # 减少团队技能数量
        self.n_z = 2  # 减少智能体技能数量

# 安全创建向量化环境的工具函数
def safe_create_vec_env(env_fns, use_subproc=True, start_method='spawn', logger=None):
    """
    安全地创建向量化环境，提供回退机制
    
    参数:
        env_fns: 环境创建函数列表
        use_subproc: 是否使用多进程环境
        start_method: 多进程启动方法 ('spawn', 'fork', 'forkserver')
        logger: 日志记录器实例
        
    返回:
        vec_env: 向量化环境实例
    """
    if logger is None:
        logger = get_logger("VecEnv")
        
    if use_subproc:
        try:
            logger.info(f"创建SubprocVecEnv，start_method={start_method}")
            return SubprocVecEnv(env_fns, start_method=start_method)
        except Exception as e:
            logger.warning(f"创建SubprocVecEnv失败: {e}，回退到DummyVecEnv")
            return DummyVecEnv(env_fns)
    else:
        logger.info("创建DummyVecEnv")
        return DummyVecEnv(env_fns)

# 重新定义get_device函数，使用local logger而不是依赖main_logger
def get_device(device_pref, logger):
    """
    根据偏好选择计算设备
    
    参数:
        device_pref: 设备偏好 ('auto', 'cuda', 'cpu')
        logger: 日志记录器实例
        
    返回:
        device: torch.device对象
    """
    if device_pref == 'auto':
        if torch.cuda.is_available():
            logger.info("检测到GPU可用，使用CUDA")
            return torch.device('cuda')
        else:
            logger.info("未检测到GPU，使用CPU")
            return torch.device('cpu')
    elif device_pref == 'cuda':
        if torch.cuda.is_available():
            logger.info("使用CUDA")
            return torch.device('cuda')
        else:
            logger.warning("请求使用CUDA但未检测到GPU，回退到CPU")
            return torch.device('cpu')
    else:  # 'cpu'或其他值
        logger.info("使用CPU")
        return torch.device('cpu')

# 重新定义evaluate函数，使用local logger而不是依赖main_logger，添加错误处理
def evaluate(vec_env, agent, n_episodes=10, render=False, logger=None, timeout=300):
    """
    评估HMASD代理 (使用 SubprocVecEnv)，增强错误处理

    参数:
        vec_env: SubprocVecEnv 实例
        agent: HMASD代理实例
        n_episodes: 评估的episode数量 (总共要评估的episode数量)
        render: 是否渲染环境 (只渲染第一个环境)
        logger: 日志记录器实例
        timeout: 评估超时时间(秒)，防止无限等待

    返回:
        mean_reward: 平均奖励
        std_reward: 奖励标准差
        min_reward: 最小奖励
        max_reward: 最大奖励
    """
    # 使用传入的logger或者创建一个默认的
    if logger is None:
        logger = get_logger("Evaluate")
    
    # 打印评估参数
    num_envs = vec_env.num_envs
    logger.info(f"开始评估: 目标完成 {n_episodes} 个episodes，使用 {num_envs} 个并行环境，是否渲染: {render}")
    
    # 用于计时的变量
    eval_start_time = time.time()
    step_times = []
    agent_step_times = []
    env_step_times = []
    episode_rewards = []
    episode_lengths = []
    eval_step = getattr(agent, 'global_step', 0) # Get current training step if available
    num_envs = vec_env.num_envs

    # 重置所有环境并获取初始状态
    results = vec_env.env_method('reset')
    observations = np.array([res[0] for res in results])
    initial_infos = [res[1] for res in results]
    # Use agent.config.state_dim for default state shape
    states = np.array([info.get('state', np.zeros(agent.config.state_dim)) for info in initial_infos]) # Use agent's state_dim

    # 环境状态跟踪
    env_steps = np.zeros(num_envs, dtype=int)
    env_rewards = np.zeros(num_envs)
    active_envs = np.ones(num_envs, dtype=bool) # Track which envs are still running for the current eval round
    completed_episodes = 0
    
    # 统计信息
    all_team_skills = []  # 记录每个时间步的团队技能
    all_agent_skills = []  # 记录每个时间步的个体技能
    total_served_users = []  # 记录每个episode的服务用户数
    total_coverage_ratios = []  # 记录每个episode的覆盖率
    
    # 奖励统计
    high_level_rewards = []  # 高层奖励 (环境奖励)
    low_level_rewards = {   # 底层奖励组成
        'env_component': [],        # 环境奖励部分
        'team_disc_component': [],  # 团队判别器部分
        'ind_disc_component': []    # 个体判别器部分
    }

    # 设置确定性评估模式
    with torch.no_grad():
        while completed_episodes < n_episodes:
            loop_start_time = time.time()
            
            # 为活跃环境选择动作
            all_actions_list = []
            all_agent_infos_list = [] # Store agent info for logging if needed

            # 记录agent.step总时间
            agent_step_start = time.time()
            for i in range(num_envs):
                if active_envs[i]:
                    # 记录每个agent.step调用的时间
                    step_start = time.time()
                    actions, agent_info = agent.step(states[i], observations[i], env_steps[i], deterministic=True)
                    step_end = time.time()
                    agent_step_times.append(step_end - step_start)
                    
                    all_actions_list.append(actions)
                    all_agent_infos_list.append(agent_info)
                    
                    # 收集技能分布信息
                    all_team_skills.append(agent_info['team_skill'])
                    all_agent_skills.append(agent_info['agent_skills'])
                else:
                    # Append dummy action if env is already done for this eval round
                    all_actions_list.append(np.zeros(vec_env.action_space.shape[1:])) # Use action space shape
                    all_agent_infos_list.append({}) # Dummy info
            agent_step_end = time.time()
            agent_step_total = agent_step_end - agent_step_start

            actions_array = np.array(all_actions_list)

    # 执行动作并记录环境步进时间，添加错误处理
            env_step_start = time.time()
            try:
                next_observations, rewards, dones, infos = vec_env.step(actions_array)
                env_step_end = time.time()
                env_step_times.append(env_step_end - env_step_start)
            except (BrokenPipeError, EOFError) as e:
                logger.error(f"环境通信错误: {e}")
                logger.info("评估中断，返回已收集的结果")
                break
            except Exception as e:
                logger.error(f"评估过程中发生未知错误: {e}")
                logger.error(traceback.format_exc())
                break
            
            # 每100步打印一次性能统计
            steps_done = len(agent_step_times)
            if steps_done % 100 == 0 and steps_done > 0:
                avg_agent_step = np.mean(agent_step_times[-100:])
                avg_env_step = np.mean(env_step_times[-100:])
                logger.info(f"评估性能统计 [{steps_done}步]: agent.step平均耗时: {avg_agent_step:.6f}秒/步, "
                      f"vec_env.step平均耗时: {avg_env_step:.6f}秒/步")
            
            loop_end_time = time.time()
            step_times.append(loop_end_time - loop_start_time)

            # 从 infos 提取 next_states
            # Use agent.config.state_dim for default state shape
            next_states = np.array([info.get('next_state', np.zeros(agent.config.state_dim)) for info in infos])

            # 更新环境状态
            for i in range(num_envs):
                if active_envs[i]:
                    env_steps[i] += 1
                    env_rewards[i] += rewards[i]

                    if render and i == 0:
                        try:
                            vec_env.env_method('render', indices=[0]) # Render only the first env
                        except Exception as e:
                            logger.error(f"渲染错误: {e}")
                            render = False # Disable rendering if it fails

                    # 如果环境完成
                    if dones[i]:
                        if completed_episodes < n_episodes:
                            episode_rewards.append(env_rewards[i])
                            episode_lengths.append(env_steps[i])
                            
                            # 获取服务用户数和覆盖率信息（如果可用）
                            if 'global' in infos[i] and 'served_users' in infos[i]['global']:
                                served_users = infos[i]['global']['served_users']
                                n_users = len(infos[i]['global']['connections'][0]) if infos[i]['global']['connections'].shape[0] > 0 else 0
                                coverage_ratio = served_users / n_users if n_users > 0 else 0
                                
                                total_served_users.append(served_users)
                                total_coverage_ratios.append(coverage_ratio)
                                
                                logger.info(f"评估 Episode {completed_episodes+1}/{n_episodes} (来自环境 {i}), 奖励: {env_rewards[i]:.2f}, 步数: {env_steps[i]}, 服务用户数: {served_users}/{n_users} ({coverage_ratio:.2%})")
                            else:
                                logger.info(f"评估 Episode {completed_episodes+1}/{n_episodes} (来自环境 {i}), 奖励: {env_rewards[i]:.2f}, 步数: {env_steps[i]}")

                            # 记录到TensorBoard
                            if hasattr(agent, 'writer'):
                                agent.writer.add_scalar('Eval/episode_reward', env_rewards[i], eval_step + completed_episodes)
                                agent.writer.add_scalar('Eval/episode_length', env_steps[i], eval_step + completed_episodes)
                                if 'global' in infos[i] and 'served_users' in infos[i]['global']:
                                    agent.writer.add_scalar('Eval/served_users', served_users, eval_step + completed_episodes)
                                    agent.writer.add_scalar('Eval/coverage_ratio', coverage_ratio, eval_step + completed_episodes)

                            # 记录高层奖励
                            high_level_rewards.append(env_rewards[i])
                            completed_episodes += 1

                        # 标记此环境在此评估轮次中完成
                        active_envs[i] = False

                        # 不需要手动重置，SubprocVecEnv 会自动处理
                        # 也不需要重置 env_steps 和 env_rewards，因为我们只运行 n_episodes

            # 更新状态和观测
            states = next_states
            observations = next_observations

            # 如果所有需要的 episodes 都已完成，则退出循环
            if completed_episodes >= n_episodes:
                break
            # 如果所有环境都已完成其当前 episode 但仍未达到 n_episodes，也可能需要退出或处理
            if not np.any(active_envs):
                logger.warning("所有评估环境都已完成，但尚未达到目标 episode 数量。")
                break

    mean_reward = np.mean(episode_rewards) if episode_rewards else 0
    std_reward = np.std(episode_rewards) if episode_rewards else 0
    min_reward = np.min(episode_rewards) if episode_rewards else 0
    max_reward = np.max(episode_rewards) if episode_rewards else 0
    mean_length = np.mean(episode_lengths) if episode_lengths else 0

    # 记录评估统计信息
    if hasattr(agent, 'writer'):
        agent.writer.add_scalar('Eval/mean_reward', mean_reward, eval_step)
        agent.writer.add_scalar('Eval/reward_std', std_reward, eval_step)
        agent.writer.add_scalar('Eval/mean_episode_length', mean_length, eval_step)
        agent.writer.flush()

    # 分析技能使用分布
    if all_team_skills:
        team_skill_counts = np.zeros(agent.config.n_Z)
        for skill in all_team_skills:
            team_skill_counts[skill] += 1
        team_skill_probs = team_skill_counts / len(all_team_skills)
        
        logger.info("\n===== 评估技能分布统计 =====")
        logger.info(f"团队技能使用分布: {team_skill_probs}")
        
        # 统计智能体技能使用情况
        if all_agent_skills:
            all_agent_skills_np = np.array(all_agent_skills)
            agent_skill_counts = np.zeros((agent.config.n_agents, agent.config.n_z))
            for skills in all_agent_skills:
                for i, skill in enumerate(skills):
                    if i < agent.config.n_agents:  # 确保索引在范围内
                        agent_skill_counts[i, skill] += 1
            
            # 计算每个智能体的技能使用概率
            agent_skill_probs = agent_skill_counts / len(all_agent_skills)
            
            # 打印每个智能体的技能使用情况
            for i in range(min(3, agent.config.n_agents)):  # 只打印前3个智能体以避免输出过多
                logger.info(f"智能体 {i} 技能使用分布: {agent_skill_probs[i]}")
            
            if agent.config.n_agents > 3:
                logger.info(f"... (共 {agent.config.n_agents} 个智能体)")
    
        # 记录到TensorBoard
        if hasattr(agent, 'writer'):
            for z in range(agent.config.n_Z):
                agent.writer.add_scalar(f'Eval/TeamSkill_{z}_Probability', team_skill_probs[z], eval_step)
            
            for i in range(agent.config.n_agents):
                for z in range(agent.config.n_z):
                    agent.writer.add_scalar(f'Eval/Agent{i}_Skill_{z}_Probability', agent_skill_probs[i][z], eval_step)

    # 打印奖励统计信息
    if high_level_rewards:
        logger.info("\n===== 评估奖励统计 =====")
        mean_high_level = np.mean(high_level_rewards)
        logger.info(f"高层奖励平均值: {mean_high_level:.4f}")

    # 计算并打印性能统计
    eval_total_time = time.time() - eval_start_time
    total_steps_taken = sum(episode_lengths) if episode_lengths else 0
    if total_steps_taken > 0:
        avg_step_time = eval_total_time / total_steps_taken
        avg_agent_step_time = np.mean(agent_step_times) if agent_step_times else 0
        avg_env_step_time = np.mean(env_step_times) if env_step_times else 0
        
        logger.info("\n===== 评估性能统计 =====")
        logger.info(f"总评估时间: {eval_total_time:.2f}秒 (完成 {len(episode_rewards)} episodes, 共 {total_steps_taken} 步)")
        logger.info(f"每步平均耗时: {avg_step_time:.6f}秒")
        logger.info(f"agent.step 平均耗时: {avg_agent_step_time:.6f}秒/步 (占 {avg_agent_step_time/avg_step_time*100:.1f}%)")
        logger.info(f"env.step 平均耗时: {avg_env_step_time:.6f}秒/步 (占 {avg_env_step_time/avg_step_time*100:.1f}%)")
        logger.info(f"其他操作耗时: {avg_step_time - avg_agent_step_time - avg_env_step_time:.6f}秒/步")
        
        # 将性能指标也记录到TensorBoard中
        if hasattr(agent, 'writer'):
            agent.writer.add_scalar('Performance/total_eval_time', eval_total_time, eval_step)
            agent.writer.add_scalar('Performance/avg_step_time', avg_step_time, eval_step)
            agent.writer.add_scalar('Performance/avg_agent_step_time', avg_agent_step_time, eval_step)
            agent.writer.add_scalar('Performance/avg_env_step_time', avg_env_step_time, eval_step)
    
    logger.info(f"\n评估完成 ({len(episode_rewards)} episodes): 平均奖励 {mean_reward:.2f} ± {std_reward:.2f}, 平均步数: {mean_length:.2f}")

    return mean_reward, std_reward, min_reward, max_reward

def quick_test():
    """
    快速测试训练流程，重点验证评估逻辑是否正常工作
    """
    # 初始化日志系统
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = 'test'
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"test_training_{timestamp}.log"
    
    # 初始化日志
    init_multiproc_logging(log_dir=log_dir, log_file=log_file, file_level=logging.INFO, console_level=logging.INFO)
    logger = get_logger("Test-Training")
    logger.info("开始快速测试训练流程")
    
    # 创建测试配置
    config = TestConfig()
    logger.info(f"使用测试配置：总步数={config.total_timesteps}, 评估间隔={config.eval_interval}")
    
    # 获取设备 - 注意这里传递logger
    device = get_device('auto', logger)
    
    # 环境参数
    scenario = 1  # 使用场景1（基站模式）
    n_uavs = 3  # 较少的无人机数量
    n_users = 20  # 较少的用户数量
    user_distribution = 'uniform'
    channel_model = '3gpp-36777'
    render_mode = None
    
    # 创建环境
    num_envs = config.num_envs
    eval_num_envs = config.eval_rollout_threads
    base_seed = int(time.time())
    logger.info(f"创建 {num_envs} 个训练环境和 {eval_num_envs} 个评估环境")
    
    # 创建训练环境
    train_env_fns = [make_env(
        scenario=scenario,
        n_uavs=n_uavs,
        n_users=n_users,
        user_distribution=user_distribution,
        channel_model=channel_model,
        render_mode=None,
        rank=i,
        seed=base_seed
    ) for i in range(num_envs)]
    
    # 创建评估环境
    eval_env_fns = [make_env(
        scenario=scenario,
        n_uavs=n_uavs,
        n_users=n_users,
        user_distribution=user_distribution,
        channel_model=channel_model,
        render_mode=None,
        rank=i,
        seed=base_seed + num_envs
    ) for i in range(eval_num_envs)]
    
    # 使用安全的方式创建向量化环境
    logger.info("创建向量化环境...")
    train_vec_env = safe_create_vec_env(train_env_fns, use_subproc=True, start_method='spawn', logger=logger)
    eval_vec_env = safe_create_vec_env(eval_env_fns, use_subproc=True, start_method='spawn', logger=logger)
    
    # 更新配置中的智能体数量
    try:
        n_agents_from_env = train_vec_env.get_attr('n_uavs')[0]
        config.n_agents = n_agents_from_env
        logger.info(f"从环境更新智能体数量: n_agents={config.n_agents}")
    except Exception as e:
        logger.warning(f"无法从环境获取 n_uavs: {e}. 使用默认值: {n_uavs}")
        config.n_agents = n_uavs
    
    # 更新环境维度
    state_dim = train_vec_env.get_attr('state_dim')[0]
    obs_shape = train_vec_env.observation_space.shape
    if len(obs_shape) == 3:
        obs_dim = obs_shape[2]
    else:
        obs_dim = train_vec_env.get_attr('obs_dim')[0]
    
    config.update_env_dims(state_dim, obs_dim)
    logger.info(f"环境维度: state_dim={state_dim}, obs_dim={obs_dim}")
    
    # 创建代理
    test_log_dir = os.path.join(log_dir, f"test_run_{timestamp}")
    os.makedirs(test_log_dir, exist_ok=True)
    agent = HMASDAgent(config, log_dir=test_log_dir, device=device)
    logger.info("代理已创建")
    
    # 训练参数
    total_steps = 0
    n_episodes = 0
    episode_rewards = []
    update_times = 0
    last_eval_step = 0  # 跟踪上次评估的步数
    best_reward = float('-inf')
    
    # 环境状态跟踪
    env_steps = np.zeros(num_envs, dtype=int)
    env_rewards = np.zeros(num_envs)
    env_skill_durations = np.zeros(num_envs, dtype=int)
    
    # 重置环境
    logger.info("重置环境...")
    results = train_vec_env.env_method('reset')
    observations = np.array([res[0] for res in results])
    initial_infos = [res[1] for res in results]
    states = np.array([info.get('state', np.zeros(agent.config.state_dim)) for info in initial_infos])
    
    # 训练循环
    logger.info("开始训练循环，目标步数：{}".format(config.total_timesteps))
    start_time = time.time()
    eval_counts = 0
    
    while total_steps < config.total_timesteps:
        # 代理选择动作
        all_actions_list = []
        all_agent_infos_list = []
        
        for i in range(num_envs):
            actions, agent_info = agent.step(states[i], observations[i], env_steps[i], deterministic=False)
            all_actions_list.append(actions)
            all_agent_infos_list.append(agent_info)
        
        # 将动作列表转换为NumPy数组
        actions_array = np.array(all_actions_list)
        
        # 执行动作并添加错误处理
        try:
            next_observations, rewards, dones, infos = train_vec_env.step(actions_array)
        except (BrokenPipeError, EOFError) as e:
            logger.error(f"训练中发生管道错误: {e}")
            logger.info("尝试自动恢复通信...")
            
            # 创建新的训练环境
            try:
                train_vec_env.close()
            except:
                logger.warning("关闭原有环境失败，继续创建新环境")
                
            train_vec_env = safe_create_vec_env(train_env_fns, use_subproc=True, start_method='spawn', logger=logger)
            
            # 重置环境状态
            results = train_vec_env.env_method('reset')
            next_observations = np.array([res[0] for res in results]) 
            infos = [res[1] for res in results]
            next_states = np.array([info.get('state', np.zeros(agent.config.state_dim)) for info in infos])
            
            # 设置虚拟奖励和done信号
            rewards = np.zeros(num_envs)
            dones = np.ones(num_envs)  # 将所有环境标记为完成，强制重置
            
            logger.info("环境已重新创建和重置")
        
        # 提取next_states
        next_states = np.array([info.get('next_state', np.zeros(state_dim)) for info in infos])
        
        # 更新环境状态和存储经验
        for i in range(num_envs):
            current_agent_info = all_agent_infos_list[i]
            skill_timer_value = env_skill_durations[i]
            
            # 存储转换
            agent.store_transition(
                states[i], next_states[i], observations[i], next_observations[i],
                actions_array[i], rewards[i], dones[i], current_agent_info['team_skill'],
                current_agent_info['agent_skills'], current_agent_info['action_logprobs'],
                log_probs=current_agent_info['log_probs'],
                skill_timer_for_env=skill_timer_value,
                env_id=i
            )
            
            # 在存储转换后更新技能持续时间
            if dones[i]:
                env_skill_durations[i] = 0
            elif skill_timer_value == config.k - 1:
                env_skill_durations[i] = 0
            elif current_agent_info['skill_changed']:
                env_skill_durations[i] = 0
            else:
                env_skill_durations[i] += 1
            
            # 更新环境状态跟踪
            env_steps[i] += 1
            env_rewards[i] += rewards[i]
            
            if current_agent_info['skill_changed']:
                agent.log_skill_distribution(
                    current_agent_info['team_skill'],
                    current_agent_info['agent_skills'],
                    episode=n_episodes
                )
            
            # 处理完成的环境
            if dones[i]:
                n_episodes += 1
                episode_rewards.append(env_rewards[i])
                logger.info(f"环境 {i} 完成: Episode {n_episodes}, 奖励: {env_rewards[i]:.2f}, 步数: {env_steps[i]}")
                
                # 重置环境状态跟踪
                env_steps[i] = 0
                env_rewards[i] = 0
        
        # 更新网络
        if total_steps > config.batch_size and total_steps % 128 == 0:
            if len(agent.low_level_buffer) >= agent.config.batch_size:
                try:
                    update_info = agent.update()
                    update_times += 1
                    if update_times % 10 == 0:
                        logger.info(f"更新 {update_times}, 总步数 {total_steps}, 高层损失: {update_info.get('coordinator_loss', 0):.4f}")
                except Exception as e:
                    logger.error(f"更新时出错: {e}")
        
        # 评估 (使用修复后的逻辑)
        if total_steps >= last_eval_step + config.eval_interval:
            eval_counts += 1
            logger.info(f"即将进行评估 #{eval_counts}, 当前步数: {total_steps}, 距离上次评估: {total_steps - last_eval_step} 步")
            
            # 进行评估 - 注意这里传递logger
            eval_reward, eval_std, eval_min, eval_max = evaluate(
                eval_vec_env, agent, n_episodes=config.eval_episodes, logger=logger
            )
            
            logger.info(f"评估 #{eval_counts} 完成: 平均奖励 {eval_reward:.2f} ± {eval_std:.2f}")
            
            # 保存最佳模型
            if eval_reward > best_reward:
                best_reward = eval_reward
                model_path = os.path.join(test_log_dir, 'best_model.pt')
                agent.save_model(model_path)
                logger.info(f"保存最佳模型，奖励: {best_reward:.2f}")
            
            # 更新上次评估步数
            last_eval_step = total_steps
        
        # 增加总步数
        total_steps += num_envs
        
        # 更新状态和观测
        states = next_states
        observations = next_observations
        
        # 打印进度
        if total_steps % 500 == 0:
            elapsed = time.time() - start_time
            logger.info(f"进度: {total_steps}/{config.total_timesteps} 步 ({total_steps/config.total_timesteps*100:.1f}%), 用时: {elapsed:.1f}s")
    
    # 训练结束
    elapsed = time.time() - start_time
    logger.info(f"训练完成! 总步数: {total_steps}, 总episodes: {n_episodes}, 总用时: {elapsed:.1f}s")
    logger.info(f"总共进行了 {eval_counts} 次评估，最佳奖励: {best_reward:.2f}")
    
    # 保存最终模型
    final_model_path = os.path.join(test_log_dir, 'final_model.pt')
    agent.save_model(final_model_path)
    logger.info(f"最终模型已保存到 {final_model_path}")
    
    # 关闭环境
    train_vec_env.close()
    eval_vec_env.close()
    
    # 进行最终评估 - 使用安全创建环境的函数
    logger.info("进行最终评估...")
    final_eval_env_fns = [make_env(
        scenario=scenario,
        n_uavs=n_uavs,
        n_users=n_users,
        user_distribution=user_distribution,
        channel_model=channel_model,
        render_mode=None,
        rank=0,
        seed=base_seed + 1000
    ) for _ in range(1)]
    
    # 使用DummyVecEnv进行最终评估，避免潜在的多进程问题
    final_eval_env = safe_create_vec_env(final_eval_env_fns, use_subproc=False, logger=logger)
    
    try:
        final_eval_reward, _, _, _ = evaluate(
            final_eval_env, 
            agent, 
            n_episodes=5, 
            logger=logger
        )
        logger.info(f"最终评估完成: 平均奖励 {final_eval_reward:.2f}")
    except Exception as e:
        logger.error(f"最终评估失败: {e}")
        logger.error(traceback.format_exc())
        final_eval_reward = best_reward  # 如果评估失败，使用之前的最佳奖励
    finally:
        # 确保环境被关闭
        try:
            final_eval_env.close()
        except:
            logger.warning("关闭最终评估环境时出错")
    
    return {
        "total_steps": total_steps,
        "episodes": n_episodes,
        "duration_seconds": elapsed,
        "best_reward": best_reward,
        "final_reward": final_eval_reward,
        "eval_counts": eval_counts,
        "average_time_per_step": elapsed / total_steps if total_steps > 0 else 0
    }

if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    try:
        result = quick_test()
        print("\n========= 测试结果摘要 =========")
        print(f"总步数: {result['total_steps']}")
        print(f"总episodes: {result['episodes']}")
        print(f"总用时: {result['duration_seconds']:.1f}秒")
        print(f"每步平均时间: {result['average_time_per_step']*1000:.2f}毫秒")
        print(f"评估次数: {result['eval_counts']}")
        print(f"最佳奖励: {result['best_reward']:.2f}")
        print(f"最终奖励: {result['final_reward']:.2f}")
        print("================================")
    finally:
        # 关闭日志系统
        shutdown_logging()
