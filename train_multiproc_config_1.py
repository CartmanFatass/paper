import os
import time
import numpy as np
import torch
import argparse
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import multiprocessing as mp
# from functools import partial # No longer needed for make_env directly
from logger import init_multiproc_logging, get_logger, shutdown_logging, LOG_LEVELS, set_log_level

# 导入 Stable Baselines3 的向量化环境
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env # Can also use this helper

# 导入论文中的配置
from config_1 import Config
from hmasd.agent import HMASDAgent
from envs.pettingzoo.scenario1 import UAVBaseStationEnv
from envs.pettingzoo.scenario2 import UAVCooperativeNetworkEnv
from envs.pettingzoo.env_adapter import ParallelToArrayAdapter

# Removed VectorizedEnvAdapter class

# 获取计算设备
def get_device(device_pref):
    """
    根据偏好选择计算设备
    
    参数:
        device_pref: 设备偏好 ('auto', 'cuda', 'cpu')
        
    返回:
        device: torch.device对象
    """
    if device_pref == 'auto':
        if torch.cuda.is_available():
            main_logger.info("检测到GPU可用，使用CUDA")
            return torch.device('cuda')
        else:
            main_logger.info("未检测到GPU，使用CPU")
            return torch.device('cpu')
    elif device_pref == 'cuda':
        if torch.cuda.is_available():
            main_logger.info("使用CUDA")
            return torch.device('cuda')
        else:
            main_logger.warning("请求使用CUDA但未检测到GPU，回退到CPU")
            return torch.device('cpu')
    else:  # 'cpu'或其他值
        main_logger.info("使用CPU")
        return torch.device('cpu')

# 创建环境函数 (修改后用于 SubprocVecEnv)
def make_env(scenario, n_uavs, n_users, user_distribution, channel_model, max_hops=None, render_mode=None, rank=0, seed=0):
    """
    创建环境实例的函数 (用于 SubprocVecEnv)

    参数:
        scenario: 场景编号 (1=基站模式, 2=协作组网模式)
        n_uavs: 无人机数量
        n_users: 用户数量
        user_distribution: 用户分布类型
        channel_model: 信道模型
        max_hops: 最大跳数 (仅用于场景2)
        render_mode: 渲染模式
        rank: 环境的索引 (用于设置不同的种子)
        seed: 基础随机种子

    返回:
        一个返回环境实例的函数
    """
    def _init():
        env_seed = seed + rank # 为每个并行环境设置不同的种子
        if scenario == 1:
            raw_env = UAVBaseStationEnv(
                n_uavs=n_uavs,
                n_users=n_users,
                user_distribution=user_distribution,
                channel_model=channel_model,
                render_mode=render_mode,
                seed=env_seed # 将种子传递给原始环境
            )
        elif scenario == 2:
            raw_env = UAVCooperativeNetworkEnv(
                n_uavs=n_uavs,
                n_users=n_users,
                max_hops=max_hops,
                user_distribution=user_distribution,
                channel_model=channel_model,
                render_mode=render_mode,
                seed=env_seed # 将种子传递给原始环境
            )
        else:
            raise ValueError(f"未知的场景: {scenario}")

        # 使用适配器包装环境，并传递种子
        env = ParallelToArrayAdapter(raw_env, seed=env_seed)
        return env

    return _init

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='使用论文《Hierarchical Multi-Agent Skill Discovery》中的超参数运行HMASD (多进程版本)')
    # 运行模式和环境参数
    parser.add_argument('--mode', type=str, default='train', help='运行模式: train或eval')
    parser.add_argument('--scenario', type=int, default=2, help='场景: 1=基站模式, 2=协作组网模式')
    parser.add_argument('--model_path', type=str, default='models/hmasd_multiproc_paper_config.pt', help='模型保存/加载路径')
    parser.add_argument('--log_dir', type=str, default='logs', help='日志目录')
    parser.add_argument('--log_level', type=str, default='info', 
                        choices=['debug', 'info', 'warning', 'error', 'critical'], 
                        help='日志级别 (debug=详细, info=信息, warning=警告, error=错误, critical=严重)')
    parser.add_argument('--console_log_level', type=str, default='error', 
                        choices=['debug', 'info', 'warning', 'error', 'critical'], 
                        help='控制台日志级别')
    parser.add_argument('--eval_episodes', type=int, default=10, help='评估的episode数量')
    parser.add_argument('--render', action='store_true', help='是否渲染环境')
    parser.add_argument('--device', type=str, default='auto', 
                        choices=['auto', 'cuda', 'cpu'], help='计算设备: auto=自动选择, cuda=GPU, cpu=CPU')

    # 环境参数
    parser.add_argument('--n_uavs', type=int, default=5, help='初始无人机数量')
    parser.add_argument('--n_users', type=int, default=50, help='用户数量')
    parser.add_argument('--max_hops', type=int, default=3, help='最大跳数 (仅用于场景2)')
    parser.add_argument('--user_distribution', type=str, default='uniform', 
                        choices=['uniform', 'cluster', 'hotspot'], help='用户分布类型')
    parser.add_argument('--channel_model', type=str, default='3gpp-36777',
                        choices=['free_space', 'urban', 'suburban','3gpp-36777'], help='信道模型')
    
    # 并行参数
    parser.add_argument('--num_envs', type=int, default=0, 
                        help='并行环境数量 (0=使用配置文件中的值)')
    parser.add_argument('--eval_rollout_threads', type=int, default=0, 
                        help='评估时的并行线程数 (0=使用配置文件中的值)')
    
    return parser.parse_args()

# 训练函数
def train(vec_env, eval_vec_env, config, args, device): # Add eval_vec_env parameter
    """
    训练HMASD代理 (多进程版本)

    参数:
        vec_env: 训练用的向量化环境实例
        eval_vec_env: 评估用的向量化环境实例
        config: 配置对象
        args: 命令行参数
        device: 计算设备
    """
    num_envs = vec_env.num_envs # Get num_envs from SubprocVecEnv
    main_logger.info(f"开始训练HMASD (多进程版本，使用 {num_envs} 个并行环境)...")

    # 更新环境维度 (从 SubprocVecEnv 获取)
    # 注意: SubprocVecEnv 没有 get_state_dim 方法。我们需要从适配器获取或推断。
    # 我们可以通过 get_attr 获取适配器的属性
    state_dim = vec_env.get_attr('state_dim')[0] # 获取第一个环境的 state_dim
    # obs_dim 可以从 observation_space 获取
    obs_shape = vec_env.observation_space.shape
    # obs_shape is (num_envs, n_uavs, obs_dim_per_uav) if ParallelToArrayAdapter returns (n_uavs, obs_dim)
    # Or obs_shape is (num_envs, obs_dim_flat) if adapter returns flattened obs
    # Based on ParallelToArrayAdapter, it returns (n_uavs, obs_dim), so obs_shape[2] is obs_dim_per_uav
    if len(obs_shape) == 3:
         obs_dim = obs_shape[2]
         n_uavs_check = obs_shape[1] # Should match config.n_agents
         main_logger.info(f"从 observation_space 推断: obs_dim={obs_dim}, n_uavs={n_uavs_check}")
         if n_uavs_check != config.n_agents:
              main_logger.warning(f"从 observation_space 推断的 n_uavs ({n_uavs_check}) 与配置 ({config.n_agents}) 不匹配。")
              # Fallback to getting from adapter attribute
              obs_dim = vec_env.get_attr('obs_dim')[0]
    else:
         # Fallback if shape is not as expected
         main_logger.warning("无法从 observation_space 推断 obs_dim，尝试从适配器属性获取。")
         obs_dim = vec_env.get_attr('obs_dim')[0]

    config.update_env_dims(state_dim, obs_dim)
    main_logger.info(f"更新配置: state_dim={state_dim}, obs_dim={obs_dim}")

    # 创建日志目录
    log_dir = os.path.join(args.log_dir, f"sb3_multiproc_paper_config_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    model_dir = os.path.dirname(args.model_path)
    os.makedirs(model_dir, exist_ok=True)
    
    # 创建HMASD代理
    agent = HMASDAgent(config, log_dir=log_dir, device=device)
    
    # 记录超参数
    agent.writer.add_text('Parameters/n_agents', str(config.n_agents), 0)
    agent.writer.add_text('Parameters/n_Z', str(config.n_Z), 0)
    agent.writer.add_text('Parameters/n_z', str(config.n_z), 0)
    agent.writer.add_text('Parameters/k', str(config.k), 0)
    agent.writer.add_text('Parameters/gamma', str(config.gamma), 0)
    agent.writer.add_text('Parameters/lambda_e', str(config.lambda_e), 0)
    agent.writer.add_text('Parameters/lambda_D', str(config.lambda_D), 0)
    agent.writer.add_text('Parameters/lambda_d', str(config.lambda_d), 0)
    agent.writer.add_text('Parameters/lambda_h', str(config.lambda_h), 0)
    agent.writer.add_text('Parameters/lambda_l', str(config.lambda_l), 0)
    agent.writer.add_text('Parameters/hidden_size', str(config.hidden_size), 0)
    agent.writer.add_text('Parameters/lr', str(config.lr_coordinator), 0)
    agent.writer.add_text('Parameters/num_envs', str(num_envs), 0) # Use num_envs variable

    # 训练变量
    total_steps = 0
    n_episodes = 0
    max_episodes = config.total_timesteps // config.buffer_size  # 估计的最大episode数量
    episode_rewards = []
    update_times = 0
    best_reward = float('-inf')
    last_eval_step = 0  # 跟踪上次评估的步数
    
    # 高层样本累积检测变量
    high_level_samples_collected_total = 0  # 总共收集的高层样本数
    last_check_total_steps = 0              # 上次检查时的总步数
    last_check_hl_samples = 0               # 上次检查时的高层样本数
    last_high_level_buffer_size = 0         # 上次检查时的高层缓冲区大小
    check_interval_steps = config.buffer_size * num_envs  # 检查间隔步数
    warning_threshold_ratio = 0.1  # 如果实际样本数少于预期的10%，则发出警告
    error_threshold_steps = config.k * num_envs * 10  # 足够执行10个完整技能周期的步数
    
    # 记录训练开始时间
    start_time = time.time()

    # 重置所有环境 (使用 SubprocVecEnv)
    # SubprocVecEnv.reset() 只返回 observations
    # 我们需要通过 env_method 获取初始状态
    main_logger.info("重置并行环境...")
    results = vec_env.env_method('reset') # This calls reset on each env in parallel
    observations = np.array([res[0] for res in results]) # Shape: (num_envs, n_uavs, obs_dim)
    initial_infos = [res[1] for res in results]
    # Use agent.config.state_dim for default state shape
    states = np.array([info.get('state', np.zeros(agent.config.state_dim)) for info in initial_infos]) # Extract initial states, provide default
    main_logger.info(f"环境已重置。观测形状: {observations.shape}, 状态形状: {states.shape}")

    # 环境状态跟踪
    env_steps = np.zeros(num_envs, dtype=int)  # 每个环境的步数
    env_rewards = np.zeros(num_envs)  # 每个环境的累积奖励
    env_skill_durations = np.zeros(num_envs, dtype=int)  # 每个环境当前技能的持续时间
    # env_dones is handled by SubprocVecEnv's return value
    
    # 训练循环
    while total_steps < config.total_timesteps:
        # 代理为所有环境选择动作 (假设 agent.step 可以处理批处理或需要循环)
        # HMASDAgent.step 似乎是为单个环境设计的，我们需要调整
        # 选项1: 修改 HMASDAgent.step 以处理批处理 (复杂)
        # 选项2: 在这里循环调用 HMASDAgent.step (简单，但可能不是最高效)
        # 我们将使用选项2，因为它需要对代理进行最少的更改

        all_actions_list = [] # 存储每个环境的动作
        all_agent_infos_list = [] # 存储每个环境的代理信息

        for i in range(num_envs):
            # 代理选择动作
            # 注意: agent.step 可能需要当前环境的步数，我们用 env_steps[i]
            actions, agent_info = agent.step(states[i], observations[i], env_steps[i], deterministic=False)
            all_actions_list.append(actions)
            all_agent_infos_list.append(agent_info)

            # 如果技能发生变化，记录技能分布 (移到存储经验之后)
            # if agent_info['skill_changed']:
            #     agent.log_skill_distribution(...)

        # 将动作列表转换为 NumPy 数组
        actions_array = np.array(all_actions_list) # Shape: (num_envs, n_uavs, action_dim)

        # 执行动作 (使用 SubprocVecEnv)
        # 返回: next_observations, rewards, dones, infos
        next_observations, rewards, dones, infos = vec_env.step(actions_array)

        # 从 infos 提取 next_states
        # ParallelToArrayAdapter 的 step 返回的 info 包含 'next_state'
        next_states = np.array([info.get('next_state', np.zeros(state_dim)) for info in infos])

        # 更新环境状态和存储经验
        for i in range(num_envs):
            # 获取当前环境的代理信息
            current_agent_info = all_agent_infos_list[i]

            # 存储经验
            # 注意: dones[i] 是整个环境的 done 标志 (terminated or truncated)
            # 使用自己维护的技能持续时间计时器，而不是agent的共享计时器
            skill_timer_value = env_skill_durations[i]
            
            # 在存储转换前记录技能变化计时器状态（对调试高层缓冲区问题很重要）
            # 同时检查是否即将产生高层样本
            will_store_high_level = (skill_timer_value == config.k - 1 or dones[i])
            
            if dones[i]:
                main_logger.debug(f"环境 {i} episode终止，技能变化计时器={skill_timer_value}, k={agent.config.k}")
                
            # 记录存储前的高层缓冲区大小
            pre_store_high_level_buffer_size = len(agent.high_level_buffer)
            
            # 记录存储前的高层缓冲区大小
            pre_store_high_level_buffer_size = len(agent.high_level_buffer)
            
            agent.store_transition(
                states[i], next_states[i], observations[i], next_observations[i],
                actions_array[i], rewards[i], dones[i], current_agent_info['team_skill'],
                current_agent_info['agent_skills'], current_agent_info['action_logprobs'],
                log_probs=current_agent_info['log_probs'],
                skill_timer_for_env=skill_timer_value,  # 使用环境特定的技能计时器
                env_id=i  # 环境ID参数
            )
            
            # 检测是否已存储高层经验，使用agent内部计数器而不是缓冲区大小来判断
            post_store_high_level_samples_total = agent.high_level_samples_total
            
            # 如果内部样本计数器增加，说明已成功添加样本
            if post_store_high_level_samples_total > high_level_samples_collected_total:
                # 更新本地跟踪的样本总数
                samples_added = post_store_high_level_samples_total - high_level_samples_collected_total
                high_level_samples_collected_total = post_store_high_level_samples_total
                
                # 记录高层经验存储的时机和原因
                if will_store_high_level:
                    if skill_timer_value == config.k - 1:
                        reason = "技能周期结束"
                    else:  # dones[i] == True
                        reason = "环境终止"
                    
                    current_buffer_size = len(agent.high_level_buffer)
                    main_logger.info(f"环境 {i} 存储了 {samples_added} 个高层经验，原因: {reason}，当前高层缓冲区大小: {current_buffer_size}, 累积总数: {high_level_samples_collected_total}")
                else:
                    current_buffer_size = len(agent.high_level_buffer)
                    main_logger.warning(f"环境 {i} 在预期之外存储了 {samples_added} 个高层经验，技能计时器值: {skill_timer_value}，当前高层缓冲区大小: {current_buffer_size}")
            
            # 检查存储后高层缓冲区大小是否增加，判断是否存储了高层经验
            #post_store_high_level_buffer_size = len(agent.high_level_buffer)
            #if post_store_high_level_buffer_size > pre_store_high_level_buffer_size:
                # 高层经验已存储，更新计数
            #    high_level_samples_collected_total += 1
                
                # 记录高层经验存储的时机和原因
            #    if will_store_high_level:
            #        if skill_timer_value == config.k - 1:
            #            reason = "技能周期结束"
            #        else:  # dones[i] == True
            #            reason = "环境终止"
                    
            #        print(f"环境 {i} 存储了高层经验，原因: {reason}，当前高层缓冲区大小: {post_store_high_level_buffer_size}")
            #    else:
            #        print(f"警告: 环境 {i} 在预期之外存储了高层经验，技能计时器值: {skill_timer_value}，当前高层缓冲区大小: {post_store_high_level_buffer_size}")
            
            # 在存储转换后更新技能持续时间
            if dones[i]:
                # 如果环境终止，下一个episode的技能计时器应该重置为0
                env_skill_durations[i] = 0
            elif skill_timer_value == config.k - 1:
                # 如果当前技能周期刚好结束（使用了k步），重置技能计时器开始新的周期
                # 这是关键修复：确保每k步后技能计时器重置，而不是继续增长
                env_skill_durations[i] = 0
                if not will_store_high_level:
                    main_logger.warning(f"环境 {i} 的技能周期已结束 (达到 k-1={config.k-1})，但似乎没有存储高层经验")
            elif current_agent_info['skill_changed']:
                # 如果当前步中技能发生了变化，技能计时器应该重置为0
                # 但当前步使用的是变化前的技能，所以已经使用了正确的值存储经验
                env_skill_durations[i] = 0
            else:
                # 如果技能继续使用，增加计时器
                env_skill_durations[i] += 1

            # 更新环境状态跟踪
            env_steps[i] += 1
            env_rewards[i] += rewards[i]
            
            # 注意：total_steps不应该在这个循环内更新，而是在循环外部更新一次

            # 如果技能发生变化，记录技能分布
            if current_agent_info['skill_changed']:
                 agent.log_skill_distribution(
                     current_agent_info['team_skill'],
                     current_agent_info['agent_skills'],
                     episode=n_episodes # 使用当前的 episode 计数
                 )

            # 如果环境完成 (done[i] is True)
            if dones[i]:
                n_episodes += 1
                episode_rewards.append(env_rewards[i])

                # 记录 episode 奖励到 TensorBoard
                agent.training_info['episode_rewards'].append(env_rewards[i])
                agent.writer.add_scalar('Reward/episode_reward', env_rewards[i], n_episodes)
                agent.writer.add_scalar('Reward/episode_length', env_steps[i], n_episodes)

                main_logger.info(f"环境 {i}/{num_envs} 完成: Episode {n_episodes}, 奖励: {env_rewards[i]:.2f}, 步数: {env_steps[i]}")

                # 重置该环境的状态跟踪 (环境已在 SubprocVecEnv 内部自动重置)
                env_steps[i] = 0
                env_rewards[i] = 0
                # 注意: SubprocVecEnv 在 dones[i] 为 True 时返回的 next_observations[i] 和 infos[i]
                # 对应于重置后的新 episode 的初始状态。我们需要确保状态也正确更新。
                # next_states[i] 应该已经是重置后的状态 (从 info 中提取)。
                # observations[i] 将在下一次循环开始时使用 next_observations[i]。

                # 检查奖励统计和绘图逻辑是否仍然适用
                if len(episode_rewards) >= 10:
                    recent_rewards = episode_rewards[-10:]
                    avg_reward = np.mean(recent_rewards)
                    std_reward = np.std(recent_rewards)
                    max_reward = np.max(recent_rewards)
                    min_reward = np.min(recent_rewards)

                    agent.writer.add_scalar('Reward/avg_reward_10', avg_reward, n_episodes)
                    agent.writer.add_scalar('Reward/std_reward_10', std_reward, n_episodes)
                    agent.writer.add_scalar('Reward/max_reward_10', max_reward, n_episodes)
                    agent.writer.add_scalar('Reward/min_reward_10', min_reward, n_episodes)

                    main_logger.info(f"最近10个episodes: 平均奖励 {avg_reward:.2f} ± {std_reward:.2f}, 最大/最小: {max_reward:.2f}/{min_reward:.2f}")

                if n_episodes % 10 == 0:
                    plt.figure(figsize=(10, 5))
                    plt.plot(episode_rewards)
                    plt.title('Episode Rewards')
                    plt.xlabel('Episode')
                    plt.ylabel('Reward')
                    plt.savefig(os.path.join(log_dir, 'rewards.png'))
                    plt.close()

            # 更新网络 (基于总步数)
            # 注意: total_steps 现在每个循环增加 num_envs 次
            # 我们可能需要调整更新频率或 total_steps 的计算方式
            # 让我们保持 total_steps 每次循环增加 num_envs
            # Check if enough steps have passed to potentially fill the buffer
            if total_steps // num_envs > 0 and (total_steps // num_envs) % (config.buffer_size // num_envs) == 0:
                 # Check if the low_level_buffer has enough samples for a batch update
                 # (agent.update() itself checks buffer sizes internally, but this adds an explicit check)
                 if len(agent.low_level_buffer) >= agent.config.batch_size:
                     try:
                         update_info = agent.update()
                         update_times += 1
                         elapsed = time.time() - start_time

                         # 打印训练信息 (奖励统计现在基于完成的 episodes) - 每10240步打印一次
                         if total_steps % 10240 == 0:
                             main_logger.info(f"更新 {update_times}, 总步数 {total_steps} (来自 {num_envs} 个并行环境), "
                                  f"高层损失 {update_info['coordinator_loss']:.4f}, "
                                  f"低层损失 {update_info['discoverer_loss']:.4f}, "
                                  f"判别器损失 {update_info['discriminator_loss']:.4f}, "
                                  f"已用时间 {elapsed:.2f}s")
                     except ValueError as e:
                         main_logger.error(f"错误: {e}")
                         main_logger.error(f"错误类型: {type(e).__name__}")
                         main_logger.error(f"捕获到异常，这可能是因为update方法的返回值结构变化导致的。")
                         # 仍然增加update_times，以便于恢复训练
                         update_times += 1
                 else:
                     main_logger.info(f"步骤 {total_steps}: 缓冲区未满，跳过更新。")

            # 加强高层样本的累积情况监控
            if total_steps >= last_check_total_steps + check_interval_steps:
                # 获取当前高层缓冲区大小
                current_high_level_buffer_size = len(agent.high_level_buffer)
                
                # 从agent获取总收集的高层样本数(现在总是准确的，不受缓冲区满的影响)
                current_high_level_samples_total = agent.high_level_samples_total
                
                # 计算自上次检查以来的步数和增加的高层样本数
                steps_since_last_check = total_steps - last_check_total_steps
                parallel_steps_since_last_check = steps_since_last_check // num_envs
                samples_since_last_check = current_high_level_samples_total - last_check_hl_samples
                
                # 记录样本收集情况
                main_logger.info(f"高层样本收集统计: 当前总样本数={current_high_level_samples_total}, "
                               f"上次检查时样本数={last_check_hl_samples}, 新增样本数={samples_since_last_check}")
                
                # 更改理论期望样本计算：每k个时间步应该产生一个高层样本
                # 考虑到不同环境可能步调不一致，使用更宽松的期望值
                min_expected_environments = num_envs * 0.5  # 假设至少一半的环境应该贡献样本
                expected_samples_min = (parallel_steps_since_last_check / config.k) * min_expected_environments
                
                # 获取智能体中的样本收集统计
                high_level_samples_by_env = getattr(agent, 'high_level_samples_by_env', {})
                high_level_samples_by_reason = getattr(agent, 'high_level_samples_by_reason', {})
                
                # 统计各环境的技能计时器状态和奖励累积，以便排查问题
                env_timers_status = {env_id: agent.env_timers.get(env_id, -1) for env_id in range(num_envs)}
                env_rewards_status = {env_id: agent.env_reward_sums.get(env_id, -1.0) for env_id in range(num_envs)}
                
                # 分析样本收集情况
                contributing_envs = sum(1 for count in high_level_samples_by_env.values() if count > 0)
                
                # 记录当前检查点的统计信息（增加环境分析）
                main_logger.info(f"高层样本累积检查: 总步数: {total_steps}, 并行步数: {total_steps//num_envs}, "
                     f"自上次检查增加的高层样本数: {samples_since_last_check}, "
                     f"当前高层缓冲区大小: {current_high_level_buffer_size}/(需要{config.high_level_batch_size}), "
                     f"正在贡献的环境数: {contributing_envs}/{num_envs}")
                
                # 记录详细统计信息
                main_logger.info(f"高层样本收集原因: {high_level_samples_by_reason}")
                main_logger.info(f"环境技能计时器状态: {env_timers_status}")
                main_logger.info(f"环境奖励累积状态: {env_rewards_status}")
                
                # 检查是否有环境未贡献样本
                non_contributing_envs = [env_id for env_id in range(num_envs) 
                                         if high_level_samples_by_env.get(env_id, 0) == 0]
                if non_contributing_envs:
                    main_logger.warning(f"存在{len(non_contributing_envs)}个环境未贡献高层样本: {non_contributing_envs}")
                    
                    # 对未贡献样本的环境强制收集
                    for env_id in non_contributing_envs:
                        if hasattr(agent, 'force_high_level_collection'):
                            agent.force_high_level_collection[env_id] = True
                            main_logger.info(f"已标记环境{env_id}强制收集高层样本")
                
                # 如果自上次检查以来收集样本很少，发出警告
                if parallel_steps_since_last_check > config.k * 2 and samples_since_last_check < 1:
                    warning_msg = (
                        f"警告：高层经验累积速度不足！\n"
                        f"在过去的 {parallel_steps_since_last_check} 个并行时间步中 (总步数 {steps_since_last_check}), "
                        f"没有收集到高层样本。"
                    )
                    main_logger.warning(warning_msg)
                
                # 如果长时间内高层样本几乎没有增长，记录严重错误但不中断训练
                if parallel_steps_since_last_check > config.k * 5 and samples_since_last_check == 0:
                    error_msg = (
                        f"高层经验累积速度严重不足！\n"
                        f"在过去的 {parallel_steps_since_last_check} 个并行时间步中 (总步数 {steps_since_last_check}), "
                        f"仅收集到 {samples_since_last_check} 个高层样本。\n"
                        f"预期至少收集到约 {expected_samples_min:.1f} 个 (基于 k={config.k}, num_envs={num_envs})。\n"
                        f"当前高层缓冲区总大小: {current_high_level_buffer_size} (批次需求: {config.high_level_batch_size})。"
                    )
                    main_logger.error(error_msg)
                    
                    # 修改：不再中断训练，而是尝试通过循环强制收集
                    if hasattr(agent, 'force_high_level_collection'):
                        for env_id in range(num_envs):
                            agent.force_high_level_collection[env_id] = True
                        main_logger.info("已强制标记所有环境在下一个技能周期结束时贡献样本")
                
                # 更新检查点变量
                last_check_total_steps = total_steps
                last_check_hl_samples = current_high_level_samples_total
                last_high_level_buffer_size = current_high_level_buffer_size
                
                # 将高层样本累积情况记录到TensorBoard（增强记录指标）
                agent.writer.add_scalar('Buffer/high_level_buffer_size', current_high_level_buffer_size, total_steps)
                agent.writer.add_scalar('Buffer/high_level_samples_collected_total', current_high_level_samples_total, total_steps)
                agent.writer.add_scalar('Buffer/contributing_environments', contributing_envs, total_steps)
                if parallel_steps_since_last_check > 0:
                    samples_per_k_steps = (samples_since_last_check / parallel_steps_since_last_check) * config.k
                    agent.writer.add_scalar('Buffer/high_level_samples_per_k_steps', samples_per_k_steps, total_steps)
                    
                # 记录各种收集原因的比例
                for reason, count in high_level_samples_by_reason.items():
                    agent.writer.add_scalar(f'Buffer/collection_reason_{reason}', count, total_steps)
            
            # 评估 (基于总步数和上次评估的时间)
            if total_steps >= last_eval_step + config.eval_interval:
                 main_logger.info(f"即将进行评估，将评估 {config.eval_episodes} 个episodes...")
                 main_logger.info(f"当前步数: {total_steps}, 距离上次评估: {total_steps - last_eval_step} 步")
                 # 使用 eval_vec_env 进行评估
                 eval_reward, eval_std, eval_min, eval_max = evaluate(eval_vec_env, agent, config.eval_episodes)
                 main_logger.info(f"评估完成 ({config.eval_episodes} 个episodes): 平均奖励 {eval_reward:.2f} ± {eval_std:.2f}, 最大/最小: {eval_max:.2f}/{eval_min:.2f}")

                 # 保存最佳模型
                 if eval_reward > best_reward:
                     best_reward = eval_reward
                     agent.save_model(args.model_path)
                     main_logger.info(f"保存最佳模型，奖励: {best_reward:.2f}")
                 
                 # 更新上次评估步数
                 last_eval_step = total_steps

        # 在完成所有环境的一次迭代后，一次性增加总步数
        total_steps += num_envs

        # 更新状态和观测以进行下一次迭代
        states = next_states
        observations = next_observations

    main_logger.info(f"训练完成! 总步数: {total_steps}, 总episodes: {n_episodes}")
    main_logger.info(f"最佳奖励: {best_reward:.2f}")

    # 保存最终模型
    final_model_path = os.path.join(model_dir, 'hmasd_sb3_multiproc_paper_config_final.pt') # Update filename
    agent.save_model(final_model_path)
    main_logger.info(f"最终模型已保存到 {final_model_path}")
    
    return agent

# 评估函数
def evaluate(vec_env, agent, n_episodes=10, render=False):
    """
    评估HMASD代理 (使用 SubprocVecEnv)

    参数:
        vec_env: SubprocVecEnv 实例
        agent: HMASD代理实例
        n_episodes: 评估的episode数量 (总共要评估的episode数量)
        render: 是否渲染环境 (只渲染第一个环境)

    返回:
        mean_reward: 平均奖励
        std_reward: 奖励标准差
        min_reward: 最小奖励
        max_reward: 最大奖励
    """
    # 打印评估参数
    num_envs = vec_env.num_envs
    main_logger.info(f"开始评估: 目标完成 {n_episodes} 个episodes，使用 {num_envs} 个并行环境，是否渲染: {render}")
    
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

            # 执行动作并记录环境步进时间
            env_step_start = time.time()
            next_observations, rewards, dones, infos = vec_env.step(actions_array)
            env_step_end = time.time()
            env_step_times.append(env_step_end - env_step_start)
            
            # 每100步打印一次性能统计
            steps_done = len(agent_step_times)
            if steps_done % 100 == 0 and steps_done > 0:
                avg_agent_step = np.mean(agent_step_times[-100:])
                avg_env_step = np.mean(env_step_times[-100:])
                main_logger.info(f"评估性能统计 [{steps_done}步]: agent.step平均耗时: {avg_agent_step:.6f}秒/步, "
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
                            main_logger.error(f"渲染错误: {e}")
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
                                
                                main_logger.info(f"评估 Episode {completed_episodes+1}/{n_episodes} (来自环境 {i}), 奖励: {env_rewards[i]:.2f}, 步数: {env_steps[i]}, 服务用户数: {served_users}/{n_users} ({coverage_ratio:.2%})")
                            else:
                                main_logger.info(f"评估 Episode {completed_episodes+1}/{n_episodes} (来自环境 {i}), 奖励: {env_rewards[i]:.2f}, 步数: {env_steps[i]}")

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
                main_logger.warning("所有评估环境都已完成，但尚未达到目标 episode 数量。")
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
        
        main_logger.info("\n===== 评估技能分布统计 =====")
        main_logger.info(f"团队技能使用分布: {team_skill_probs}")
        
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
                main_logger.info(f"智能体 {i} 技能使用分布: {agent_skill_probs[i]}")
            
            if agent.config.n_agents > 3:
                main_logger.info(f"... (共 {agent.config.n_agents} 个智能体)")
    
        # 记录到TensorBoard
        if hasattr(agent, 'writer'):
            for z in range(agent.config.n_Z):
                agent.writer.add_scalar(f'Eval/TeamSkill_{z}_Probability', team_skill_probs[z], eval_step)
            
            for i in range(agent.config.n_agents):
                for z in range(agent.config.n_z):
                    agent.writer.add_scalar(f'Eval/Agent{i}_Skill_{z}_Probability', agent_skill_probs[i][z], eval_step)

    # 打印奖励统计信息
    if high_level_rewards:
        main_logger.info("\n===== 评估奖励统计 =====")
        mean_high_level = np.mean(high_level_rewards)
        main_logger.info(f"高层奖励平均值: {mean_high_level:.4f}")

    # 计算并打印性能统计
    eval_total_time = time.time() - eval_start_time
    total_steps_taken = sum(episode_lengths) if episode_lengths else 0
    if total_steps_taken > 0:
        avg_step_time = eval_total_time / total_steps_taken
        avg_agent_step_time = np.mean(agent_step_times) if agent_step_times else 0
        avg_env_step_time = np.mean(env_step_times) if env_step_times else 0
        
        main_logger.info("\n===== 评估性能统计 =====")
        main_logger.info(f"总评估时间: {eval_total_time:.2f}秒 (完成 {len(episode_rewards)} episodes, 共 {total_steps_taken} 步)")
        main_logger.info(f"每步平均耗时: {avg_step_time:.6f}秒")
        main_logger.info(f"agent.step 平均耗时: {avg_agent_step_time:.6f}秒/步 (占 {avg_agent_step_time/avg_step_time*100:.1f}%)")
        main_logger.info(f"env.step 平均耗时: {avg_env_step_time:.6f}秒/步 (占 {avg_env_step_time/avg_step_time*100:.1f}%)")
        main_logger.info(f"其他操作耗时: {avg_step_time - avg_agent_step_time - avg_env_step_time:.6f}秒/步")
        
        # 将性能指标也记录到TensorBoard中
        if hasattr(agent, 'writer'):
            agent.writer.add_scalar('Performance/total_eval_time', eval_total_time, eval_step)
            agent.writer.add_scalar('Performance/avg_step_time', avg_step_time, eval_step)
            agent.writer.add_scalar('Performance/avg_agent_step_time', avg_agent_step_time, eval_step)
            agent.writer.add_scalar('Performance/avg_env_step_time', avg_env_step_time, eval_step)
    
    main_logger.info(f"\n评估完成 ({len(episode_rewards)} episodes): 平均奖励 {mean_reward:.2f} ± {std_reward:.2f}, 平均步数: {mean_length:.2f}")

    return mean_reward, std_reward, min_reward, max_reward

# 主函数
def main():
    args = parse_args()
    
    # 创建日志目录
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 为训练会话创建固定的日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"hmasd_training_{timestamp}.log"
    
    # 初始化多进程日志系统
    file_level = LOG_LEVELS.get(args.log_level.lower(), logging.INFO)
    console_level = LOG_LEVELS.get(args.console_log_level.lower(), logging.WARNING)
    init_multiproc_logging(
        log_dir=args.log_dir, 
        log_file=log_file, 
        file_level=file_level, 
        console_level=console_level
    )
    
    # 获取main_logger实例
    global main_logger
    main_logger = get_logger("HMASD-Main")
    main_logger.info(f"多进程日志系统已初始化: 文件级别={args.log_level}, 控制台级别={args.console_log_level}")
    main_logger.info(f"日志文件: {os.path.join(args.log_dir, log_file)}")
    
    # 使用config_1.py中的配置（基于论文超参数）
    config = Config()
    
    # 获取计算设备
    device = get_device(args.device)
    
    # 确定并行环境数量
    num_envs = args.num_envs if args.num_envs > 0 else config.num_envs
    eval_rollout_threads = args.eval_rollout_threads if args.eval_rollout_threads > 0 else config.eval_rollout_threads
    
    main_logger.info(f"使用 {num_envs} 个并行训练环境和 {eval_rollout_threads} 个并行评估环境")
    
    # 创建环境构造函数列表 (使用修改后的 make_env)
    base_seed = config.seed if hasattr(config, 'seed') else int(time.time()) # Use config seed or time
    main_logger.info(f"基础种子: {base_seed}")

    train_env_fns = [make_env(
        scenario=args.scenario,
        n_uavs=args.n_uavs,
        n_users=args.n_users,
        user_distribution=args.user_distribution,
        channel_model=args.channel_model,
        max_hops=args.max_hops if args.scenario == 2 else None,
        render_mode=None,
        rank=i,
        seed=base_seed
    ) for i in range(num_envs)]

    eval_env_fns = [make_env(
        scenario=args.scenario,
        n_uavs=args.n_uavs,
        n_users=args.n_users,
        user_distribution=args.user_distribution,
        channel_model=args.channel_model,
        max_hops=args.max_hops if args.scenario == 2 else None,
        render_mode="human" if args.render and i == 0 else None, # 只在第一个评估环境中渲染
        rank=i,
        seed=base_seed + num_envs # Use different seeds for eval envs
    ) for i in range(eval_rollout_threads)]

    # 创建向量化环境 (使用 SubprocVecEnv)
    main_logger.info("创建 SubprocVecEnv...")
    train_vec_env = SubprocVecEnv(train_env_fns, start_method='spawn') # Use spawn for better compatibility
    eval_vec_env = SubprocVecEnv(eval_env_fns, start_method='spawn')
    main_logger.info("SubprocVecEnv 已创建。")

    # 更新配置中的智能体数量 (从环境中获取)
    # SubprocVecEnv wraps the adapter, which wraps the raw env.
    # We need to get n_uavs from the adapter.
    try:
         # Get n_uavs from the first environment's adapter instance
         n_agents_from_env = train_vec_env.get_attr('n_uavs')[0]
         config.n_agents = n_agents_from_env
         main_logger.info(f"从环境更新智能体数量: n_agents={config.n_agents}")
    except Exception as e:
         main_logger.warning(f"无法从环境获取 n_uavs: {e}. 使用命令行参数: {args.n_uavs}")
         config.n_agents = args.n_uavs # Fallback to argument

    # 获取 state_dim 和 obs_dim 用于打印 (已在 train 函数中处理)
    # state_dim_print = train_vec_env.get_attr('state_dim')[0]
    # obs_dim_print = train_vec_env.get_attr('obs_dim')[0]
    # print(f"环境已创建: n_agents={config.n_agents}, state_dim={state_dim_print}, obs_dim={obs_dim_print}")
    main_logger.info(f"使用论文中的超参数: n_Z={config.n_Z}, n_z={config.n_z}, k={config.k}, lambda_e={config.lambda_e}")

    if args.mode == 'train':
        # Pass eval_vec_env to the train function
        agent = train(train_vec_env, eval_vec_env, config, args, device)
    elif args.mode == 'eval':
        # 加载模型
        if not os.path.exists(args.model_path):
            main_logger.error(f"模型文件 {args.model_path} 不存在")
            return
        
        # 更新环境维度 (在 evaluate 函数内部处理，或在这里获取)
        # state_dim_eval = eval_vec_env.get_attr('state_dim')[0]
        # obs_dim_eval = eval_vec_env.get_attr('obs_dim')[0]
        # config.update_env_dims(state_dim_eval, obs_dim_eval) # Ensure config matches eval env if different

        # 创建日志目录
        log_dir = os.path.join(args.log_dir, f"eval_sb3_multiproc_paper_config_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建代理并加载模型
        agent = HMASDAgent(config, log_dir=log_dir, device=device)
        agent.load_model(args.model_path)
        
        # 记录模型配置
        agent.writer.add_text('Eval/model_path', args.model_path, 0)
        agent.writer.add_text('Eval/scenario', str(args.scenario), 0)
        agent.writer.add_text('Eval/n_agents', str(config.n_agents), 0)
        agent.writer.add_text('Eval/num_envs', str(eval_vec_env.num_envs), 0)

        # 评估模型
        evaluate(eval_vec_env, agent, n_episodes=args.eval_episodes, render=args.render)
    else:
        main_logger.error(f"未知的运行模式: {args.mode}")
    
    # 关闭环境
    train_vec_env.close()
    eval_vec_env.close()

# 全局队列引用，供子进程使用
_shared_log_queue = None

# 为SubprocVecEnv的子进程添加一个直接日志记录的辅助函数
def env_log(level, message, queue=None):
    """
    在子进程中记录日志的辅助函数
    
    参数:
        level: 日志级别 (如 logging.INFO)
        message: 日志消息
        queue: 日志队列 (如果为None，则使用全局队列)
    """
    global _shared_log_queue
    # 使用显式传入的队列或全局队列
    q = queue if queue is not None else _shared_log_queue
    
    try:
        import logging
        # 获取当前进程ID，用于区分不同的环境
        pid = os.getpid()
        
        if q:
            # 创建一个日志记录
            record = logging.LogRecord(
                name=f"Env-{pid}",
                level=level,
                pathname="",
                lineno=0,
                msg=message,
                args=(),
                exc_info=None
            )
            # 将记录直接放入队列
            q.put_nowait(record)
            return True
        else:
            # 如果队列不可用，至少打印到控制台
            print(f"[Env-{pid}] {message} (队列不可用)")
            return False
    except Exception as e:
        # 如果日志记录失败，至少打印到控制台
        pid = os.getpid()
        print(f"[Env-{pid}] {message} (日志记录失败: {e})")
        return False

if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    try:
        main()
    finally:
        # 确保关闭日志系统，刷新所有日志
        try:
            shutdown_logging()
            print("日志系统已关闭")
        except Exception as e:
            print(f"关闭日志系统时出错: {e}")
