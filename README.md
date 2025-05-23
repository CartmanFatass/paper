# HMASD: 层次化多智能体技能发现 - 多无人机基站场景

这个项目是论文《Hierarchical Multi-Agent Skill Discovery》(Yang 等，2023) 中提出的 HMASD 算法的 PyTorch 实现，并将其应用于多无人机基站服务场景。HMASD 是一个用于多智能体强化学习的层次化技能发现方法，能够同时学习团队技能和个体技能，并有效地应用于稀疏奖励环境。

## 算法概述

HMASD 包含三个主要组件：

1. **技能协调器 (Skill Coordinator)**：高层策略，使用 Transformer 结构为智能体分配团队技能和个体技能。
2. **技能发现器 (Skill Discoverer)**：低层策略，根据分配的技能为每个智能体选择原始动作。
3. **技能判别器 (Skill Discriminator)**：包括团队技能判别器和个体技能判别器，用于生成内在奖励，促进技能多样性。

## 项目结构

```
.
├── hmasd/                  # HMASD 核心实现
│   ├── agent.py            # HMASD 代理实现
│   ├── networks.py         # 网络模型定义
│   └── utils.py            # 工具函数
├── envs/                   # 环境实现
│   └── pettingzoo/         # 基于PettingZoo的环境
│       ├── uav_env.py      # 基础UAV环境
│       ├── scenario1.py    # 场景1：独立基站模式
│       └── scenario2.py    # 场景2：协作组网模式
├── config.py               # 配置参数
├── main.py                 # 训练和评估的入口点
├── requirements.txt        # 依赖包列表
└── README.md               # 项目文档
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 多无人机基站场景

本项目实现了两个多无人机基站服务场景：

### 场景1：独立基站模式

- 所有无人机直接作为基站服务地面用户
- 不考虑回程和中继
- 优化目标是最大化用户覆盖率和服务质量

### 场景2：协作组网模式

- 无人机可根据情况合作组网，分别担任基站以及中继
- 需要回程到地面基站
- 跳数最多为3-5可调
- 优化目标是最大化用户覆盖率、服务质量和网络连通性

## 使用说明

### 1. 训练模型

```bash
# 场景1：独立基站模式
python main.py --mode train --scenario 1 --n_uavs 5 --n_users 50 --user_distribution uniform

# 场景2：协作组网模式
python main.py --mode train --scenario 2 --n_uavs 8 --n_users 100 --max_hops 3 --user_distribution hotspot
```

参数说明：
- `--mode`: 训练模式 (`train`) 或评估模式 (`eval`)
- `--scenario`: 场景选择 (1=独立基站模式, 2=协作组网模式)
- `--n_uavs`: 无人机数量
- `--n_users`: 用户数量
- `--max_hops`: 最大跳数 (仅用于场景2)
- `--user_distribution`: 用户分布类型 (`uniform`, `cluster`, `hotspot`)
- `--channel_model`: 信道模型 (`free_space`, `urban`, `suburban`)
- `--model_path`: 模型保存/加载路径（默认：models/hmasd_model.pt）

### 2. 评估模型

```bash
# 场景1：独立基站模式
python main.py --mode eval --scenario 1 --model_path models/hmasd_model.pt --render --eval_episodes 5

# 场景2：协作组网模式
python main.py --mode eval --scenario 2 --model_path models/hmasd_model.pt --render --eval_episodes 5
```

参数说明：
- `--model_path`: 要加载的模型路径
- `--render`: 是否渲染环境
- `--eval_episodes`: 评估的episode数量（默认：10）

## 配置参数

主要配置参数在 `config.py` 中定义，包括：

- 环境参数：智能体数量、状态维度、观测维度、动作维度
- HMASD参数：团队技能数量、个体技能数量、技能分配间隔
- 网络参数：隐藏层大小、嵌入维度、Transformer层数等
- PPO参数：折扣因子、GAE参数、裁剪参数等
- 损失权重：外部奖励权重、技能判别器奖励权重等

## 环境参数调整

可以通过命令行参数调整环境的各种参数：

1. **无人机数量**：通过 `--n_uavs` 参数设置，建议范围为3-20
2. **用户数量**：通过 `--n_users` 参数设置，建议范围为20-200
3. **用户分布**：
   - `uniform`: 均匀分布
   - `cluster`: 聚类分布
   - `hotspot`: 热点分布
4. **信道模型**：
   - `free_space`: 自由空间路径损耗模型
   - `urban`: 城市环境路径损耗模型
   - `suburban`: 郊区环境路径损耗模型
5. **最大跳数**（仅场景2）：通过 `--max_hops` 参数设置，建议范围为3-5

## 引用

如果你使用了这个代码，请引用原论文：

```
@inproceedings{yangHMASD2023,
  title={Hierarchical Multi-Agent Skill Discovery},
  author={Yang, Mingyu and Yang, Yaodong and Lu, Zhenbo and Zhou, Wengang and Li, Houqiang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```

## 许可证

MIT
