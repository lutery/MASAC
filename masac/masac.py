"""MASAC.

Implementation is derived from [cleanRL](https://github.com/vwxyzjn/cleanrl). The main changes are:
* Support for PettingZoo API;
* Parameter sharing between agents (shared critic, actor conditioned on ID).
"""
import argparse
import os
import random
import time
from distutils.util import strtobool
from typing import Dict

import einops
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pettingzoo import ParallelEnv
from pettingzoo.mpe import simple_spread_v3
from pettingzoo.utils.env import AgentID, ObsType
from torch.utils.tensorboard import SummaryWriter

from ma_buffer import Experience, MAReplayBuffer
from utils import extract_agent_id


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="MASAC",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="florian-felten",
                        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, default=1000000,
                        help="total timesteps of the experiments") # 设置全局训练的步数
    parser.add_argument("--buffer-size", type=int, default=int(1e4),
                        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256, 
                        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=5e3,
                        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
                        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=1e-3,
                        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--policy-frequency", type=int, default=1,
                        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1,  # Denis Yarats' implementation delays this by 2.
                        help="the frequency of updates for the target nerworks")
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="automatic tuning of the entropy coefficient")
    args = parser.parse_args()
    # fmt: on
    return args


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env: ParallelEnv):
        super().__init__()
        single_action_space = env.action_space(env.agents[0])
        # Global state, joint actions space -> ... -> Q value
        # env.state().shape 返回的事所有env智能体的全局状态，比如所有的位置、所有的速度
        # np.prod(single_action_space.shape) * env.num_agents：预测的动作维度 * 智能体的数量，看起来输入的全局状态+每个智能体的执行的动作
        self.fc1 = nn.Linear(np.array(env.state().shape).prod() + np.prod(single_action_space.shape) * env.num_agents, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1) # 预测全体的Q值 todo 是否有办法改写为为每一个env单独预测

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env: ParallelEnv):
        super().__init__()
        single_action_space = env.action_space(env.agents[0]) # 类似Box(low=-1.0, high=1.0, shape=(5,), dtype=float32)
        single_observation_space = env.observation_space(env.agents[0]) # 类似 Box(low=-inf, high=inf, shape=(18,), dtype=float64)
        # Local state, agent id -> ... -> local action
        self.fc1 = nn.Linear(np.array(single_observation_space.shape).prod() + 1, 256) # 这里的np.array(single_observation_space.shape).prod() 是直接获取obs的shape为18，其其中的+1是agent id
        self.fc2 = nn.Linear(256, 256)
        # 经过两个全连接特征提取层后，分别预测连续动作的均值和方差的log，用log事因为避免方差为负数
        self.fc_mean = nn.Linear(256, np.prod(single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(single_action_space.shape))
        # action rescaling 这两个是什么 todo
        self.register_buffer(
            "action_scale", torch.tensor((single_action_space.high - single_action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((single_action_space.high + single_action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x) # 网络输出的 log_std 是无界的，如果 log_std 输出一个极端值（比如 -100 或 +100），exp(log_std) 会变成 0 或无穷大，导致梯度爆炸或 NaN。
        log_std = torch.tanh(log_std) # SAC 算法中的一个数值稳定技巧，用 tanh 把它压缩到 [-1, 1]
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats 再线性映射到合法区间 [LOG_STD_MIN, LOG_STD_MAX]

        return mean, log_std

    def get_action(self, x):
        # 输入x（包含obs和agent id）
        mean, log_std = self(x)
        std = log_std.exp() # 恢复到 真正的std
        normal = torch.distributions.Normal(mean, std) # 构建概率分布
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1)) 采样动作
        y_t = torch.tanh(x_t) # 压缩动作到 -1 ～ 1
        action = y_t * self.action_scale + self.action_bias # 将动作恢复到动作空间
        # 这三行是 SAC 的变量替换公式（change of variables），计算经过 tanh 压缩和线性缩放后的实际动作的 log 概率。
        log_prob = normal.log_prob(x_t) # 类似离散动作一样，得到了执行的动作，则计算该动作对应的log概率值
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True) # 把 5 个独立动作维度的 log 概率相加，得到联合动作的 log 概率（假设各维度条件独立）：
        mean = torch.tanh(mean) * self.action_scale + self.action_bias # 将动作的均值恢复真实动作的空间
        # todo log_prob 主要用来做啥
        return action, log_prob, mean


def concat_id(local_obs: np.ndarray, id: AgentID) -> np.ndarray:
    """Concatenate the agent id to the local observation.

    Args:
        local_obs: the local observation
        id: the agent id to concatenate

    Returns: the concatenated observation

    """
    return np.concatenate([local_obs, np.array([extract_agent_id(id)], dtype=np.float32)])


if __name__ == "__main__":
    args = parse_args()
    run_name = f"Circle__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # device = torch.device("mps") if torch.backends.mps.is_available() else device

    # env setup
    '''
    `simpol_spread_v3` 是 MPE 中的一个**合作博弈**场景。地图上有 N 个智能体（粒子）和 N 个目标点，智能体需要协作移动以覆盖所有目标点。覆盖越多目标点 → 共享奖励越高。

    参数含义：

    | 参数 | 值 | 含义 |
    |------|-----|------|
    | `N` | 3 | 3 个智能体和 3 个目标点 |
    | `local_ratio` | 0.5 | 智能体的局部观测范围比例。0=完全全局观测，1=完全局部观测（只看视野内的东西） |
    | `max_cycles` | 25 | 每局最多 25 步，到步数自动截断（truncate） |
    | `continuous_actions` | `True` | 连续动作空间，每个智能体输出 5 维向量（2D 速度 + 2D 位置偏移 + 智能体间通讯）。设为 `False` 则变为离散动作 |

    观测空间：单个智能体的观测是自身位置、速度、与目标的相对位置、与其他智能体的相对位置、通讯信号等组成的 Box 空间向量。

    奖励：所有智能体共享同一个全局奖励——当前时刻所有目标距离最近智能体的距离之和的负值（距离越近奖励越大）。
    '''
    env = simple_spread_v3.parallel_env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=True)
    env.reset(seed=args.seed)
    # 看来动作空间和观察空间和gym基本差不多，这里获取的都是单个智能体的shape
    single_action_space = env.action_space(env.unwrapped.agents[0])
    single_observation_space = env.observation_space(env.unwrapped.agents[0])
    assert isinstance(single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(single_action_space.high[0]) # 因为是连续动作，所以这里获取连续动作的最大值

    actor = Actor(env).to(device) # 动作预测
    qf1 = SoftQNetwork(env).to(device) # Q值预测
    qf2 = SoftQNetwork(env).to(device)
    qf1_target = SoftQNetwork(env).to(device)
    qf2_target = SoftQNetwork(env).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning 这里应该是确认熵的权重是按照训练的来还是固定值
    if args.autotune:
        # todo 这里存在一个瑕疵，这里的目标熵是单智能体的熵，但是在计算的时候是把所有智能体的动作的log概率加起来作为联合动作的log概率，所以这里的目标熵应该是单智能体熵的智能体数量倍才对
        target_entropy = -torch.prod(torch.Tensor(single_action_space.shape).to(device)).item() # 这里计算最佳的熵，定理，最佳的熵就是这么计算的
        log_alpha = torch.zeros(1, requires_grad=True, device=device) # 创建熵权重log值，这里同样是为了防止熵变成负数
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    single_observation_space.dtype = np.float32 # 修改观察空间的数据类型
    rb = MAReplayBuffer(
        global_obs_shape=env.state().shape,
        local_obs_shape=single_observation_space.shape,
        action_dim=single_action_space.shape[0],
        num_agents=env.max_num_agents,
    ) # 重放缓冲区
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, info = env.reset(seed=args.seed)
    global_return = 0.0 # 计算总体的回报值
    global_obs: np.ndarray = env.state() # 获取全局obs
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        # 
        if global_step < args.learning_starts:
            # 可能出现的所有智能体列表（全集，固定不变），所以这里一定返回3
            # env.action_space(agent) 指定特定的智能体进行动作采样
            # actions 返回每个环境采样的东走
            actions: Dict[str, np.ndarray] = {agent: env.action_space(agent).sample() for agent in env.possible_agents}
        else:
            # 当到达一定步数的时候，就开始使用Actor模型进行预测
            actions: Dict[str, np.ndarray] = {}
            with torch.no_grad():
                for agent_id in env.possible_agents:
                    # 将 观察和agent id组合在一起，让动作预测能够结合观察和
                    obs_with_id = torch.Tensor(concat_id(obs[agent_id], agent_id)).to(device)
                    act, _, _ = actor.get_action(obs_with_id.unsqueeze(0)) # 预测动作
                    act = act.detach().cpu().numpy()
                    actions[agent_id] = act.flatten() # 将预测的动作放到actions

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs: Dict[str, ObsType]
        rewards: Dict[str, float]
        next_obs, rewards, terminateds, truncateds, infos = env.step(actions) # 执行动作

        terminated: bool = any(terminateds.values()) 
        truncated: bool = any(truncateds.values())

        # TRY NOT TO MODIFY: save data to replay buffer; handle `final_observation`
        real_next_obs = next_obs
        # TODO PZ doesn't have that yet
        # if truncated:
        #     real_next_obs = infos["final_observation"].copy()
        # 将动作存储到重访缓冲区
        rb.add(
            global_obs=global_obs,
            local_obs=obs,
            joint_actions=np.array(list(actions.values())).flatten(), # 存储动作采用的是展平 动作存储
            reward=np.array(list(rewards.values())).sum(), # todo 为啥奖励要这么存储，将奖励计算总和
            next_global_obs=env.state(),
            next_local_obs=real_next_obs,
            terminated=terminated, # todo 为啥结束要这么存储，将中断计算一个any bool
        )

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        global_return += sum(rewards.values()) #更新全局回报
        global_obs = env.state()  # 得到最新的全局obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts: # 只有收集到一定的数据后才开始训练
            data: Experience = rb.sample(args.batch_size, to_tensor=True, device=device, add_id_to_local_obs=True)
            with torch.no_grad():
                # Computes q value from target networks
                # flatten data.next_local_obs to forward for all agents at once
                flattened_next_local_obs = data.next_local_obs.reshape(
                    (args.batch_size * env.unwrapped.max_num_agents, np.prod(single_observation_space.shape) + 1)
                ) # 展平并行env之间的动作
                # forward pass to get next actions and log probs 预测下一个obs该执行的动作
                next_state_actions, next_state_log_pi, _ = actor.get_action(flattened_next_local_obs)
                next_joint_actions = next_state_actions.reshape(
                    (args.batch_size, np.prod(single_action_space.shape) * env.unwrapped.max_num_agents)
                )# 这里又将动作重新展平为每个env的动作维度 * 智能体数量的形式，准备输入到Q网络中，看起来Q网络就是根据全局obs+所有智能体的动作来预测全局的Q值的
                # Sums the log probs of the actions in the agent dimension to get the joint log prob
                next_state_log_pi = einops.reduce(
                    next_state_log_pi.reshape((args.batch_size, env.unwrapped.max_num_agents)), "b a -> b ()", "sum"
                ) # 把每个智能体独立的 log 概率加起来，得到联合动作的 log 概率，参与后面的 SAC 的 Bellman 方程中目标 Q 值需要减去熵奖励项

                # SAC Bellman equation 
                qf1_next_target = qf1_target(data.next_global_obs, next_joint_actions) # 预测下一个状态的Q值，全局Q值，所以维度是1
                qf2_next_target = qf2_target(data.next_global_obs, next_joint_actions) # 预测下一个状态的Q值，全局Q值，所以维度是1
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi # 为什么要减去可以看md文档
                next_q_value = data.rewards.flatten() + (1 - data.terminateds.flatten()) * args.gamma * (
                    min_qf_next_target
                ).view(-1)

            # Computes q loss 计算当前动作的Q值，并与目标Q值计算MSE损失
            qf1_a_values = qf1(data.global_obs, data.joint_actions).view(-1)
            qf2_a_values = qf2(data.global_obs, data.joint_actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # 训练Q值网络，这里的Q值网络是全局OBS的Q值
            # 看起来局部Q值仅用在预测智能体执行的动作
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support 每隔一定的间隔才训练一次动作策略网络，来补偿策略更新的延迟
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    # flatten data.local_obs to forward for all agents at once
                    flattened_local_obs = data.local_obs.reshape(
                        (args.batch_size * env.unwrapped.max_num_agents, np.prod(single_observation_space.shape) + 1)
                    ) # 展平并行env的obs，方便后续的动作预测
                    # forward pass to get next actions and log probs
                    pi, log_pi, _ = actor.get_action(flattened_local_obs)
                    next_joint_actions = pi.reshape(
                        (args.batch_size, np.prod(single_action_space.shape) * env.unwrapped.max_num_agents)
                    ) # 将预测的动作重新展平为每个env的动作维度 * 智能体数量的形式，准备输入到Q网络中，看起来Q网络就是根据全局obs+所有智能体的动作来预测全局的Q值的
                    # Sums the log probs of the actions in the agent dimension to get the joint log prob
                    # TODO check if this is correct
                    log_pi = einops.reduce(
                        log_pi.reshape((args.batch_size, env.unwrapped.max_num_agents)), "b a -> b ()", "sum"
                    ) # 把每个智能体独立的 log 概率加起来，得到联合动作的 log 概率，参与后面的 SAC 的 Bellman 方程中目标 Q 值需要减去熵奖励项

                    # SAC pi update
                    qf1_pi = qf1(data.global_obs, next_joint_actions) # 全局obs+所有智能体的动作来预测全局的Q值的
                    qf2_pi = qf2(data.global_obs, next_joint_actions) # 全局obs+所有智能体的动作来预测全局的Q值的
                    min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean() # 得到当前状态下预测的Q值，和之前一样减去熵奖励项，得到动作策略的损失函数（但是由于是最大化Q值所以这里用了负号）
                    # 而 (alpha * log_pi) 依旧是熵奖励项，鼓励策略保持足够的随机性，避免过早收敛到次优策略

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune: # 自动调整熵权重，对于这条线是通过
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(flattened_local_obs) # 根据当前状态预测动作概率的log值
                            log_pi = einops.reduce(
                                log_pi.reshape((args.batch_size, env.unwrapped.max_num_agents)), "b a -> b ()", "sum"
                            )# 将每个智能体独立的 log 概率加起来，得到联合动作的 log 概率，参与后面的 SAC 的 Bellman 方程中目标 Q 值需要减去熵奖励项
                        alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean() # 具体的数学推论和直观推论看md文档

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks 同步权重到目标网络
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0: # 记录各种训练记录
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

        # 因为是并行环境，所以只要有一个智能体的episode结束了，就重置环境
        # 这里是简化了处理，不用考虑哪个智能体结束了，对训练数据的进行重新调整
        if terminated or truncated: # 如果有任何一个智能体的episode结束了，就重置环境
            obs, info = env.reset()
            writer.add_scalar("charts/return", global_return, global_step)
            global_return = 0.0
            global_obs = env.state()

    # Saves the trained actor for execution
    torch.save(actor.state_dict(), "actor.pth")

    env.close()
    writer.close()
