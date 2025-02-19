import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from luxai_s3.params import EnvParams, env_params_ranges
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode
from torch.utils.tensorboard import SummaryWriter
import os
from process_observation import ProcessObservation, reshape_obs
import torch.nn.functional as F  
import time
# 用于产生随机的参数配置
def RandomizedEnvParams(env_params):
    """
    随机化 EnvParams 的所有配置项
    """
    for param, value_range in env_params_ranges.items():
        # 使用 `setattr` 将随机选中的值设置为实例的属性
        setattr(env_params, param, random.choice(value_range))

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gelu = nn.GELU()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // 16, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        x = self.gelu(self.conv1(x))
        x = self.conv2(x)
        x = x * self.se(x)
        return x + residual


class DoubleConeBlock(nn.Module):
    def __init__(self, channels):
        super(DoubleConeBlock, self).__init__()
        self.down_conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        self.up_conv = nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.res_blocks = nn.Sequential(*[ResBlock(channels) for _ in range(6)])
        self.gelu = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.gelu(self.down_conv(x))
        x = self.res_blocks(x)
        x = self.gelu(self.up_conv(x))
        return x + residual

class ConvHead(nn.Module):
    def __init__(self, in_channels, out_channels, num_agents, hidden_channels=256, num_layers=3):
        super().__init__()
        layers = []
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
                nn.GELU()
            ])
            in_channels = hidden_channels
        layers.append(nn.Conv2d(in_channels, out_channels * num_agents, kernel_size=1))
        self.net = nn.Sequential(*layers)
        self.out_channels = out_channels
        self.num_agents = num_agents

    def forward(self, x):
        x = self.net(x)
        x = x.mean(dim=[2, 3])  # 全局平均池化
        return x.view(-1, self.num_agents, self.out_channels)

class ConvCriticHead(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=256, num_layers=3):
        super().__init__()
        layers = []
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
                nn.GELU()
            ])
            in_channels = hidden_channels
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x.mean(dim=[2, 3])  # 全局平均池化得到标量

class ActorCriticNet(nn.Module):
    def __init__(self, input_shape, num_actions, device):
        super(ActorCriticNet, self).__init__()
        self.device = device
        self.num_agents = 16  # 智能体数量
        self.num_discrete_actions = 5  # 离散动作数量 (0-4)
        self.continuous_range = 8  

        self.initial_conv = nn.Conv2d(input_shape[0], 128, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(*[ResBlock(128) for _ in range(3)])
        self.double_cone = DoubleConeBlock(128)
        self.final_res_blocks = nn.Sequential(*[ResBlock(128) for _ in range(3)])
        self.gelu = nn.GELU()
        
       # 使用卷积头
        self.discrete_head = ConvHead(
            in_channels=128,
            out_channels=self.num_discrete_actions,
            num_agents=self.num_agents,
            hidden_channels=128,
            num_layers=3
        )
        
        self.continuous_head_1 = ConvHead(
            in_channels=128,
            out_channels=(self.continuous_range * 2 + 1),
            num_agents=self.num_agents,
            hidden_channels=128,
            num_layers=3
        )
        
        self.continuous_head_2 = ConvHead(
            in_channels=128,
            out_channels=(self.continuous_range * 2 + 1),
            num_agents=self.num_agents,
            hidden_channels=128,
            num_layers=3
        )

        self.critic_head = ConvCriticHead(
            in_channels=128,
            hidden_channels=256,
            out_channels = self.num_agents,
            num_layers=3
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # 重新排列维度
        x = self.gelu(self.initial_conv(x))
        x = self.res_blocks(x)
        x = self.double_cone(x)
        x = self.final_res_blocks(x)

        # 输出动作和状态值
        discrete_logits = self.discrete_head(x)
        continuous_logits_1 = self.continuous_head_1(x)
        continuous_logits_2 = self.continuous_head_2(x)
        
        state_value = self.critic_head(x)
        return discrete_logits, continuous_logits_1, continuous_logits_2, state_value
    
    def save_model(self, file_path):
        torch.save({
            'model_state_dict': self.state_dict(),
        }, file_path)

    def load_model(self, file_path, map_location=None):
        checkpoint = torch.load(file_path, weights_only=True, map_location=map_location)
        self.load_state_dict(checkpoint['model_state_dict'])
        return self
    
def train_actor_critic(model, optimizer, replay_buffer, batch_size, gamma=0.95, CLIP_EPS = 0.2, entropy_coef=0.001*0): #gama两个地方有
    # 从 ReplayBuffer 采样
    data = replay_buffer.sample(batch_size) 
    states, actions, rewards, dones, next_states, log_prob, advantages = data

    # 将每个张量转移到 GPU
    states = states.to(model.device)
    actions = actions.to(model.device)
    rewards = rewards.to(model.device)
    dones = dones.to(model.device)
    next_states = next_states.to(model.device)
    log_prob = log_prob.to(model.device)
    advantages = advantages.to(model.device)

    # 模型前向传播
    discrete_logits, continuous_logits_1, continuous_logits_2, state_values = model(states)
    _, _, _, next_state_values = model(next_states)

    # 计算目标值（TD目标）
    next_state_values = next_state_values.detach().squeeze(-1)  # 下一个状态值
    targets = rewards + next_state_values * gamma * (1 - dones.unsqueeze(-1))  # TD 目标值
    state_values = state_values.squeeze(-1)  # 当前状态值
    # TD误差
    td_error = targets - state_values

    # 1. 离散动作的策略损失 (第 0 维)
    log_probs_discrete = F.log_softmax(discrete_logits, dim=-1)
    selected_log_probs_discrete = log_probs_discrete.gather(
        -1, 
        actions[..., 0].unsqueeze(-1)
    )

    discrete_ratio = torch.exp((selected_log_probs_discrete.squeeze(-1) - log_prob[...,0]))
    discrete_surr1 = discrete_ratio * advantages
    discrete_surr2 = torch.clamp(discrete_ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
    discrete_loss = -torch.min(discrete_surr1, discrete_surr2).mean()

    # 2. 连续动作的策略损失 (第 1、2 维)
    log_probs_cont1 = F.log_softmax(continuous_logits_1, dim=-1)
    selected_log_probs_cont1 = log_probs_cont1.gather(
        -1, 
        (actions[..., 1] + model.continuous_range).unsqueeze(-1)
    )

    cont1_ratio = torch.exp((selected_log_probs_cont1.squeeze(-1) - log_prob[...,1]))
    cont1_surr1 = cont1_ratio * advantages
    cont1_surr2 = torch.clamp(cont1_ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
    continuous_loss_1 = -torch.min(cont1_surr1, cont1_surr2).mean()

    # 同理处理连续动作2
    log_probs_cont2 = F.log_softmax(continuous_logits_2, dim=-1)
    selected_log_probs_cont2 = log_probs_cont2.gather(
        -1, 
        (actions[..., 2] + model.continuous_range).unsqueeze(-1)
    )

    cont2_ratio = torch.exp((selected_log_probs_cont2.squeeze(-1) - log_prob[...,2]))
    cont2_surr1 = cont2_ratio * advantages
    cont2_surr2 = torch.clamp(cont2_ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
    continuous_loss_2 = -torch.min(cont2_surr1, cont2_surr2).mean()

    # 3. 值函数的损失 (Critic Loss)
    critic_loss = td_error.pow(2).mean() * 5 # MSE

    # 4. 策略熵
    # 离散动作的熵
    entropy_discrete = -(log_probs_discrete * torch.exp(log_probs_discrete)).sum(dim=-1).mean()
    
    # 连续动作的熵
    entropy_cont1 = -(log_probs_cont1 * torch.exp(log_probs_cont1)).sum(dim=-1).mean()
    entropy_cont2 = -(log_probs_cont2 * torch.exp(log_probs_cont2)).sum(dim=-1).mean()

    # 总熵
    total_entropy = entropy_discrete #+ entropy_cont1 + entropy_cont2

    # 总损失
    actor_loss = discrete_loss #+ continuous_loss_1 + continuous_loss_2
    total_loss = actor_loss + critic_loss - total_entropy * entropy_coef 

    # 反向传播和优化
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 添加梯度裁剪
    optimizer.step()

    # 打印调试信息
    print(discrete_loss.item())
    print(continuous_loss_1.item())
    print(continuous_loss_2.item())
    print(total_entropy.item())
    print(f"Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}, Entropy Loss: {total_entropy.item():.4f}, Total Loss: {total_loss.item():.4f}")

    return total_loss.item()


def select_action(model, state, temperature: float = 1):
    # 将输入状态移到 GPU（与模型相同的设备）
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(model.device)  # 添加 batch 维度，并移到模型所在的设备
    
    discrete_logits, continuous_logits_1, continuous_logits_2, critic_output = model(state) # torch.Size([1, 16, 5]) torch.Size([1, 16, 31]) torch.Size([1, 16, 31])
    discrete_logits = discrete_logits/temperature
    continuous_logits_1 = continuous_logits_1/temperature
    continuous_logits_2 = continuous_logits_2/temperature

    # 离散动作采样
    discrete_dist = Categorical(logits=discrete_logits)
    discrete_action = discrete_dist.sample()  # 采样离散动作

    # 连续动作采样
    continuous_dist_1 = Categorical(logits=continuous_logits_1)
    continuous_dist_2 = Categorical(logits=continuous_logits_2)
    continuous_action_1 = continuous_dist_1.sample() 
    continuous_action_2 = continuous_dist_2.sample() 

    discrete_action_log_prob = discrete_dist.log_prob(discrete_action)
    continuous_action_1_log_prob = continuous_dist_1.log_prob(continuous_action_1)
    continuous_action_2_log_prob = continuous_dist_2.log_prob(continuous_action_2)

    continuous_action_1 -= model.continuous_range  # 映射回去
    continuous_action_2 -= model.continuous_range  # 映射回去

    # 组合动作
    actions = torch.stack([discrete_action, continuous_action_1, continuous_action_2], dim=-1)  # [batch, agents, 3]
    log_prob = torch.stack([discrete_action_log_prob, continuous_action_1_log_prob, continuous_action_2_log_prob], dim=-1)
    # 将张量从 GPU 移到 CPU 并转换为 NumPy 数组
    actions = actions.detach().squeeze(0).cpu().numpy()  # 使用 .cpu() 移到 CPU 并转换为 NumPy 数组
    log_prob = log_prob.detach().squeeze(0).cpu().numpy()
    critic_output = critic_output.detach().squeeze(0).cpu().numpy()
    return actions, log_prob, critic_output

def select_action_deterministic(model, state):
    # 将输入状态移到 GPU（与模型相同的设备）
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(model.device)  # 添加 batch 维度，并移到模型所在的设备
    
    discrete_logits, continuous_logits_1, continuous_logits_2, _ = model(state) # torch.Size([1, 16, 5]) torch.Size([1, 16, 31]) torch.Size([1, 16, 31])

    # 离散动作选择
    discrete_action = torch.argmax(discrete_logits, dim=-1)

    # 连续动作选择
    continuous_action_1 = torch.argmax(continuous_logits_1, dim=-1)
    continuous_action_2 = torch.argmax(continuous_logits_2, dim=-1)

    continuous_action_1 -= model.continuous_range  # 映射回去
    continuous_action_2 -= model.continuous_range  # 映射回去

    # 组合动作
    actions = torch.stack([discrete_action, continuous_action_1, continuous_action_2], dim=-1)  # [batch, agents, 3]
    # 将张量从 GPU 移到 CPU 并转换为 NumPy 数组
    actions = actions.detach().squeeze(0).cpu().numpy()  # 使用 .cpu() 移到 CPU 并转换为 NumPy 数组
    return actions,0,0


if __name__ == "__main__":
    # 检查是否有 GPU 可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化环境和模型
    env = LuxAIS3GymEnv()
    env = RecordEpisode(env, save_dir="episodes")
    env_params = EnvParams(map_type=0, max_steps_in_match=100)
    RandomizedEnvParams(env_params)
    
    input_shape = (169, 24, 24)  # 更新此形状以适应环境
    num_actions = 3

    model = ActorCriticNet(input_shape, num_actions, device).to(device)
    model.load_model('kits/python/actor_critic_model_1000.pth')

    for episode in range(2):
        next_obs, info = env.reset(seed=1, options=dict(params=env_params))
        P_O0 = ProcessObservation(0,1)
        P_O1 = ProcessObservation(1,0)
        obs_data0, reward_return0, done, terminate = P_O0.process_observation(reshape_obs(next_obs['player_0']))
        obs_data1, reward_return1, done, terminate = P_O1.process_observation(reshape_obs(next_obs['player_1']))

        while not terminate:
            action0, _, state_value = select_action(model, obs_data0)
            action1, _, _ = select_action(model, obs_data1)
            action = {'player_0': action0, 'player_1': action1}
            next_obs, _, _, _, _ = env.step(action)

            obs_data0, reward_return0, done, terminate = P_O0.process_observation(reshape_obs(next_obs['player_0']))
            obs_data1, reward_return1, done, terminate = P_O1.process_observation(reshape_obs(next_obs['player_1']))

            env.render()

    env.close()