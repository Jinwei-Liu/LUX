from torchrl.data import ListStorage, ReplayBuffer
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from luxai_s3.params import EnvParams, env_params_ranges
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode
from torch.utils.tensorboard import SummaryWriter
import os, shutil
from process_observation import ProcessObservation, reshape_obs
import torch.nn.functional as F  
from gym.vector import AsyncVectorEnv
import torch.multiprocessing as mp
import time
from queue import Full
from ppo import *
from datetime import datetime
def compute_gae_for_episode(episode_buffer, gamma=0.95, gae_lambda=0.95):
    """
    输入:
        episode_buffer: 本回合的所有transition列表，
            每个元素形如 (state, action, reward, done, log_prob, value, next_value)
        gamma: 折扣因子
        gae_lambda: GAE衰减系数
    输出:
        带有 (state, action, reward, done, log_prob, value, advantage, return) 的列表
        您可以根据自己需要再加别的字段
    """
    advantages = []
    gae = 0.0

    # 反向遍历整条序列
    for t in reversed(range(len(episode_buffer))):
        (_, _, reward, done, _, value, next_value, _) = episode_buffer[t]

        # 如果是最后一个 transition，就可以用 next_value，否则可以再看一看环境是否截断
        # 这里 done=1 意味着下一时刻的价值为0（回合结束）
        # 也可以直接用 next_value，如果下一状态也在同一个回合
        mask = 1.0 - done  # done=1时表示episode结束，mask=0

        # 计算 \delta
        delta = reward + gamma * next_value * mask - value
        # 计算当前时刻的 GAE
        gae = delta + gamma * gae_lambda * mask * gae
        advantages.append(gae)
    
    advantages.reverse()  # 上面是逆序计算，这里翻转回来

    # 归一化优势
    advantages = torch.tensor(np.array(advantages), dtype=torch.float32)
    advantage_mean = advantages.mean()
    advantage_std = advantages.std()
    advantages = (advantages - advantage_mean) / (advantage_std + 1e-8)  # 防止除以0
    advantages = advantages.detach().cpu().numpy()
    # 把 advantage 和 return 都组装进去
    out = []
    for i, (state, action, reward, done, log_prob, value, _, next_state) in enumerate(episode_buffer):
        advantage = advantages[i]
        out.append((
            torch.tensor(state, dtype=torch.float32),     
            torch.tensor(action, dtype=torch.long),       
            torch.tensor(reward, dtype=torch.float32),    
            torch.tensor(done, dtype=torch.float32),      
            torch.tensor(next_state, dtype=torch.float32),
            torch.tensor(log_prob, dtype=torch.float32),  
            torch.tensor(advantage, dtype=torch.float32)  
        ))
    return out

def worker_process(rank, local_model, env_params, replay_queue, signal_queue, signal_load_queue, reward_queue, temperature):

    """子进程：采样数据，并等待主进程更新模型"""
    torch.manual_seed(rank)
    env = LuxAIS3GymEnv()
    # env = RecordEpisode(env, save_dir=f"episodes_worker_{rank}")
    inner_model = ActorCriticNet(input_shape=(169, 24, 24), num_actions=3, device="cuda").to("cuda")

    while True:
        # 重置环境
        reward0_total = 0
        P_O0 = ProcessObservation(0, 1)
        P_O1 = ProcessObservation(1, 0)
        inner_model.load_state_dict(local_model.state_dict())
        RandomizedEnvParams(env_params) # 随机化环境参数

        obs, info = env.reset(seed=random.randint(0, 1e5), options=dict(params=env_params)) #random.randint(0, 1e5)
        obs_data0, reward_return0, done, terminate = P_O0.process_observation(reshape_obs(obs['player_0']))
        obs_data1, reward_return1, done, terminate = P_O1.process_observation(reshape_obs(obs['player_1']))
        reward0_total += reward_return0
         # 用于暂存当前回合的所有transition
        episode_buffer_player0 = []
        episode_buffer_player1 = []

        # 采样一轮数据
        while not terminate:
            with torch.no_grad():  # 推理阶段不需要梯度
                action0, log_prob0, state_value0 = select_action(inner_model, obs_data0, temperature)
                action1, log_prob1, state_value1 = select_action(inner_model, obs_data1, temperature)
            
            action = {'player_0': action0, 'player_1': action1}
            next_obs, _, _, _, _ = env.step(action)

            next_obs_data0, reward_return0, done, terminate = P_O0.process_observation(reshape_obs(next_obs['player_0']))
            next_obs_data1, reward_return1, done, terminate = P_O1.process_observation(reshape_obs(next_obs['player_1']))

            reward0_total += reward_return0

            _, _, next_state_value0 = select_action(inner_model, next_obs_data0, temperature)
            _, _, next_state_value1 = select_action(inner_model, next_obs_data1, temperature)

            episode_buffer_player0.append((
                obs_data0, 
                action0, 
                reward_return0, 
                done, 
                log_prob0, 
                state_value0,
                next_state_value0,
                next_obs_data0
            ))
            episode_buffer_player1.append((
                obs_data1, 
                action1, 
                reward_return1, 
                done, 
                log_prob1, 
                state_value1,
                next_state_value1,
                next_obs_data1
            ))

            # 分别计算两个队伍各自的 advantage / return
            if done:
                gae_samples_player0 = compute_gae_for_episode(episode_buffer_player0)
                gae_samples_player1 = compute_gae_for_episode(episode_buffer_player1)
                episode_buffer_player0 = []
                episode_buffer_player1 = []
                
                replay_queue.put(gae_samples_player0)
                replay_queue.put(gae_samples_player1)
                
            obs_data0 = next_obs_data0
            obs_data1 = next_obs_data1
            if next_obs['player_0'].steps == 504:
                print('finally_point',next_obs['player_0'].team_points,'see_relic_reward_num',obs_data0[:,:,-1].sum(),'reward',reward0_total)

        # 环境完成后，发送信号给主进程并等待模型更新
        signal_queue.put(rank)  # 通知主进程当前子进程完成采样
        reward_queue.put(reward0_total.mean())
        signal_load_queue.get()  # 等待主进程更新模型参数

import copy

def main():
    # 初始化环境参数和模型
    env_params = EnvParams(map_type=0, max_steps_in_match=100)
    num_workers = 7  # 子进程数量
    replay_queue = mp.Queue()  # 数据队列
    signal_queue = mp.Queue(maxsize=num_workers)  # 数据队列
    signal_load_queue = mp.Queue(maxsize=num_workers)  # 数据队列
    reward_queue = mp.Queue(maxsize=num_workers)  # 数据队列

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化模型
    model = ActorCriticNet(input_shape=(169, 24, 24), num_actions=3, device=device).to(device)
    # model.load_model('LUX/kits/python/actor_critic_model_1000.pth')
    local_model = ActorCriticNet(input_shape=(169, 24, 24), num_actions=3, device="cpu")
    local_model.load_state_dict(model.state_dict())
    local_model.share_memory()  # 共享模型参数
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    timenow = str(datetime.now())[0:-10]
    timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
    writepath = 'runs/' + timenow
    if os.path.exists(writepath): shutil.rmtree(writepath)
    writer = SummaryWriter(log_dir=writepath)

    # 启动子进程
    temperature_values = np.linspace(1, 100, num_workers)  # 生成均匀分布的 temperature
    processes = []
    for rank in range(num_workers):
        temp = 1 #temperature_values[rank]
        p = mp.Process(target=worker_process, args=(rank, local_model, env_params, replay_queue, signal_queue, signal_load_queue, reward_queue, temp))
        p.start()
        processes.append(p)

    # 初始化 ReplayBuffer
    replay_buffer = ReplayBuffer(storage=ListStorage(1010*num_workers)) #data_count = len(replay_buffer)

    print("Training Actor-Critic Network")
    for episode in range(10000):
        # 等待所有子进程完成采样
        for _ in range(num_workers):
            signal_queue.get() # 接收子进程完成信号
        
        #接收子进程奖励
        reward_total = 0
        for _ in range(num_workers):
            reward_total += reward_queue.get() # 接收子进程完成信号
        reward_total = reward_total/num_workers

        # 从 replay_queue 中获取采样数据并存入 replay_buffer
        while replay_queue.qsize():  # 避免队列中有未处理的数据
            data = replay_queue.get()
            replay_buffer.extend(copy.deepcopy(data))  # 深拷贝数据，确保数据被完全存储
            del data  # 释放 data 的引用
        # 从 ReplayBuffer 中采样并训练
        loss_list = []
        for _ in range(10*num_workers):  # 每次训练 100 次
            loss = train_actor_critic(model, optimizer, replay_buffer, batch_size=505)
            loss_list.append(loss)

        # 输出训练信息
        loss_mean = np.mean(loss_list)
        print(f'Episode {episode}, Loss/train: {loss_mean:.4f}')
        writer.add_scalar('Loss/train', loss_mean, episode)
        writer.add_scalar('Reward', reward_total, episode)

        # 广播模型参数到所有子进程
        local_model.load_state_dict(model.state_dict())

        for _ in range(num_workers):
            signal_load_queue.put('True')  # 将更新的模型参数发送给子进程

        # 保存模型
        if episode % 100 == 0:
            save_dir = "save_model"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            model.save_model(os.path.join(save_dir, f"actor_critic_model_{episode}.pth"))
            print(f"Episode {episode} completed")

    # 停止子进程
    for p in processes:
        p.terminate()
    for p in processes:
        p.join()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
