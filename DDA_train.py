import os
import sys
import time
from datetime import datetime
from pprint import pprint
import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# import our Gym environment
from DDA_train_env import AirHockeyEnv

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Hyperparameters
LEARNING_RATE       = 3e-4
GAMMA               = 0.99
GAE_LAMBDA          = 0.95
CLIP_EPSILON        = 0.2
VALUE_COEF          = 0.5
ENTROPY_COEF        = 0.01
BATCH_SIZE          = 256
EPOCHS              = 10
TIMESTEPS_PER_UPDATE= 2048
MAX_STEPS_PER_EPISODE=5000
SEED                = 123

# Actor-Critic Network (2-dim action)
class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.common = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        # actor head -> mean logits for 2-dim action
        self.actor = nn.Linear(128, output_dim)
        # one log_std per action dimension
        self.log_std = nn.Parameter(torch.zeros(output_dim))
        # critic head
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        h = self.common(x)
        mu_logits = self.actor(h)     # unconstrained
        value     = self.critic(h)
        return mu_logits, value

def select_action(agent, state, device):
    """
    Sample a 2-dim action from a diagonal Gaussian in the real-action space.
    Returns:
      action_np: (2,) numpy array in [-action_scale, +action_scale]
      log_prob:  tensor log π(a|s)
      value:     tensor V(s)
    """
    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)  # [1, obs_dim]
    logits, value = agent(state_t)                              # [1,2], [1,1]

    # map logits -> mean in [-action_scale, +action_scale]
    mu = torch.tanh(logits) * agent.action_scale                # [1,2]
    # std in the same unit
    std = torch.exp(agent.log_std).unsqueeze(0) * agent.action_scale  # [1,2]

    dist = torch.distributions.Normal(mu, std)
    action = dist.sample()                                      # [1,2]
    action = action.clamp(-agent.action_scale, agent.action_scale)
    log_prob = dist.log_prob(action).sum(dim=-1)                # [1]

    return action.squeeze(0).cpu().numpy(), log_prob.squeeze(), value.squeeze()

def rollout_one_episode(env, agent, device, render=False):
    """
    采集一次完整比赛（从 reset() 到 done=True）的 transitions
    """
    obs = env.reset()
    done = False

    # 存储本场比赛的数据
    batch = {
        'states': [], 'actions': [], 'log_probs': [],
        'values': [], 'rewards': [], 'next_states': [], 'dones': []
    }

    ep_reward = 0.0

    while not done:
        # 1) B 模型选动作
        state_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        logits, value = agent(state_t)
        mu   = torch.tanh(logits) * agent.action_scale
        std  = torch.exp(agent.log_std).unsqueeze(0) * agent.action_scale
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample().clamp(-agent.action_scale, agent.action_scale)
        logp   = dist.log_prob(action).sum(-1)

        action_np = action.squeeze(0).cpu().numpy()

        # 2) 与环境交互
        next_obs, reward, done, _ = env.step(action_np)

        # —— 在这里 render & 处理事件 —— #
        if render:
            env.render()                # 会调用 game.draw() 和 pygame.display.flip()
            pygame.event.pump()         # 让 Pygame 处理窗口事件，否则窗口会“未响应”
            # time.sleep(0.01)            # 控制帧率，可酌情调整

        # 3) 存储 transition
        batch['states'].append(obs)
        batch['actions'].append(action_np)
        batch['log_probs'].append(logp.item())
        batch['values'].append(value.item())
        batch['rewards'].append(reward)
        batch['next_states'].append(next_obs)
        batch['dones'].append(done)

        obs = next_obs
        ep_reward += reward

    # 把 list → numpy
    for k in batch:
        batch[k] = np.array(batch[k])
    return batch, ep_reward

def compute_gae(rewards, values, dones, next_values):
    advantages = np.zeros_like(rewards)
    last_adv = 0
    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + GAMMA * next_values[t] * mask - values[t]
        advantages[t] = last_adv = delta + GAMMA * GAE_LAMBDA * mask * last_adv
    returns = advantages + values
    return advantages, returns

def main():
    RENDER = False
    # 1) 随机种子
    set_seed(SEED)

    # 2) 环境与模型
    A_MODEL_PATH = 'saved_models_20250518_151900/model_episode_10500.pth'
    env = AirHockeyEnv(model_path=A_MODEL_PATH, render_mode='human')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = ActorCritic(obs_dim, act_dim).to(device)
    # action_scale 注册
    agent.action_scale = torch.FloatTensor(env.action_space.high).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)

    # 3) 日志和模型存储
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=f'runs/ppo_{current_time}')
    save_dir = f'DDA_saved_models_{current_time}'
    os.makedirs(save_dir, exist_ok=True)

    total_steps = 0
    episode     = 0
    MAX_EPISODES = 10000  # 或者你想跑的回合数

    # 4) 训练主循环：每个 env.done 为一个 episode
    while episode < MAX_EPISODES:
        batch, ep_reward = rollout_one_episode(env, agent, device, render=RENDER)

        # 转成 torch
        states     = torch.FloatTensor(batch['states']).to(device)
        actions    = torch.FloatTensor(batch['actions']).to(device)
        old_log_ps = torch.FloatTensor(batch['log_probs']).to(device)
        rewards    = batch['rewards']
        dones      = batch['dones']
        next_states= torch.FloatTensor(batch['next_states']).to(device)

        # 计算 values & next_values
        with torch.no_grad():
            _, values      = agent(states)
            _, next_values = agent(next_states)
        values      = values.squeeze().cpu().numpy()
        next_values = next_values.squeeze().cpu().numpy()

        # GAE
        advantages, returns = compute_gae(rewards, values, dones, next_values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 转成 tensor
        advantages = torch.FloatTensor(advantages).to(device)
        returns    = torch.FloatTensor(returns).to(device)

        # PPO 多轮更新
        for _ in range(EPOCHS):
            logits, value_preds = agent(states)
            mu  = torch.tanh(logits) * agent.action_scale
            std = torch.exp(agent.log_std).unsqueeze(0) * agent.action_scale
            dist= torch.distributions.Normal(mu, std)

            new_log_ps = dist.log_prob(actions).sum(-1)
            entropy    = dist.entropy().sum(-1)

            ratios = (new_log_ps - old_log_ps).exp()
            surr1  = ratios * advantages
            surr2  = torch.clamp(ratios, 1-CLIP_EPSILON, 1+CLIP_EPSILON) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss  = VALUE_COEF * (returns - value_preds.squeeze()).pow(2).mean()
            entropy_loss= -ENTROPY_COEF * entropy.mean()

            loss = policy_loss + value_loss + entropy_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 累加统计
        total_steps += len(batch['states'])
        episode     += 1

        # 日志
        writer.add_scalar('Reward/Episode', ep_reward, episode)
        writer.add_scalar('Loss/Policy', policy_loss.item(), episode)
        writer.add_scalar('Loss/Value',  value_loss.item(), episode)
        writer.add_scalar('Loss/Entropy',entropy_loss.item(), episode)
        print(f"Episode {episode}, Steps {total_steps}, EpReward {ep_reward:.2f}")

        # 定期存模型
        if episode % 10 == 0:
            torch.save(agent.state_dict(),
                       os.path.join(save_dir, f'DDA_agent_ep{episode}.pth'))

    env.close()
    writer.close()

if __name__ == '__main__':
    main()
