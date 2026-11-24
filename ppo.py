import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from air_hockey_env import AirHockeyEnv
from pprint import pprint
import pygame
import sys
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time

# Hyperparameters
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.05
BATCH_SIZE = 256
EPOCHS = 10
TIMESTEPS_PER_UPDATE = 2048
MAX_STEPS_PER_EPISODE = 5000  # 最大步數限制
ACTION_STD = 0.8  # 取樣與更新共用的動作標準差

# Define Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.common = nn.Sequential(
            nn.Linear(input_dim, 128),  # 增加節點數
            nn.ReLU(),
            nn.Linear(128, 128),  # 增加層數
            nn.ReLU(),
            nn.Linear(128, 128),  # 增加層數
            nn.ReLU(),
        )
        self.actor = nn.Linear(128, output_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        common_out = self.common(x)
        action_logits = self.actor(common_out)
        value = self.critic(common_out)
        return action_logits, value

# PPO Agent
class PPOAgent:
    def __init__(self, env):
        super(PPOAgent, self).__init__()
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.model = ActorCritic(self.obs_dim, self.action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.gamma = GAMMA
        self.lambda_ = GAE_LAMBDA
        self.epsilon = CLIP_EPSILON
        self.value_coef = VALUE_COEF
        self.entropy_coef = ENTROPY_COEF
        self.epochs = EPOCHS
        
        # Create directories for saving models and logs
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = f'saved_models_{current_time}'
        self.log_dir = f'runs/ppo_{current_time}'
        os.makedirs(self.save_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        # Initialize episode counter
        self.episode_count = 0

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            action_logits, _ = self.model(state)
        action_mean = torch.tanh(action_logits)
        action_std = torch.ones_like(action_mean) * ACTION_STD
        dist = torch.distributions.Normal(action_mean, action_std)

        # log_prob 用未裁切的取樣值計算；送進環境前才裁到 [-1, 1]
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)

        return action.cpu().numpy(), log_prob, entropy

    def save_model(self, episode):
        save_path = os.path.join(self.save_dir, f'model_episode_{episode}.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode': episode
        }, save_path)
        print(f"Model saved at episode {episode}")
        
    def compute_advantages(self, rewards, values, next_values, dones):
        """GAE。done 的那一步不 bootstrap 下一狀態，也不把 advantage 傳回前一個 episode。"""
        advantages = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0
        for i in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[i])
            delta = rewards[i] + self.gamma * next_values[i] * mask - values[i]
            gae = delta + self.gamma * self.lambda_ * mask * gae
            advantages[i] = gae
        return advantages

    def update(self, trajectories):
        states = np.array([t[0] for t in trajectories])
        actions = np.array([t[1] for t in trajectories])
        rewards = np.array([t[2] for t in trajectories], dtype=np.float32)
        next_states = np.array([t[3] for t in trajectories])
        dones = np.array([t[4] for t in trajectories], dtype=np.float32)
        old_log_probs = np.array([t[5] for t in trajectories])

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            _, current_values = self.model(states)
            _, next_values = self.model(next_states)
        current_values = current_values.squeeze(-1)
        next_values = next_values.squeeze(-1)

        advantages = self.compute_advantages(
            rewards, current_values.cpu().numpy(), next_values.cpu().numpy(), dones)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = advantages + current_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):
            new_action_logits, new_values = self.model(states)
            new_action_mean = torch.tanh(new_action_logits)
            # 必須和 select_action 用同一個標準差，ratio 才是新舊策略的機率比
            dist = torch.distributions.Normal(new_action_mean, torch.ones_like(new_action_mean) * ACTION_STD)
            new_log_probs = dist.log_prob(actions).sum(axis=-1)

            ratios = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = self.value_coef * (returns - new_values.squeeze(-1)).pow(2).mean()
            # ACTION_STD 固定時 entropy 是常數，這一項不影響梯度，只用來記錄
            entropy_loss = -self.entropy_coef * dist.entropy().mean()

            loss = policy_loss + value_loss + entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 記錄損失值
            self.writer.add_scalar("Loss/Policy", policy_loss.item(), self.episode_count)
            self.writer.add_scalar("Loss/Value", value_loss.item(), self.episode_count)
            self.writer.add_scalar("Loss/Entropy", entropy_loss.item(), self.episode_count)


    def train(self, total_timesteps):
        state = self.env.reset()
        trajectories = []
        episode_rewards = []
        total_steps = 0
        episode_reward = 0
        episode_steps = 0

        while total_steps < total_timesteps:
            # 初始化每個更新週期的數據收集列表
            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_next_states = []
            batch_dones = []
            batch_log_probs = []

            # 收集一個批次的數據
            for _ in range(TIMESTEPS_PER_UPDATE):
                self.env.render()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        self.writer.close()
                        return

                # 獲取動作
                action, log_prob, _ = self.select_action(state)
                # print(action)
                next_state, reward, done, _ = self.env.step(np.clip(action, -1.0, 1.0))
                # time.sleep(1/20)
                # 檢查是否超過最大步數
                if episode_steps >= MAX_STEPS_PER_EPISODE:
                    done = True

                # 儲存轉換
                batch_states.append(state)
                batch_actions.append(action)
                batch_rewards.append(reward)
                batch_next_states.append(next_state)
                batch_dones.append(done)
                batch_log_probs.append(log_prob.detach().cpu().numpy())

                episode_reward += reward
                total_steps += 1
                episode_steps += 1

                state = next_state

                if done:
                    state = self.env.reset()
                    episode_rewards.append(episode_reward)

                    # 記錄獎勵
                    self.writer.add_scalar('Reward/episode', episode_reward, self.episode_count)
                    print(f"Episode {self.episode_count} finished. Steps: {episode_steps}, Reward: {episode_reward}")

                    # 每10個episode儲存模型
                    if self.episode_count % 10 == 0:
                        self.save_model(self.episode_count)

                    self.episode_count += 1
                    episode_reward = 0
                    episode_steps = 0

            # 確保所有數據都是正確的形狀
            if len(batch_states) > 0:  # 確保有收集到數據
                batch_states = np.array(batch_states)
                batch_actions = np.array(batch_actions)
                batch_rewards = np.array(batch_rewards)
                batch_next_states = np.array(batch_next_states)
                batch_dones = np.array(batch_dones)
                batch_log_probs = np.array(batch_log_probs)

                trajectories = list(zip(
                    batch_states,
                    batch_actions,
                    batch_rewards,
                    batch_next_states,
                    batch_dones,
                    batch_log_probs
                ))

                self.update(trajectories)

            if len(episode_rewards) > 0 and total_steps % 100 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Step: {total_steps}, Avg Reward: {avg_reward}")
                sys.stdout.flush()
        self.env.close()
        self.writer.close()
# Train the PPO agent
if __name__ == "__main__":
    env = AirHockeyEnv(render_mode=None)
    agent = PPOAgent(env)
    agent.train(total_timesteps=7500000)