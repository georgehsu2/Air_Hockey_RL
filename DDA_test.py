#!/usr/bin/env python3
import os
import glob
import torch
import numpy as np
import time
from DDA_train_env import AirHockeyEnv
from DDA_train import ActorCritic  # 如果你的 ActorCritic 在别的文件，修改这里

# 自动寻找最新的 B 模型文件
def find_latest_b_model():
    candidates = glob.glob("DDA_saved_models_20250719_192533/DDA_agent_ep4470.pth", recursive=True)
    if not candidates:
        raise FileNotFoundError("找不到任何 B 模型（DDA_agent*.pth）")
    latest = max(candidates, key=os.path.getmtime)
    print(f"使用最新的 B 模型：{latest}")
    return latest

def main():
    # --- 参数设定 ---
    A_MODEL_PATH = 'saved_models_20250518_151900/model_episode_10500.pth'
    B_MODEL_PATH = find_latest_b_model()
    EPISODES     = 100
    RENDER       = True   # 改成 True 即可渲染画面
    
    # --- 环境与模型准备 ---
    env = AirHockeyEnv(model_path=A_MODEL_PATH,
                       render_mode='human' if RENDER else 'none')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    agent = ActorCritic(obs_dim, act_dim).to(device)
    ckpt  = torch.load(B_MODEL_PATH, map_location=device)
    # 支持直接存 state_dict 也支持整 dict
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        agent.load_state_dict(ckpt['state_dict'])
    else:
        agent.load_state_dict(ckpt)
    agent.eval()
    agent.action_scale = torch.FloatTensor(env.action_space.high).to(device)
    
    # --- 测试循环 ---
    rewards = []
    for epi in range(1, EPISODES + 1):
        obs = env.reset()
        done = False
        total_r = 0.0
        
        while not done:
            # 1) B 模型确定性策略
            with torch.no_grad():
                t_obs = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                logits, _ = agent(t_obs)
                mu = torch.tanh(logits) * agent.action_scale
            action = mu.squeeze(0).cpu().numpy()
            print(action)
            # 2) 交给环境
            obs, reward, done, _ = env.step(action)
            time.sleep(0.05)
            total_r += reward
            
            # 3) 可视化
            if RENDER:
                env.render()
        
        rewards.append(total_r)
        print(f"[Episode {epi:03d}] total reward = {total_r:.3f}")
    
    # --- 结果统计 ---
    mean_r = np.mean(rewards)
    std_r  = np.std(rewards)
    print(f"\n--- 测试结束 ({EPISODES} 回合) ---")
    print(f"平均 reward = {mean_r:.3f} ± {std_r:.3f}")
    
    env.close()

if __name__ == "__main__":
    main()