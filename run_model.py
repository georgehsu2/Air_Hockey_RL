import pygame
import numpy as np
import torch
import torch.nn as nn
from no_rulebased_game import AirHockeyGame
import math

class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.common = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.actor = nn.Linear(128, output_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        common_out = self.common(x)
        action_logits = self.actor(common_out)
        value = self.critic(common_out)
        return action_logits, value

def load_model(model_path, input_dim, output_dim):
    model = ActorCritic(input_dim, output_dim)
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def play_against_ai(model_path, is_upper_paddle=False):
    """
    在 pygame 窗口中运行一局 AirHockeyGame，让玩家1用鼠标控制，
    玩家2 用已训练好的 ActorCritic 模型控制。通过 is_upper_paddle 标志
    来决定是否对模型输出的纵向动作做翻转（用于上方拍子）。
    """
    # 初始化游戏
    game = AirHockeyGame()
    game.reset()

    # 获取网络输入输出维度
    input_dim = game.observation_space.shape[0]
    output_dim = game.action_space.shape[0]

    # 加载模型
    model = load_model(model_path, input_dim, output_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    clock = pygame.time.Clock()
    running = True

    while running:
        # 处理窗口事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- 玩家1：鼠标控制 ---
        mouse_x, mouse_y = pygame.mouse.get_pos()
        game.player1_vel[0] = mouse_x - game.player1_pos[0]
        game.player1_vel[1] = mouse_y - game.player1_pos[1]
        # 限速
        speed = math.hypot(*game.player1_vel)
        if speed > game.PADDLE_MAX_SPEED:
            scale = game.PADDLE_MAX_SPEED / speed
            game.player1_vel[0] *= scale
            game.player1_vel[1] *= scale
        # 更新位置并约束区域
        game.player1_pos[0] = np.clip(game.player1_pos[0] + game.player1_vel[0],
                                       game.PADDLE_RADIUS, game.WIDTH - game.PADDLE_RADIUS)
        game.player1_pos[1] = np.clip(game.player1_pos[1] + game.player1_vel[1],
                                       game.HEIGHT//2 + game.PADDLE_RADIUS, game.HEIGHT - game.PADDLE_RADIUS)

        # --- 玩家2 (AI) ---
        state = game.get_observation()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_logits, _ = model(state_tensor)
            action = torch.tanh(action_logits).squeeze().cpu().numpy()

        # 将模型输出映射到速度
        vx = action[0] * game.PADDLE_MAX_SPEED
        vy = action[1] * game.PADDLE_MAX_SPEED
        # 如果是在上方拍子，则对纵向分量取反
        if is_upper_paddle:
            vy = -vy

        game.player2_vel = [vx, vy]
        # 更新位置并约束区域
        game.player2_pos[0] = np.clip(game.player2_pos[0] + vx,
                                       game.PADDLE_RADIUS, game.WIDTH - game.PADDLE_RADIUS)
        game.player2_pos[1] = np.clip(game.player2_pos[1] + vy,
                                       game.PADDLE_RADIUS, game.HEIGHT//2 - game.PADDLE_RADIUS)

        # 更新物理和碰撞
        game.move_ball()
        game.check_collisions()

        # 渲染
        game.draw()
        pygame.display.flip()
        clock.tick(60)

        # 结束判定
        if game.player1_score >= 5 or game.player2_score >= 5:
            print("Game Over!")
            print(f"Player 1 Score: {game.player1_score}, Player 2 Score: {game.player2_score}")
            running = False

    pygame.quit()

if __name__ == "__main__":
    # stable
    # saved_models_20250317_193104/model_episode_6000.pth 
    # saved_models_20250518_053028/model_episode_9680.pth 守在門前
    # saved_models_20250518_151900/model_episode_10500.pth
    model_path = "saved_models_20250518_151900/model_episode_10500.pth"
    play_against_ai(model_path, is_upper_paddle=True)