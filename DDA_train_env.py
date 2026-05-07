import pygame
import numpy as np
import torch
from gym import Env, spaces

# Import the existing game implementation and constants
# Make sure you've saved your original game code as air_hockey_game.py
from DDA_game import AirHockeyGame, WIDTH, HEIGHT, PADDLE_RADIUS, BALL_RADIUS, PADDLE_MAX_SPEED, BALL_MAX_SPEED


class AirHockeyEnv(Env):
    def __init__(self, model_path: str, render_mode: str = 'human'):
        super().__init__()

        # render flag
        self.render_mode = render_mode
        self.game = AirHockeyGame(model_path)
        self.game.render = (render_mode != 'none')

        # episode settings
        self.max_steps = 5000
        self.current_step = 0

        # action & observation spaces
        self.action_space = spaces.Box(
            low = np.array([-PADDLE_MAX_SPEED, -PADDLE_MAX_SPEED], dtype=np.float32),
            high= np.array([ PADDLE_MAX_SPEED,  PADDLE_MAX_SPEED], dtype=np.float32),
            dtype=np.float32
        )
        low_extra  = -PADDLE_MAX_SPEED * np.ones(2, dtype=np.float32)
        high_extra =  PADDLE_MAX_SPEED * np.ones(2, dtype=np.float32)
        obs_low = np.concatenate([
            [0, 0, -PADDLE_MAX_SPEED, -PADDLE_MAX_SPEED,
             0, 0, -PADDLE_MAX_SPEED, -PADDLE_MAX_SPEED,
             0, 0, -BALL_MAX_SPEED, -BALL_MAX_SPEED,
             1, 1, 1, np.inf],
            low_extra
        ], axis=0)
        obs_high = np.concatenate([
            [WIDTH, HEIGHT,  PADDLE_MAX_SPEED,  PADDLE_MAX_SPEED,
             WIDTH, HEIGHT,  PADDLE_MAX_SPEED,  PADDLE_MAX_SPEED,
             WIDTH, HEIGHT,  BALL_MAX_SPEED,     BALL_MAX_SPEED,
             1, 1, 1, np.inf],
            high_extra
        ], axis=0)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # score & distance trackers
        self.prev_score1 = 0
        self.prev_score2 = 0
        self.prev_dist   = 0.0
        self._predicted_top_triggered = False

        # new: one-bounce reward toggle & edge penalty weight
        self.allow_one_bounce = True
        self.reward_weights = {
            'one_bounce_goal': 10.0,
            'edge_penalty':    -0.001,
        }
    def reset(self):
        self.game.reset_ball()
        self.prev_score1 = self.game.score1
        self.prev_score2 = self.game.score2
        self.current_step = 0

        p2 = np.array(self.game.player2_pos, dtype=np.float32)
        b  = np.array(self.game.ball_pos,    dtype=np.float32)
        self.prev_dist = np.linalg.norm(p2 - b)

        self._predicted_top_triggered = False
        # reset one-bounce toggle
        self.allow_one_bounce = True

        obs_16 = self._get_obs()
        zero_extra = np.zeros(2, dtype=np.float32)
        return np.concatenate([obs_16, zero_extra], axis=0)

    def step(self, action: np.ndarray):
        self.current_step += 1

        # 1. 記錄分差（變動前）
        diff_before = abs(self.game.score2 - self.game.score1)

        # 2. 底部球拍（rule-based）
        self.game.control_bottom()

        # 3. 基礎模型（top）輸出 A
        norm_obs = self.game.get_observation()
        t_obs = torch.FloatTensor(norm_obs).unsqueeze(0).to(self.game.device)
        with torch.no_grad():
            logits, _ = self.game.model(t_obs)
            base_act = torch.tanh(logits).squeeze().cpu().numpy() * PADDLE_MAX_SPEED

        # 4. 合併動作（A + B）
        combined = base_act + action
        vx, vy = combined

        # 5. 套用到 player2
        self.game.player2_vel = [float(vx), float(-vy)]
        self.game.player2_pos[0] = np.clip(
            self.game.player2_pos[0] + vx,
            PADDLE_RADIUS, WIDTH - PADDLE_RADIUS
        )
        self.game.player2_pos[1] = np.clip(
            self.game.player2_pos[1] - vy,
            PADDLE_RADIUS, HEIGHT//2 - PADDLE_RADIUS
        )

        # 6. 物理更新
        self.game.move_ball()
        self.game.check_collisions()

        # 7. 記錄分差（變動後）
        diff_after = abs(self.game.score2 - self.game.score1)

        # 8. 獎勵：只有分差變化時才給 ±1（縮小 +1，擴大 -1，其餘 0）
        reward = int(diff_before - diff_after)

        # 9. 判斷是否結束
        done = (
            self.game.score1 >= 15
            or self.game.score2 >= 15
            or self.current_step >= self.max_steps
        )

        # 10. 更新追蹤用的分數（可選，看你有沒有在別處用到）
        self.prev_score1 = self.game.score1
        self.prev_score2 = self.game.score2

        # 11. 回傳 observation、reward、done
        obs_16 = self._get_obs()
        # 如果你還想保留 base_act 在 observation 裡，就用下面這行：
        obs = np.concatenate([obs_16, base_act], axis=0)
        # 否則直接回傳 obs_16 即可
        return obs, reward, done, {}



    def _get_obs(self) -> np.ndarray:
        # Gather raw positions and velocities
        p1x, p1y = self.game.player1_pos
        v1x, v1y = self.game.player1_vel
        p2x, p2y = self.game.player2_pos
        v2x, v2y = self.game.player2_vel
        bx, by   = self.game.ball_pos
        bvx, bvy = self.game.ball_vel

        # One-hot score state: [top_lead, bottom_lead, tie]
        if self.game.score2 > self.game.score1:
            score_state = [1, 0, 0]
        elif self.game.score1 > self.game.score2:
            score_state = [0, 1, 0]
        else:
            score_state = [0, 0, 1]

        # Score difference (top - bottom)
        score_diff = float(self.game.score2 - self.game.score1)

        return np.array([
            p1x, p1y, v1x, v1y,
            p2x, p2y, v2x, v2y,
            bx,  by,  bvx, bvy,
            *score_state,
            score_diff
        ], dtype=np.float32)

    def render(self, mode='human'):
        self.game.draw()
        pygame.display.flip()

    def close(self):
        pygame.quit()
