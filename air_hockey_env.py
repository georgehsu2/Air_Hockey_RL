import numpy as np
from gym import spaces
import pygame
import gym
from air_hockey_game_mouse import AirHockeyGame

class AirHockeyEnv(gym.Env):
    def __init__(self, render_mode="human"):
        super().__init__()
        do_render = (render_mode == "human")
        self.game = AirHockeyGame(render=do_render)
        self.render_mode = render_mode
        
        # 初始化遊戲實例
        self.game = AirHockeyGame()
        
        # 定義觀測空間 (改進：更規範化的狀態表示)
        self.observation_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(18,),  # 18個特徵維度
            dtype=np.float32
        )
        
        # 定義動作空間 (連續速度控制)
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(2,), 
            dtype=np.float32
        )
        
        # 訓練相關狀態追蹤
        self.prev_ball_distance = None
        self.last_ball_owner = None  # 最後觸球方 (改進：增加球權追蹤)
        self.allow_one_bounce = True
        self.episode_step = 0
        
        # 獎勵參數 (可調參數)
        self.reward_weights = {
            'goal': 10.0,
            'hit': 5.0,
            'distance': 0.1,
            'direction': 0.2,
            'time_penalty': -0.01,
            'one_bounce_goal': 10.0,
            'edge_penalty': -1.0,
        }
    def _normalize(self, value, min_val, max_val):
            """将数值规范到 [-1, 1] 范围（支持 NumPy 数组）"""
            value = np.array(value, dtype=np.float32)
            min_val = np.array(min_val, dtype=np.float32)
            max_val = np.array(max_val, dtype=np.float32)
            return 2. * (value - min_val) / (max_val - min_val + 1e-8) - 1.

    def get_observation(self):
        """获取规范化的观察值（修复数组操作）"""
        # 基础状态（转换为 NumPy 数组）
        ball_pos = np.array(self.game.ball_pos, dtype=np.float32)
        ball_vel = np.array(self.game.ball_vel, dtype=np.float32)
        paddle1_pos = np.array(self.game.player1_pos, dtype=np.float32)
        paddle1_vel = np.array(self.game.player1_vel, dtype=np.float32)
        paddle2_pos = np.array(self.game.player2_pos, dtype=np.float32)
        paddle2_vel = np.array(self.game.player2_vel, dtype=np.float32)
        
        # 相对位置计算（使用 NumPy 操作）
        ball_to_paddle = ball_pos - paddle1_pos
        ball_to_goal = np.array([self.game.WIDTH/2, 0], dtype=np.float32) - ball_pos
        ball_to_opp = ball_pos - paddle2_pos
        
        # 组合观察向量（确保所有部分为 1D 数组）
        obs = np.concatenate([
            self._normalize(ball_pos, [0, 0], [self.game.WIDTH, self.game.HEIGHT]).flatten(),
            (ball_vel / self.game.BALL_MAX_SPEED).flatten(),
            self._normalize(paddle1_pos, [0, 0], [self.game.WIDTH, self.game.HEIGHT]).flatten(),
            (paddle1_vel / self.game.PADDLE_MAX_SPEED).flatten(),
            self._normalize(paddle2_pos, [0, 0], [self.game.WIDTH, self.game.HEIGHT]).flatten(),
            (paddle2_vel / self.game.PADDLE_MAX_SPEED).flatten(),
            (ball_to_paddle / max(self.game.WIDTH, self.game.HEIGHT)).flatten(),
            (ball_to_goal / max(self.game.WIDTH, self.game.HEIGHT)).flatten(),
            (ball_to_opp / max(self.game.WIDTH, self.game.HEIGHT)).flatten()
        ])

        return obs.astype(np.float32)


    def _calculate_rewards(self, player1_scored, player2_scored):
        """综合奖励计算（修复数组操作）"""
        rewards = 0.0

        if player2_scored:
            rewards -= self.reward_weights['goal']
            
        # —— 下半场距离奖励 —— 
        # 只在球在 Player1 半场（y > HEIGHT/2）时生效
        ball = np.array(self.game.ball_pos, dtype=np.float32)
        paddle = np.array(self.game.player1_pos, dtype=np.float32)
        curr_dist = np.linalg.norm(ball - paddle)
        if ball[1] > (self.game.HEIGHT / 2.1):
            # 比较与上一步的距离
            if self.prev_ball_distance is not None:
                if curr_dist < self.prev_ball_distance:
                    rewards += 0.1
                else:
                    rewards -= 0.1
 
        # —— 新增：只要下方拍子碰到视窗边缘就给 negative reward —— 
        px, py = self.game.player1_pos
        R = self.game.PADDLE_RADIUS
        W, H = self.game.WIDTH, self.game.HEIGHT
        # 左、右、上（中线）、下 边缘判定
        if (px <= R) or (px >= W - R) or (py <= H//2 + R) or (py >= H - R):
            rewards += self.reward_weights['edge_penalty']
            print("edge penalty")

        return rewards

    def _check_ball_owner(self):
        """检测当前球权（修复数组操作）"""
        ball_pos = np.array(self.game.ball_pos, dtype=np.float32)
        player1_pos = np.array(self.game.player1_pos, dtype=np.float32)
        player2_pos = np.array(self.game.player2_pos, dtype=np.float32)
        
        dist_p1 = np.linalg.norm(ball_pos - player1_pos)
        dist_p2 = np.linalg.norm(ball_pos - player2_pos)
        
        if dist_p1 < dist_p2 and dist_p1 < (self.game.PADDLE_RADIUS + self.game.BALL_RADIUS) * 1.5:
            return 'player1'
        elif dist_p2 < (self.game.PADDLE_RADIUS + self.game.BALL_RADIUS) * 1.5:
            return 'player2'
        return None

    def step(self, action):
        self.game.last_paddle_hit = None
        self.game.just_one_bounce_hit = False  
        # 保存当前得分
        prev_p1_score = self.game.player1_score
        prev_p2_score = self.game.player2_score

        # 将 PPO 代理的动作应用到玩家1
        self.game.player1_vel[0] = action[0] * self.game.PADDLE_MAX_SPEED
        self.game.player1_vel[1] = action[1] * self.game.PADDLE_MAX_SPEED

        # 更新游戏状态
        self.game.move_paddles()
        self.game.move_ball()
        self.game.check_collisions()
        if self.game.last_paddle_hit is not None:
            self.last_ball_owner = self.game.last_paddle_hit
        # 只要 Player2 真正击过球，就重新允许下次的 one_bounce 奖励
        if self.last_ball_owner == 'player2':
            self.allow_one_bounce = True
        # 检查得分
        p1_scored = self.game.player1_score > prev_p1_score
        p2_scored = self.game.player2_score > prev_p2_score

        # 获取观察值
        obs = self.get_observation()

        # 计算奖励
        reward = self._calculate_rewards(p1_scored, p2_scored)
        # —— 只有在 allow_one_bounce=True 且本帧 just_one_bounce_hit 才给奖励 —— 
        if self.game.just_one_bounce_hit and self.allow_one_bounce:
            reward += self.reward_weights['one_bounce_goal']
            # 给完后立即关掉，必须等 Player2 下一次击球才可再开
            self.allow_one_bounce = False
            self.game.just_one_bounce_hit = False
        
        # 终止条件
        # self.episode_step += 1
        done = p1_scored or p2_scored or (self.episode_step >= 1000)
        if done and not (p1_scored or p2_scored) and self.episode_step >= 1000:
            reward -= 10

        # 信息字典
        info = {
            'score_diff': self.game.player1_score - self.game.player2_score,
            'ball_owner': self._check_ball_owner(),
            'episode_steps': self.episode_step
        }

        return obs, reward, done, info

    def reset(self):
        """重置環境狀態"""
        self.game.reset()
        self.episode_step = 0
        self.last_ball_owner = None
        self.allow_one_bounce = True
        ball = np.array(self.game.ball_pos, dtype=np.float32)
        paddle = np.array(self.game.player1_pos, dtype=np.float32)
        self.prev_ball_distance = np.linalg.norm(ball - paddle)
        return self.get_observation()

    def render(self, mode='human'):
        if self.render_mode == 'human':
            self.game.draw()
            pygame.display.flip()
            return True
        return False

    def close(self):
        """關閉環境"""
        pygame.quit()