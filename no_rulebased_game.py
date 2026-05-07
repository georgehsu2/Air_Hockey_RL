import pygame
import sys
import math
import random
import numpy as np
from gym import spaces

# Define constants for the game
WIDTH, HEIGHT = 450, 850
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
BALL_RADIUS = 20
PADDLE_RADIUS = 40
CORNER_OBSTACLE_RADIUS = 30
FRICTION = 1
PADDLE_MAX_SPEED = 50
BALL_MAX_SPEED = 20
GOAL_WIDTH = 150

# Define corner obstacle positions
CORNER_OBSTACLES = [
    [CORNER_OBSTACLE_RADIUS, CORNER_OBSTACLE_RADIUS],
    [WIDTH - CORNER_OBSTACLE_RADIUS, CORNER_OBSTACLE_RADIUS],
    [CORNER_OBSTACLE_RADIUS, HEIGHT - CORNER_OBSTACLE_RADIUS],
    [WIDTH - CORNER_OBSTACLE_RADIUS, HEIGHT - CORNER_OBSTACLE_RADIUS]
]

class AirHockeyGame:
    
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Air Hockey")
        self.clock = pygame.time.Clock()
        self.reset()  # 初始化遊戲狀態
        self.BALL_RADIUS = BALL_RADIUS
        self.PADDLE_RADIUS = PADDLE_RADIUS
        self.PADDLE_MAX_SPEED = PADDLE_MAX_SPEED
        self.BALL_MAX_SPEED = BALL_MAX_SPEED
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.GOAL_WIDTH = GOAL_WIDTH
        self.FRICTION = FRICTION
        self.observation_space = spaces.Box(
            low=np.array([
                0, 0,                # ball_pos_flipped (2維)
                -20, -20,            # ball_vel_flipped (2維)
                0, 0,                # paddle1_pos (2維)
                0, 0,                # paddle1_vel (2維)
                -10, -10,            # paddle2_pos (2維)
                -10, -10,            # paddle2_vel (2維)
                -450, -850,          # ball_to_paddle (2維) - 最大可能距離為場地大小
                -450, -850,          # ball_to_goal (2維) - 最大可能距離為場地大小
                -450, -850           # ball_to_opp (2維) - 最大可能距離為場地大小
            ]),
            high=np.array([
                450, 800,            # ball_pos_flipped (2維)
                20, 20,              # ball_vel_flipped (2維)
                450, 800,            # paddle1_pos (2維)
                450, 800,            # paddle1_vel (2維)
                10, 10,              # paddle2_pos (2維)
                10, 10,              # paddle2_vel (2維)
                450, 850,            # ball_to_paddle (2維)
                450, 850,            # ball_to_goal (2維)
                450, 850             # ball_to_opp (2維)
            ]),
            dtype=np.float32
        )
                # 定義動作空間
        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            dtype=np.float32
        )
    def reset(self):
        # Reset game state
        self.ball_pos = [WIDTH // 2, HEIGHT // 2]
        self.ball_vel = [0, 0]
        self.player1_pos = [WIDTH // 2, HEIGHT - 50]
        self.player2_pos = [WIDTH // 2, 50]
        self.player1_vel = [0, 0]
        self.player2_vel = [0, 0]
        self.player1_score = 0
        self.player2_score = 0

    def reset_ball(self):
        # Reset ball velocity
        self.ball_vel = [0, 0]

        # 固定由下方發球
        self.ball_pos = [WIDTH // 2, 3 * HEIGHT // 4]  # 球的位置在下方半場

        # Reset paddle positions
        self.player1_pos = [WIDTH // 2, HEIGHT - 50]
        self.player2_pos = [WIDTH // 2, 50]

    def _normalize(self, value, min_val, max_val):
            """将数值规范到 [-1, 1] 范围（支持 NumPy 数组）"""
            value = np.array(value, dtype=np.float32)
            min_val = np.array(min_val, dtype=np.float32)
            max_val = np.array(max_val, dtype=np.float32)
            return 2. * (value - min_val) / (max_val - min_val + 1e-8) - 1.

    # 其他方法保持不變
    def get_observation(self):
        """获取规范化的观察值（从player2的视角，但格式与player1相同）"""
        # 基础状态（转换为 NumPy 数组）
        ball_pos = np.array(self.ball_pos, dtype=np.float32)
        ball_vel = np.array(self.ball_vel, dtype=np.float32)
        paddle1_pos = np.array(self.player1_pos, dtype=np.float32)
        paddle1_vel = np.array(self.player1_vel, dtype=np.float32)
        paddle2_pos = np.array(self.player2_pos, dtype=np.float32)
        paddle2_vel = np.array(self.player2_vel, dtype=np.float32)
        
        # 翻转坐标系（假设游戏在垂直方向上对称）
        flipped_ball_pos = np.array([ball_pos[0], self.HEIGHT - ball_pos[1]], dtype=np.float32)
        flipped_ball_vel = np.array([ball_vel[0], -ball_vel[1]], dtype=np.float32)
        flipped_paddle1_pos = np.array([paddle2_pos[0], self.HEIGHT - paddle2_pos[1]], dtype=np.float32)
        flipped_paddle1_vel = np.array([paddle2_vel[0], -paddle2_vel[1]], dtype=np.float32)
        flipped_paddle2_pos = np.array([paddle1_pos[0], self.HEIGHT - paddle1_pos[1]], dtype=np.float32)
        flipped_paddle2_vel = np.array([paddle1_vel[0], -paddle1_vel[1]], dtype=np.float32)
        
        # 相对位置计算（使用 NumPy 操作）
        ball_to_paddle = flipped_ball_pos - flipped_paddle1_pos
        ball_to_goal = np.array([self.WIDTH/2, 0], dtype=np.float32) - flipped_ball_pos
        ball_to_opp = flipped_ball_pos - flipped_paddle2_pos
        
        # 组合观察向量（确保所有部分为 1D 数组）
        obs = np.concatenate([
            self._normalize(flipped_ball_pos, [0, 0], [self.WIDTH, self.HEIGHT]).flatten(),
            (flipped_ball_vel / self.BALL_MAX_SPEED).flatten(),
            self._normalize(flipped_paddle1_pos, [0, 0], [self.WIDTH, self.HEIGHT]).flatten(),
            (flipped_paddle1_vel / self.PADDLE_MAX_SPEED).flatten(),
            self._normalize(flipped_paddle2_pos, [0, 0], [self.WIDTH, self.HEIGHT]).flatten(),
            (flipped_paddle2_vel / self.PADDLE_MAX_SPEED).flatten(),
            (ball_to_paddle / max(self.WIDTH, self.HEIGHT)).flatten(),
            (ball_to_goal / max(self.WIDTH, self.HEIGHT)).flatten(),
            (ball_to_opp / max(self.WIDTH, self.HEIGHT)).flatten()
        ])
        
        return obs
    def draw(self):
        self.screen.fill(WHITE)
        pygame.draw.line(self.screen, BLACK, (0, HEIGHT // 2), (WIDTH, HEIGHT // 2), 2)

        # Draw goals
        pygame.draw.rect(self.screen, RED, ((WIDTH - GOAL_WIDTH) // 2, HEIGHT - 10, GOAL_WIDTH, 10))
        pygame.draw.rect(self.screen, BLUE, ((WIDTH - GOAL_WIDTH) // 2, 0, GOAL_WIDTH, 10))

        # Draw corner obstacles
        for obstacle in CORNER_OBSTACLES:
            pygame.draw.circle(self.screen, GREEN, (int(obstacle[0]), int(obstacle[1])), CORNER_OBSTACLE_RADIUS)

        # Draw ball and paddles
        pygame.draw.circle(self.screen, BLACK, (int(self.ball_pos[0]), int(self.ball_pos[1])), BALL_RADIUS)
        pygame.draw.circle(self.screen, RED, (int(self.player1_pos[0]), int(self.player1_pos[1])), PADDLE_RADIUS)
        pygame.draw.circle(self.screen, BLUE, (int(self.player2_pos[0]), int(self.player2_pos[1])), PADDLE_RADIUS)

        # Display scores
        font = pygame.font.Font(None, 74)
        score_text = font.render(f"{self.player2_score} - {self.player1_score}", True, BLACK)
        self.screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, HEIGHT // 2 - 40))

    def move_ball(self):
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]

        # Apply friction
        self.ball_vel[0] *= FRICTION
        self.ball_vel[1] *= FRICTION

        # Handle wall collisions
        if self.ball_pos[0] - BALL_RADIUS <= 0 or self.ball_pos[0] + BALL_RADIUS >= WIDTH:
            self.ball_vel[0] = -self.ball_vel[0]
            self.ball_pos[0] = max(BALL_RADIUS, min(WIDTH - BALL_RADIUS, self.ball_pos[0]))

        if self.ball_pos[1] - BALL_RADIUS <= 0:  # Ball reaches top
            if (WIDTH - GOAL_WIDTH) // 2 <= self.ball_pos[0] <= (WIDTH + GOAL_WIDTH) // 2:
                self.player1_score += 1
                self.reset_ball()  # Player 1 scores
            else:
                self.ball_vel[1] = -self.ball_vel[1]
                self.ball_pos[1] = BALL_RADIUS

        if self.ball_pos[1] + BALL_RADIUS >= HEIGHT:  # Ball reaches bottom
            if (WIDTH - GOAL_WIDTH) // 2 <= self.ball_pos[0] <= (WIDTH + GOAL_WIDTH) // 2:
                self.player2_score += 1
                self.reset_ball()  # Player 2 scores
            else:
                self.ball_vel[1] = -self.ball_vel[1]
                self.ball_pos[1] = HEIGHT - BALL_RADIUS

        # Limit ball speed
        speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
        if speed > BALL_MAX_SPEED:
            scale = BALL_MAX_SPEED / speed
            self.ball_vel[0] *= scale
            self.ball_vel[1] *= scale

    def move_paddles(self):
            # 如果要訓練需要關閉下面三行
            mouse_x, mouse_y = pygame.mouse.get_pos()
            self.player1_vel[0] = mouse_x - self.player1_pos[0]
            self.player1_vel[1] = mouse_y - self.player1_pos[1]

            # 限制玩家1的速度
            speed = math.hypot(self.player1_vel[0], self.player1_vel[1])
            if speed > self.PADDLE_MAX_SPEED:
                scale = self.PADDLE_MAX_SPEED / speed
                self.player1_vel[0] *= scale
                self.player1_vel[1] *= scale

            # 更新玩家1的位置
            self.player1_pos[0] += self.player1_vel[0]
            self.player1_pos[1] += self.player1_vel[1]

            # 保持玩家1在合法范围内
            self.player1_pos[0] = max(self.PADDLE_RADIUS, min(self.WIDTH - self.PADDLE_RADIUS, self.player1_pos[0]))
            self.player1_pos[1] = max(self.HEIGHT // 2 + self.PADDLE_RADIUS, min(self.HEIGHT - self.PADDLE_RADIUS, self.player1_pos[1]))

            # # 玩家2的AI逻辑保持不变
            # if random.random() < 0.1:  # 10% 機率採取隨機動作
            #     self.player2_vel[0] = random.uniform(-self.PADDLE_MAX_SPEED, self.PADDLE_MAX_SPEED)
            #     self.player2_vel[1] = random.uniform(-self.PADDLE_MAX_SPEED, self.PADDLE_MAX_SPEED)
            # else:  # 正常跟隨球
            #     if self.ball_pos[1] < self.HEIGHT // 2:
            #         if self.ball_pos[0] > self.player2_pos[0]:
            #             self.player2_vel[0] = min(self.PADDLE_MAX_SPEED, self.ball_pos[0] - self.player2_pos[0])
            #         elif self.ball_pos[0] < self.player2_pos[0]:
            #             self.player2_vel[0] = max(-self.PADDLE_MAX_SPEED, self.ball_pos[0] - self.player2_pos[0])

            #         if self.ball_pos[1] > self.player2_pos[1]:
            #             self.player2_vel[1] = min(self.PADDLE_MAX_SPEED, self.ball_pos[1] - self.player2_pos[1])
            #         elif self.ball_pos[1] < self.player2_pos[1]:
            #             self.player2_vel[1] = max(-self.PADDLE_MAX_SPEED, self.ball_pos[1] - self.player2_pos[1])

            # 更新玩家2的位置
            self.player2_pos[0] += self.player2_vel[0]
            self.player2_pos[1] += self.player2_vel[1]

            # 保持玩家2在合法範圍內
            self.player2_pos[0] = max(self.PADDLE_RADIUS, min(self.WIDTH - self.PADDLE_RADIUS, self.player2_pos[0]))
            self.player2_pos[1] = max(self.PADDLE_RADIUS, min(self.HEIGHT // 2 - self.PADDLE_RADIUS, self.player2_pos[1]))

    def check_collisions(self):
        # Check collisions with corner obstacles
        for obstacle in CORNER_OBSTACLES:
            dist = math.hypot(self.ball_pos[0] - obstacle[0], self.ball_pos[1] - obstacle[1])
            overlap = CORNER_OBSTACLE_RADIUS + BALL_RADIUS - dist
            if overlap > 0:
                nx = (self.ball_pos[0] - obstacle[0]) / dist
                ny = (self.ball_pos[1] - obstacle[1]) / dist
                ball_speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
                self.ball_vel[0] = nx * ball_speed * 1.2
                self.ball_vel[1] = ny * ball_speed * 1.2
                self.ball_pos[0] += nx * overlap
                self.ball_pos[1] += ny * overlap

        # Check collisions with paddles
        for paddle_pos, paddle_vel in zip(
            [self.player1_pos, self.player2_pos], [self.player1_vel, self.player2_vel]
        ):
            dist = math.hypot(self.ball_pos[0] - paddle_pos[0], self.ball_pos[1] - paddle_pos[1])
            overlap = PADDLE_RADIUS + BALL_RADIUS - dist
            if overlap > 0:
                nx = (self.ball_pos[0] - paddle_pos[0]) / dist
                ny = (self.ball_pos[1] - paddle_pos[1]) / dist
                ball_speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
                paddle_speed = math.hypot(paddle_vel[0], paddle_vel[1])
                self.ball_vel[0] = nx * (ball_speed + paddle_speed)
                self.ball_vel[1] = ny * (ball_speed + paddle_speed)
                self.ball_pos[0] += nx * overlap
                self.ball_pos[1] += ny * overlap

             # 確保球與牆壁不重疊
        if self.ball_pos[0] - BALL_RADIUS < 0:
            self.ball_pos[0] = BALL_RADIUS
            self.ball_vel[0] = abs(self.ball_vel[0])
        if self.ball_pos[0] + BALL_RADIUS > WIDTH:
            self.ball_pos[0] = WIDTH - BALL_RADIUS
            self.ball_vel[0] = -abs(self.ball_vel[0])
        if self.ball_pos[1] - BALL_RADIUS < 0:
            self.ball_pos[1] = BALL_RADIUS
            self.ball_vel[1] = abs(self.ball_vel[1])
        if self.ball_pos[1] + BALL_RADIUS > HEIGHT:
            self.ball_pos[1] = HEIGHT - BALL_RADIUS
            self.ball_vel[1] = -abs(self.ball_vel[1])

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.move_paddles()
            self.move_ball()
            self.check_collisions()
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)
    
if __name__ == "__main__":
    game = AirHockeyGame()
    game.run()