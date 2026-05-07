import pygame
import sys
import math
import random

# Define constants for the game
WIDTH, HEIGHT = 450, 800
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
BALL_RADIUS = 15
PADDLE_RADIUS = 40
CORNER_OBSTACLE_RADIUS = 30
FRICTION = 1
PADDLE_MAX_SPEED = 8
BALL_MAX_SPEED = 15
GOAL_WIDTH = 200

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
        self.serving_player = 2  # 初始由藍方發球
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
        self.serving_player = 2  # 第一球由藍方發球

    def reset_ball(self, scoring_player=None):
        # Reset ball velocity
        self.ball_vel = [0, 0]

        # Determine ball position and serving player based on scoring player
        if scoring_player == "player1":  # Player 1 scored, ball starts in Player 2's half
            self.ball_pos = [WIDTH // 2, HEIGHT // 4]  # Closer to Player 2's half
            self.serving_player = 2  # Next serve by Player 2 (Blue)
        elif scoring_player == "player2":  # Player 2 scored, ball starts in Player 1's half
            self.ball_pos = [WIDTH // 2, 3 * HEIGHT // 4]  # Closer to Player 1's half
            self.serving_player = 1  # Next serve by Player 1 (Red)
        else:  # Default case for initial game start
            self.ball_pos = [WIDTH // 2, HEIGHT // 2]

        # Reset paddle positions
        self.player1_pos = [WIDTH // 2, HEIGHT - 50]
        self.player2_pos = [WIDTH // 2, 50]

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
                self.reset_ball(scoring_player="player1")  # Player 1 scores
            else:
                self.ball_vel[1] = -self.ball_vel[1]
                self.ball_pos[1] = BALL_RADIUS

        if self.ball_pos[1] + BALL_RADIUS >= HEIGHT:  # Ball reaches bottom
            if (WIDTH - GOAL_WIDTH) // 2 <= self.ball_pos[0] <= (WIDTH + GOAL_WIDTH) // 2:
                self.player2_score += 1
                self.reset_ball(scoring_player="player1")  # Player 2 scores
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
        keys = pygame.key.get_pressed()

        # Player 1 movement (arrow keys)
        if keys[pygame.K_LEFT]:
            self.player1_vel[0] -= 1
        if keys[pygame.K_RIGHT]:
            self.player1_vel[0] += 1
        if keys[pygame.K_UP]:
            self.player1_vel[1] -= 1
        if keys[pygame.K_DOWN]:
            self.player1_vel[1] += 1

        # AI for Player 2
        if random.random() < 0.1:  # 10% 機率採取隨機動作
            self.player2_vel[0] = random.uniform(-PADDLE_MAX_SPEED, PADDLE_MAX_SPEED)
            self.player2_vel[1] = random.uniform(-PADDLE_MAX_SPEED, PADDLE_MAX_SPEED)
        else:  # 正常跟隨球
            if self.ball_pos[1] < HEIGHT // 2:
                if self.ball_pos[0] > self.player2_pos[0]:
                    self.player2_vel[0] = min(PADDLE_MAX_SPEED, self.ball_pos[0] - self.player2_pos[0])
                elif self.ball_pos[0] < self.player2_pos[0]:
                    self.player2_vel[0] = max(-PADDLE_MAX_SPEED, self.ball_pos[0] - self.player2_pos[0])

                if self.ball_pos[1] > self.player2_pos[1]:
                    self.player2_vel[1] = min(PADDLE_MAX_SPEED, self.ball_pos[1] - self.player2_pos[1])
                elif self.ball_pos[1] < self.player2_pos[1]:
                    self.player2_vel[1] = max(-PADDLE_MAX_SPEED, self.ball_pos[1] - self.player2_pos[1])

        # 更新球拍位置
        for pos, vel, min_y, max_y in zip(
            [self.player1_pos, self.player2_pos],
            [self.player1_vel, self.player2_vel],
            [HEIGHT // 2, 0],
            [HEIGHT, HEIGHT // 2],
        ):
            pos[0] += vel[0]
            pos[1] += vel[1]

            # Apply friction
            vel[0] *= FRICTION
            vel[1] *= FRICTION

            # Limit paddle speed
            speed = math.hypot(vel[0], vel[1])
            if speed > PADDLE_MAX_SPEED:
                scale = PADDLE_MAX_SPEED / speed
                vel[0] *= scale
                vel[1] *= scale

            # Keep paddles within bounds
            pos[0] = max(PADDLE_RADIUS, min(WIDTH - PADDLE_RADIUS, pos[0]))
            pos[1] = max(min_y + PADDLE_RADIUS, min(max_y - PADDLE_RADIUS, pos[1]))

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