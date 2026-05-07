import pygame
import sys
import math
import random
import os
import time
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
PADDLE_MAX_SPEED = 25
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
    
    def __init__(self, render=True):
        self.render = render
        if not self.render:
            # 告诉 SDL 用 dummy driver，不会弹窗口
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        if self.render:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Air Hockey")
        else:
            # 保证后续调用 self.screen 不出错
            self.screen = pygame.Surface((WIDTH, HEIGHT))   
        self.clock = pygame.time.Clock()
        self.just_one_bounce_hit = False
        self.reset()  # 初始化遊戲狀態
        self.BALL_RADIUS = BALL_RADIUS
        self.PADDLE_RADIUS = PADDLE_RADIUS
        self.PADDLE_MAX_SPEED = PADDLE_MAX_SPEED
        self.BALL_MAX_SPEED = BALL_MAX_SPEED
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.GOAL_WIDTH = GOAL_WIDTH
        self.serving_player = 2  # 初始由藍方發球
        self.last_paddle_hit = None
        

    def reset(self):
        # Reset game state
        self.ball_pos = [WIDTH // 2, HEIGHT // 2]
        self.ball_vel = [0, 0]
        self.player1_pos = [WIDTH // 2, HEIGHT - 100]
        self.player2_pos = [WIDTH // 2, 100]
        self.player1_vel = [0, 0]
        self.player2_vel = [0, 0]
        self.player1_score = 0
        self.player2_score = 0
        self.serving_player = 2  # 第一球由藍方發球
        self.last_paddle_hit = None
        self.just_one_bounce_hit = False

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
        self.player1_pos = [WIDTH // 2, HEIGHT - 100]
        self.player2_pos = [WIDTH // 2, 100]
    def _predict_trajectory(self, max_steps=500, dt=1.0):
        """
        返回一串 (x,y) 座標，表示從當前球心位置和速度出發
        經過左右牆反彈後的軌跡，直到 y <= BALL_RADIUS（到達對方球門線）
        或超出 max_steps 為止。
        """
        points = []
        x, y = self.ball_pos
        vx, vy = self.ball_vel
        for _ in range(max_steps):
            # 前進一步
            x += vx * dt
            y += vy * dt

            # 左右牆反彈
            if x <= BALL_RADIUS:
                x = BALL_RADIUS + (BALL_RADIUS - x)
                vx = -vx
            elif x >= WIDTH - BALL_RADIUS:
                x = (WIDTH - BALL_RADIUS) - (x - (WIDTH - BALL_RADIUS))
                vx = -vx

            points.append((int(x), int(y)))

            # 到達對方球門線或自己底線，就停
            if y <= BALL_RADIUS or y >= HEIGHT - BALL_RADIUS:
                break

        return points
    def _will_score_with_one_bounce(self):
        """
        Determines if the current ball state (self.ball_pos, self.ball_vel)
        will result in a score, either directly or via one side-wall bounce,
        into the goal it is currently heading towards.
        """
        bx, by = self.ball_pos
        vx, vy = self.ball_vel

        if abs(vy) < 1e-6: # Not moving vertically enough for a typical goal.
            return False 

        # Goal mouth definition for ball's center
        goal_actual_left_edge = (self.WIDTH - self.GOAL_WIDTH) / 2
        goal_actual_right_edge = (self.WIDTH + self.GOAL_WIDTH) / 2
        
        # Effective x-range for the ball's center to pass through for a score
        score_x_min = goal_actual_left_edge + self.BALL_RADIUS
        score_x_max = goal_actual_right_edge - self.BALL_RADIUS

        # Determine target_y (opponent's goal line y-coordinate for the ball's center)
        target_y = 0
        if vy < 0: # Moving towards top goal (Player 2's goal)
            target_y = self.BALL_RADIUS
        else: # Moving towards bottom goal (Player 1's goal)
            target_y = self.HEIGHT - self.BALL_RADIUS

        # --- Helper function to check if a straight shot scores ---
        def check_straight_shot_score(x0, y0, current_vx, current_vy, goal_plane_y):
            if abs(current_vy) < 1e-6: return False # Not moving vertically towards goal plane

            # Check if moving towards the goal plane in the y-direction
            # (goal_plane_y - y0) and current_vy must have the same sign (or one is zero if already on plane)
            if (goal_plane_y > y0 and current_vy <= 0) or \
               (goal_plane_y < y0 and current_vy >= 0):
                return False # Moving away from goal plane or parallel to it on the wrong side

            time_to_goal_plane = (goal_plane_y - y0) / current_vy
            
            if time_to_goal_plane < -1e-6: # Check for small negative t due to float precision if starting on line
                 # If already on goal line, check x. If past goal line moving away, this is false.
                 if abs(y0 - goal_plane_y) > 1e-3: # Ensure not just numerical noise if on the line
                    return False # Moving away or already past and moving further away

            x_at_goal_plane = x0 + current_vx * time_to_goal_plane
            
            return score_x_min <= x_at_goal_plane <= score_x_max

        # 1) Check Direct Shot
        if check_straight_shot_score(bx, by, vx, vy, target_y):
            return True

        # 2) Check One side-wall bounce
        time_to_target_y_direct = float('inf')
        # Calculate time to reach target_y directly IF ball is moving towards it
        if (vy < 0 and by > target_y) or (vy > 0 and by < target_y): # Ball is between its current y and target_y
            if abs(vy) > 1e-6:
                time_to_target_y_direct = (target_y - by) / vy
        
        for wall_side in ["left", "right"]:
            wall_x_ball_center = 0 # x-coordinate of ball's center when touching wall
            
            if wall_side == "left":
                if vx >= 0: continue # Not moving towards left wall
                wall_x_ball_center = self.BALL_RADIUS
            else: # wall_side == "right"
                if vx <= 0: continue # Not moving towards right wall
                wall_x_ball_center = self.WIDTH - self.BALL_RADIUS
            
            if abs(vx) < 1e-6: continue # Should be caught by vx checks, but for safety

            time_to_hit_wall = (wall_x_ball_center - bx) / vx

            # Conditions for valid wall hit leading to a potential score:
            # a) Must hit wall in positive time.
            # b) Must hit wall *before* it would have crossed the target_y plane directly.
            if time_to_hit_wall < 0 or time_to_hit_wall >= time_to_target_y_direct:
                continue

            y_at_wall_impact = by + vy * time_to_hit_wall

            # c) The y-coordinate of wall impact must be such that the ball is still on its way to target_y.
            #    (e.g., if vy < 0 (going up to target_y=BALL_RADIUS), then y_at_wall_impact must be > BALL_RADIUS)
            #    This also implicitly checks it doesn't hit the top/bottom edge of table before side wall.
            if (vy < 0 and y_at_wall_impact <= target_y) or \
               (vy > 0 and y_at_wall_impact >= target_y):
                continue 

            # If all bounce conditions met, check if shot from wall scores:
            # Position after impact: (wall_x_ball_center, y_at_wall_impact)
            # Velocity after impact: (-vx, vy)
            if check_straight_shot_score(wall_x_ball_center, y_at_wall_impact, -vx, vy, target_y):
                return True
                
        return False

    def draw(self):
        if not self.render:
            return
        self.screen.fill(WHITE)
        pygame.draw.line(self.screen, BLACK, (0, HEIGHT // 2), (WIDTH, HEIGHT // 2), 2)
        pygame.draw.rect(self.screen, RED, ((WIDTH - GOAL_WIDTH) // 2, HEIGHT - 10, GOAL_WIDTH, 10))
        pygame.draw.rect(self.screen, BLUE, ((WIDTH - GOAL_WIDTH) // 2, 0, GOAL_WIDTH, 10))
        for obs in CORNER_OBSTACLES:
            pygame.draw.circle(self.screen, BLACK, (int(obs[0]), int(obs[1])), CORNER_OBSTACLE_RADIUS)
        pygame.draw.circle(self.screen, BLACK, (int(self.ball_pos[0]), int(self.ball_pos[1])), BALL_RADIUS)
        pygame.draw.circle(self.screen, RED,   (int(self.player1_pos[0]), int(self.player1_pos[1])), PADDLE_RADIUS)
        pygame.draw.circle(self.screen, BLUE,  (int(self.player2_pos[0]), int(self.player2_pos[1])), PADDLE_RADIUS)

        # 在球之後、球拍之前，加畫綠色軌跡線
        traj = self._predict_trajectory()
        color = RED if self._will_score_with_one_bounce() else GREEN
        if len(traj) >= 2:
            pygame.draw.lines(self.screen, color, False, traj, 2)

        # 分數文字
        font = pygame.font.Font(None, 74)
        score_text = font.render(f"{self.player2_score} - {self.player1_score}", True, BLACK)
        self.screen.blit(score_text, (WIDTH//2 - score_text.get_width()//2, HEIGHT//2 - 40))
        pygame.display.flip()

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
                self.reset_ball(scoring_player="player2")  # Player 2 scores
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
        # 如果要训练需要关闭下面三行（鼠标控制）
        # mouse_x, mouse_y = pygame.mouse.get_pos()
        # self.player1_vel[0] = mouse_x - self.player1_pos[0]
        # self.player1_vel[1] = mouse_y - self.player1_pos[1]

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

        # ----- Rule-based AI for Player2 (蓝方) -----
        # 如果球在“正中心”区域，主动前去击球
        center_x, center_y = self.WIDTH / 2, self.HEIGHT / 2
        if abs(self.ball_pos[0] - center_x) < self.BALL_RADIUS and abs(self.ball_pos[1] - center_y) < self.BALL_RADIUS:
            dx = self.ball_pos[0] - self.player2_pos[0]
            dy = self.ball_pos[1] - self.player2_pos[1]
            dist = math.hypot(dx, dy)
            if dist > 1e-6:
                scale = min(self.PADDLE_MAX_SPEED, dist) / dist
                self.player2_vel[0] = dx * scale
                self.player2_vel[1] = dy * scale
            else:
                self.player2_vel = [0, 0]
        else:
            # 原有 AI 逻辑：预测拦截或随机试探
            if random.random() < 0.1:
                self.player2_vel[0] = random.uniform(-self.PADDLE_MAX_SPEED, self.PADDLE_MAX_SPEED)
                self.player2_vel[1] = random.uniform(-self.PADDLE_MAX_SPEED, self.PADDLE_MAX_SPEED)
            else:
                # 预测球到某 y_line 时的拦截点
                y_line = random.uniform(self.HEIGHT * 0.01, self.HEIGHT * 0.4)
                if abs(self.ball_vel[1]) < 1e-6:
                    intercept_x = self.ball_pos[0]
                else:
                    t = (y_line - self.ball_pos[1]) / self.ball_vel[1]
                    intercept_x = self.ball_pos[0] + self.ball_vel[0] * t
                intercept_x = max(self.BALL_RADIUS, min(self.WIDTH - self.BALL_RADIUS, intercept_x))
                dx = intercept_x - self.player2_pos[0]
                dy = y_line        - self.player2_pos[1]
                dist = math.hypot(dx, dy)
                if dist > 1e-6:
                    scale = min(self.PADDLE_MAX_SPEED, dist) / dist
                    self.player2_vel[0] = dx * scale
                    self.player2_vel[1] = dy * scale
                else:
                    self.player2_vel = [0, 0]

        # 更新玩家2的位置
        self.player2_pos[0] += self.player2_vel[0]
        self.player2_pos[1] += self.player2_vel[1]

        # 保持玩家2在合法范围内
        self.player2_pos[0] = max(self.PADDLE_RADIUS, min(self.WIDTH - self.PADDLE_RADIUS, self.player2_pos[0]))
        self.player2_pos[1] = max(self.PADDLE_RADIUS, min(self.HEIGHT // 2 - self.PADDLE_RADIUS, self.player2_pos[1]))

    def check_collisions(self):
        self.just_one_bounce_hit = False
        # 检查与球拍的碰撞
        for idx, (paddle_pos, paddle_vel) in enumerate(
            zip([self.player1_pos, self.player2_pos], [self.player1_vel, self.player2_vel])
        ):
            dist = math.hypot(self.ball_pos[0] - paddle_pos[0], self.ball_pos[1] - paddle_pos[1])
            overlap = PADDLE_RADIUS + BALL_RADIUS - dist
            if overlap > 0:
                self.last_paddle_hit = 'player1' if idx == 0 else 'player2'
                if idx == 0 and self._will_score_with_one_bounce():
                    self.just_one_bounce_hit = True
                    print("One bounce hit!")
                # print(f"Last paddle hit: {self.last_paddle_hit}")
                # 避免除以零
                if dist < 1e-6:
                    nx, ny = 1.0, 0.0  # 默认方向
                else:
                    nx = (self.ball_pos[0] - paddle_pos[0]) / dist
                    ny = (self.ball_pos[1] - paddle_pos[1]) / dist
                ball_speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
                paddle_speed = math.hypot(paddle_vel[0], paddle_vel[1])
                self.ball_vel[0] = nx * (ball_speed + paddle_speed)
                self.ball_vel[1] = ny * (ball_speed + paddle_speed)
                self.ball_pos[0] += nx * overlap
                self.ball_pos[1] += ny * overlap
        # 检查与角落障碍物的碰撞
        for obstacle in CORNER_OBSTACLES:
            dist = math.hypot(self.ball_pos[0] - obstacle[0], self.ball_pos[1] - obstacle[1])
            overlap = CORNER_OBSTACLE_RADIUS + BALL_RADIUS - dist
            if overlap > 0:
                # 避免除以零
                if dist < 1e-6:
                    nx, ny = 1.0, 0.0  # 默认方向
                else:
                    nx = (self.ball_pos[0] - obstacle[0]) / dist
                    ny = (self.ball_pos[1] - obstacle[1]) / dist
                ball_speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
                self.ball_vel[0] = nx * ball_speed * 1.2
                self.ball_vel[1] = ny * ball_speed * 1.2
                self.ball_pos[0] += nx * overlap
                self.ball_pos[1] += ny * overlap
        # 确保球不超出边界
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
            # Print debugging information
            # print(f"Ball Position: {self.ball_pos}, Ball Velocity: {self.ball_vel}")
            # print(f"Player1 Position: {self.player1_pos}, Player1 Velocity: {self.player1_vel}")
            # print(f"Player2 Position: {self.player2_pos}, Player2 Velocity: {self.player2_vel}")
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)
            # time.sleep(1/20)

    
if __name__ == "__main__":
    game = AirHockeyGame()
    game.run()