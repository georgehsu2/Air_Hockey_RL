import pygame
import sys
import math
import random
import torch
import torch.nn as nn

# Game constants
WIDTH, HEIGHT = 450, 850
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
BALL_RADIUS = 20
PADDLE_RADIUS = 40
CORNER_RADIUS = 30
GOAL_WIDTH = 150
FRICTION = 1
PADDLE_MAX_SPEED = 50
BALL_MAX_SPEED = 20
# Corner obstacle positions
CORNER_POSITIONS = [
    (CORNER_RADIUS, CORNER_RADIUS),
    (WIDTH - CORNER_RADIUS, CORNER_RADIUS),
    (CORNER_RADIUS, HEIGHT - CORNER_RADIUS),
    (WIDTH - CORNER_RADIUS, HEIGHT - CORNER_RADIUS)
]

# Actor-Critic network (unchanged)
torch.backends.cudnn.benchmark = True
class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.common = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.actor = nn.Linear(128, output_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        h = self.common(x)
        return self.actor(h), self.critic(h)

# Load a trained model (unchanged)
def load_model(path, input_dim, output_dim):
    model = ActorCritic(input_dim, output_dim)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

class AirHockeyGame:
    def __init__(self, model_path):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Air Hockey AI (Pure Python Physics)")
        self.clock = pygame.time.Clock()

        # physics state
        self.reset()
        # store corners
        self.corner_positions = CORNER_POSITIONS

        # load AI
        inp_dim = len(self.get_observation())
        out_dim = 2
        self.model = load_model(model_path, inp_dim, out_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.last_paddle_hit = None
        # flag for “will score after a single side-bounce” on next collision
        self.just_one_bounce_hit = False

    def reset(self):
        self.ball_pos   = [WIDTH//2, HEIGHT//2]
        self.ball_vel   = [0.0, 0.0]
        self.player1_pos = [WIDTH//2, HEIGHT - 50]
        self.player1_vel = [0.0, 0.0]
        self.player2_pos = [WIDTH//2, 50]
        self.player2_vel = [0.0, 0.0]
        self.score1     = 0
        self.score2     = 0
        self.last_paddle_hit = None
        self.just_one_bounce_hit = False
    def get_observation(self):
        # Mirror and normalize state into [-1,1]
        fb_x, fb_y = self.ball_pos[0], HEIGHT - self.ball_pos[1]
        fv_x, fv_y = self.ball_vel[0], -self.ball_vel[1]
        fp1_x, fp1_y = self.player2_pos[0], HEIGHT - self.player2_pos[1]
        fv1_x, fv1_y = self.player2_vel[0], -self.player2_vel[1]
        fp2_x, fp2_y = self.player1_pos[0], HEIGHT - self.player1_pos[1]
        fv2_x, fv2_y = self.player1_vel[0], -self.player1_vel[1]
        to_p_x, to_p_y = fb_x - fp1_x, fb_y - fp1_y
        to_goal_x, to_goal_y = WIDTH/2 - fb_x, 0 - fb_y
        to_opp_x, to_opp_y = fb_x - fp2_x, fb_y - fp2_y
        
        def norm_pair(x, lo, hi):
            return [2*(x[0]-lo[0])/(hi[0]-lo[0]+1e-8)-1,
                    2*(x[1]-lo[1])/(hi[1]-lo[1]+1e-8)-1]
        parts = []
        parts += norm_pair([fb_x, fb_y], [0,0], [WIDTH, HEIGHT])
        parts += [fv_x/BALL_MAX_SPEED, fv_y/BALL_MAX_SPEED]
        parts += norm_pair([fp1_x, fp1_y], [0,0], [WIDTH, HEIGHT])
        parts += [fv1_x/PADDLE_MAX_SPEED, fv1_y/PADDLE_MAX_SPEED]
        parts += norm_pair([fp2_x, fp2_y], [0,0], [WIDTH, HEIGHT])
        parts += [fv2_x/PADDLE_MAX_SPEED, fv2_y/PADDLE_MAX_SPEED]
        ml = max(WIDTH, HEIGHT)
        parts += [to_p_x/ml, to_p_y/ml]
        parts += [to_goal_x/ml, to_goal_y/ml]
        parts += [to_opp_x/ml, to_opp_y/ml]
        return parts
    def _predict_trajectory(self, max_steps=500, dt=1.0):
        """
        Returns a list of (x,y) positions of the ball's center
        simulating up to one side-wall bounce until it reaches
        y <= BALL_RADIUS or y >= HEIGHT - BALL_RADIUS.
        """
        points = []
        x, y = self.ball_pos
        vx, vy = self.ball_vel
        for _ in range(max_steps):
            x += vx * dt
            y += vy * dt
            if x <= BALL_RADIUS:
                x = BALL_RADIUS + (BALL_RADIUS - x)
                vx = -vx
            elif x >= WIDTH - BALL_RADIUS:
                x = (WIDTH - BALL_RADIUS) - (x - (WIDTH - BALL_RADIUS))
                vx = -vx
            points.append((int(x), int(y)))
            if y <= BALL_RADIUS or y >= HEIGHT - BALL_RADIUS:
                break
        return points

    def _will_score_with_one_bounce(self):
        """
        Returns True if the current ball velocity and position
        will lead to a score either directly or after one wall bounce.
        """
        bx, by = self.ball_pos
        vx, vy = self.ball_vel
        if abs(vy) < 1e-6:
            return False
        left_edge = (WIDTH - GOAL_WIDTH) / 2 + BALL_RADIUS
        right_edge = (WIDTH + GOAL_WIDTH) / 2 - BALL_RADIUS
        target_y = BALL_RADIUS if vy < 0 else HEIGHT - BALL_RADIUS
        def check_shot(x0, y0, cvx, cvy):
            if abs(cvy) < 1e-6:
                return False
            t = (target_y - y0) / cvy
            if t < 0:
                return False
            x_at = x0 + cvx * t
            return left_edge <= x_at <= right_edge
        # direct
        if check_shot(bx, by, vx, vy):
            return True
        # bounce
        if abs(vx) < 1e-6:
            return False
        # time to wall
        if vx < 0:
            wall_x = BALL_RADIUS
        else:
            wall_x = WIDTH - BALL_RADIUS
        t_wall = (wall_x - bx) / vx
        if t_wall <= 0:
            return False
        y_wall = by + vy * t_wall
        if y_wall < BALL_RADIUS or y_wall > HEIGHT - BALL_RADIUS:
            return False
        # post-bounce
        return check_shot(wall_x, y_wall, -vx, vy)
    def control_bottom(self):
        """
        Rule-based AI for Player1 (红方):
        如果球在场地正中心的小区域，主动前去击球；
        否则 10% 随机试探，否则预测拦截点并移动。
        """
        # 场地中心坐标
        center_x, center_y = WIDTH / 2, HEIGHT / 2

        # 1) 如果球在中心区域，直接前往击球
        if abs(self.ball_pos[0] - center_x) < BALL_RADIUS and abs(self.ball_pos[1] - center_y) < BALL_RADIUS:
            dx = self.ball_pos[0] - self.player1_pos[0]
            dy = self.ball_pos[1] - self.player1_pos[1]
            dist = math.hypot(dx, dy)
            if dist > 1e-6:
                scale = min(PADDLE_MAX_SPEED, dist) / dist
                self.player1_vel = [dx * scale, dy * scale]
            else:
                self.player1_vel = [0, 0]
        else:
            # 2) 10% 随机试探
            if random.random() < 0.8:
                self.player1_vel = [
                    random.uniform(-PADDLE_MAX_SPEED, PADDLE_MAX_SPEED),
                    random.uniform(-PADDLE_MAX_SPEED, PADDLE_MAX_SPEED)
                ]
            else:
                # 3) 预测球到达底部半场某一 y_line 的拦截点
                y_line = random.uniform(HEIGHT * 0.6, HEIGHT - BALL_RADIUS)
                if abs(self.ball_vel[1]) < 1e-6:
                    intercept_x = self.ball_pos[0]
                else:
                    t = (y_line - self.ball_pos[1]) / self.ball_vel[1]
                    intercept_x = self.ball_pos[0] + self.ball_vel[0] * t
                # 限制拦截点在边界内
                intercept_x = max(BALL_RADIUS, min(WIDTH - BALL_RADIUS, intercept_x))
                dx = intercept_x - self.player1_pos[0]
                dy = y_line        - self.player1_pos[1]
                dist = math.hypot(dx, dy)
                if dist > 1e-6:
                    scale = min(PADDLE_MAX_SPEED, dist) / dist
                    self.player1_vel = [dx * scale, dy * scale]
                else:
                    self.player1_vel = [0, 0]

        # 4) 更新 player1 位置并钳制到下半场
        self.player1_pos[0] += self.player1_vel[0]
        self.player1_pos[1] += self.player1_vel[1]
        self.player1_pos[0] = max(PADDLE_RADIUS, min(WIDTH  - PADDLE_RADIUS, self.player1_pos[0]))
        self.player1_pos[1] = max(HEIGHT//2 + PADDLE_RADIUS, min(HEIGHT - PADDLE_RADIUS, self.player1_pos[1]))

    def control_top(self):
        # unchanged from B
        obs = self.get_observation()
        t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad(): logits, _ = self.model(t)
        act = torch.tanh(logits).squeeze().cpu().numpy()
        vx, vy = act * PADDLE_MAX_SPEED
        self.player2_vel = [vx, -vy]
        self.player2_pos[0] += self.player2_vel[0]
        self.player2_pos[1] += self.player2_vel[1]
        self.player2_pos[0] = max(PADDLE_RADIUS, min(WIDTH-PADDLE_RADIUS, self.player2_pos[0]))
        self.player2_pos[1] = max(PADDLE_RADIUS, min(HEIGHT//2-PADDLE_RADIUS, self.player2_pos[1]))

    def move_ball(self):
        # 1) 更新球的位置
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]

        # 2) 摩擦衰减
        self.ball_vel[0] *= FRICTION
        self.ball_vel[1] *= FRICTION

        # 3) 左右墙壁反弹
        if self.ball_pos[0] - BALL_RADIUS <= 0 or self.ball_pos[0] + BALL_RADIUS >= WIDTH:
            self.ball_vel[0] = -self.ball_vel[0]
            self.ball_pos[0] = max(BALL_RADIUS, min(WIDTH - BALL_RADIUS, self.ball_pos[0]))

        # 4) 顶端（上方）边界：y - BALL_RADIUS <= 0
        if self.ball_pos[1] - BALL_RADIUS <= 0:
            # 如果球心 x 在上方球门范围内，则下方玩家得分 -> score1 增加
            if (WIDTH - GOAL_WIDTH)//2 <= self.ball_pos[0] <= (WIDTH + GOAL_WIDTH)//2:
                self.score1 += 1
                # 达到 15 分时，重置比分并中场发球
                if self.score1 >= 15:
                    self.score1 = 0
                    self.score2 = 0
                    self.reset_ball()
                else:
                    self.reset_ball(scoring_player="player1")
            else:
                # 撞墙弹回
                self.ball_vel[1] = -self.ball_vel[1]
                self.ball_pos[1] = BALL_RADIUS

        # 5) 底端（下方）边界：y + BALL_RADIUS >= HEIGHT
        if self.ball_pos[1] + BALL_RADIUS >= HEIGHT:
            # 如果球心 x 在下方球门范围内，则上方玩家得分 -> score2 增加
            if (WIDTH - GOAL_WIDTH)//2 <= self.ball_pos[0] <= (WIDTH + GOAL_WIDTH)//2:
                self.score2 += 1
                # 达到 15 分时，重置比分并中场发球
                if self.score2 >= 15:
                    self.score1 = 0
                    self.score2 = 0
                    self.reset_ball()
                else:
                    self.reset_ball(scoring_player="player2")
            else:
                # 撞墙弹回
                self.ball_vel[1] = -self.ball_vel[1]
                self.ball_pos[1] = HEIGHT - BALL_RADIUS

        # 6) 限制球速不超过最大值
        speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
        if speed > BALL_MAX_SPEED:
            scale = BALL_MAX_SPEED / speed
            self.ball_vel[0] *= scale
            self.ball_vel[1] *= scale

    def reset_ball(self, scoring_player=None):
        self.ball_vel = [0, 0]
        if scoring_player == "player1":
            self.ball_pos = [WIDTH//2, HEIGHT//4]
            self.serving_player = 2
        elif scoring_player == "player2":
            self.ball_pos = [WIDTH//2, 3*HEIGHT//4]
            self.serving_player = 1
        else:
            self.ball_pos = [WIDTH//2, HEIGHT//2]

        self.player1_pos = [WIDTH//2, HEIGHT - 100]
        self.player2_pos = [WIDTH//2, 100]
        self.player1_vel = [0, 0]
        self.player2_vel = [0, 0]

        self.last_paddle_hit = None
        self.just_one_bounce_hit = False

        for pos, vel, player in [
            (self.player1_pos, self.player1_vel, "player1"),
            (self.player2_pos, self.player2_vel, "player2")
        ]:
            dx = self.ball_pos[0] - pos[0]
            dy = self.ball_pos[1] - pos[1]
            dist = math.hypot(dx, dy)
            overlap = PADDLE_RADIUS + BALL_RADIUS - dist
            if overlap > 0:
                self.last_paddle_hit = player
                if dist < 1e-6:
                    nx, ny = 1.0, 0.0
                else:
                    nx, ny = dx/dist, dy/dist
                bs = math.hypot(self.ball_vel[0], self.ball_vel[1])
                ps = math.hypot(vel[0], vel[1])
                self.ball_vel[0] = nx * (bs + ps)
                self.ball_vel[1] = ny * (bs + ps)
                self.ball_pos[0] += nx * overlap
                self.ball_pos[1] += ny * overlap
        self.last_paddle_hit = None
    def check_collisions(self):
        # clear the one-bounce flag each frame
        self.just_one_bounce_hit = False

        # corner obstacles
        for ox, oy in self.corner_positions:
            dx = self.ball_pos[0] - ox
            dy = self.ball_pos[1] - oy
            dist = math.hypot(dx, dy)
            overlap = CORNER_RADIUS + BALL_RADIUS - dist
            if overlap > 0:
                if dist < 1e-6:
                    nx, ny = 1.0, 0.0
                else:
                    nx, ny = dx/dist, dy/dist
                speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
                self.ball_vel[0] = nx * speed * 1.2
                self.ball_vel[1] = ny * speed * 1.2
                self.ball_pos[0] += nx * overlap
                self.ball_pos[1] += ny * overlap

        # paddle collisions
        for idx, (paddle_pos, paddle_vel) in enumerate([
            (self.player1_pos, self.player1_vel),
            (self.player2_pos, self.player2_vel)
        ]):
            dx = self.ball_pos[0] - paddle_pos[0]
            dy = self.ball_pos[1] - paddle_pos[1]
            dist = math.hypot(dx, dy)
            overlap = PADDLE_RADIUS + BALL_RADIUS - dist
            if overlap > 0:
                self.last_paddle_hit = "player1" if idx == 0 else "player2"
                # if agent’s paddle (player2) hits & predicts one-bounce goal → flag
                if idx == 1 and self._will_score_with_one_bounce():
                    self.just_one_bounce_hit = True
                if dist < 1e-6:
                    nx, ny = 1.0, 0.0
                else:
                    nx, ny = dx/dist, dy/dist
                bs = math.hypot(self.ball_vel[0], self.ball_vel[1])
                ps = math.hypot(paddle_vel[0], paddle_vel[1])
                self.ball_vel[0] = nx * (bs + ps)
                self.ball_vel[1] = ny * (bs + ps)
                self.ball_pos[0] += nx * overlap
                self.ball_pos[1] += ny * overlap

        # wall bounce
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

    def _will_score_with_one_bounce(self):
        """
        判斷當前球狀態是否會直接或經過一次側牆反彈後
        進入其運動方向的球門。
        返回 True/False。
        """
        bx, by = self.ball_pos
        vx, vy = self.ball_vel
        # 垂直速度不足時不考慮
        if abs(vy) < 1e-6:
            return False

        # 球門邊界
        goal_left = (WIDTH - GOAL_WIDTH) / 2 + BALL_RADIUS
        goal_right = (WIDTH + GOAL_WIDTH) / 2 - BALL_RADIUS

        # 目標 y
        if vy < 0:
            target_y = BALL_RADIUS
        else:
            target_y = HEIGHT - BALL_RADIUS

        def check_straight(x0, y0, cvx, cvy):
            # 垂直方向是否朝目標
            if (target_y - y0) * cvy <= 0:
                return False
            t = (target_y - y0) / cvy
            x_at = x0 + cvx * t
            return goal_left <= x_at <= goal_right

        # 1) 直接射門
        if check_straight(bx, by, vx, vy):
            return True

        # 2) 一次側牆反彈
        # 計算直接到達目標的時間
        t_direct = float('inf')
        if (vy < 0 and by > target_y) or (vy > 0 and by < target_y):
            t_direct = (target_y - by) / vy

        for wall in ['left', 'right']:
            if wall == 'left':
                if vx >= 0: continue
                wall_x = BALL_RADIUS
            else:
                if vx <= 0: continue
                wall_x = WIDTH - BALL_RADIUS

            t_hit = (wall_x - bx) / vx
            if t_hit <= 0 or t_hit >= t_direct:
                continue

            y_hit = by + vy * t_hit
            # 反彈後位置與速度
            new_vx = -vx
            new_vy = vy
            # 確保反彈時仍在場地內
            if (vy < 0 and y_hit <= target_y) or (vy > 0 and y_hit >= target_y):
                continue

            if check_straight(wall_x, y_hit, new_vx, new_vy):
                return True

        return False
    def draw(self):
        # 如果想關閉渲染，可設 self.render = False
        if hasattr(self, 'render') and not self.render:
            return
        self.screen.fill(WHITE)
        # 中線與球門
        pygame.draw.line(self.screen, BLACK, (0, HEIGHT // 2), (WIDTH, HEIGHT // 2), 2)
        pygame.draw.rect(self.screen, RED, ((WIDTH - GOAL_WIDTH) // 2, HEIGHT - 10, GOAL_WIDTH, 10))
        pygame.draw.rect(self.screen, BLUE, ((WIDTH - GOAL_WIDTH) // 2, 0, GOAL_WIDTH, 10))
        # 角落障礙物
        for ox, oy in CORNER_POSITIONS:
            pygame.draw.circle(self.screen, BLACK, (int(ox), int(oy)), CORNER_RADIUS)
        # 球與球拍
        pygame.draw.circle(self.screen, BLACK, (int(self.ball_pos[0]), int(self.ball_pos[1])), BALL_RADIUS)
        pygame.draw.circle(self.screen, RED,   (int(self.player1_pos[0]), int(self.player1_pos[1])), PADDLE_RADIUS)
        pygame.draw.circle(self.screen, BLUE,  (int(self.player2_pos[0]), int(self.player2_pos[1])), PADDLE_RADIUS)

        # 畫出綠/紅軌跡
        traj = self._predict_trajectory()
        color = RED if self._will_score_with_one_bounce() else GREEN
        if len(traj) >= 2:
            pygame.draw.lines(self.screen, color, False, traj, 2)

        # 分數文字顯示
        font = pygame.font.Font(None, 74)
        score_text = font.render(f"{self.score2} - {self.score1}", True, BLACK)
        self.screen.blit(score_text, (WIDTH//2 - score_text.get_width()//2, HEIGHT//2 - 40))
        pygame.display.flip()

    def run(self):
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
            self.control_bottom(); self.control_top();
            self.move_ball(); self.check_collisions(); self.draw()
            pygame.display.flip(); self.clock.tick(60)
            if self.score1 >= 99999 or self.score2 >= 99999:
                print(f"Game Over: {self.score2}-{self.score1}")
                break
        pygame.quit()

if __name__ == '__main__':
    MODEL_PATH = 'saved_models_20250518_151900/model_episode_10500.pth'
    game = AirHockeyGame(MODEL_PATH)
    game.run()
