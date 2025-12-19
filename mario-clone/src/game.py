import pygame

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 450
GROUND_HEIGHT = 60

GRAVITY = 0.35
MAX_FALL_SPEED = 8
PLAYER_SPEED = 3.6
JUMP_SPEED = 8.5

COLOR_BG = (135, 206, 235)
COLOR_GROUND = (110, 70, 40)
COLOR_PLATFORM = (120, 80, 50)
COLOR_PLAYER = (220, 30, 30)
COLOR_GOOMBA = (160, 90, 60)
COLOR_COIN = (240, 200, 0)
COLOR_GOAL = (50, 200, 70)
COLOR_TEXT = (10, 10, 10)


class Player:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, 26, 34)
        self.x = float(x)
        self.y = float(y)
        self.vx = 0.0
        self.vy = 0.0
        self.on_ground = False


class Goomba:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, 28, 20)
        self.x = float(x)
        self.y = float(y)
        self.vx = 0.0
        self.vy = 0.0
        self.direction = -1
        self.speed = 1.3
        self.on_ground = False
        self.alive = True


class Coin:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, 14, 14)
        self.collected = False


class Level:
    def __init__(self):
        ground_y = SCREEN_HEIGHT - GROUND_HEIGHT
        self.start_pos = (40, ground_y - 34)
        self.solids = [
            pygame.Rect(0, ground_y, 260, GROUND_HEIGHT),
            pygame.Rect(320, ground_y, 220, GROUND_HEIGHT),
            pygame.Rect(620, ground_y, 180, GROUND_HEIGHT),
            pygame.Rect(140, ground_y - 120, 120, 18),
            pygame.Rect(430, ground_y - 160, 140, 18),
        ]
        self.coins = [
            Coin(90, ground_y - 40),
            Coin(170, ground_y - 160),
            Coin(210, ground_y - 160),
            Coin(230, ground_y - 40),
            Coin(360, ground_y - 40),
            Coin(470, ground_y - 200),
            Coin(500, ground_y - 200),
            Coin(520, ground_y - 200),
            Coin(640, ground_y - 40),
            Coin(680, ground_y - 40),
            Coin(720, ground_y - 40),
            Coin(760, ground_y - 40),
        ]
        self.goombas = [
            Goomba(360, ground_y - 20),
            Goomba(650, ground_y - 20),
        ]
        self.goalpost = pygame.Rect(750, ground_y - 120, 20, 120)


class Game:
    def __init__(self, sfx=None):
        self.sfx = sfx or {}
        self.reset()

    def reset(self):
        self.level = Level()
        self.player = Player(*self.level.start_pos)
        self.lives = 3
        self.score = 0
        self.coins_collected = 0
        self.state = "playing"
        self.frame = 0

    def play_sound(self, name):
        sound = self.sfx.get(name)
        if sound:
            sound.play()

    def _move_horizontal(self, entity):
        entity.x += entity.vx
        entity.rect.x = int(entity.x)
        hit_wall = False
        for block in self.level.solids:
            if entity.rect.colliderect(block):
                hit_wall = True
                if entity.vx > 0:
                    entity.rect.right = block.left
                elif entity.vx < 0:
                    entity.rect.left = block.right
                entity.x = entity.rect.x
                entity.vx = 0.0
        return hit_wall

    def _move_vertical(self, entity):
        entity.y += entity.vy
        entity.rect.y = int(entity.y)
        on_ground = False
        for block in self.level.solids:
            if entity.rect.colliderect(block):
                if entity.vy > 0:
                    entity.rect.bottom = block.top
                    on_ground = True
                elif entity.vy < 0:
                    entity.rect.top = block.bottom
                entity.y = entity.rect.y
                entity.vy = 0.0
        entity.on_ground = on_ground

    def _lose_life(self):
        self.lives -= 1
        self.play_sound("hurt")
        if self.lives <= 0:
            self.state = "game_over"
            return
        self.player = Player(*self.level.start_pos)

    def update(self, action):
        if self.state != "playing":
            return

        self.frame += 1
        move_left, move_right, jump = action

        if move_left and not move_right:
            self.player.vx = -PLAYER_SPEED
        elif move_right and not move_left:
            self.player.vx = PLAYER_SPEED
        else:
            self.player.vx = 0.0

        if jump and self.player.on_ground:
            self.player.vy = -JUMP_SPEED
            self.player.on_ground = False
            self.play_sound("jump")

        self.player.vy = min(self.player.vy + GRAVITY, MAX_FALL_SPEED)
        self._move_horizontal(self.player)
        self._move_vertical(self.player)

        for coin in self.level.coins:
            if not coin.collected and self.player.rect.colliderect(coin.rect):
                coin.collected = True
                self.coins_collected += 1
                self.score += 1
                self.play_sound("coin")
                if self.coins_collected % 10 == 0:
                    self.lives += 1

        for goomba in self.level.goombas:
            if not goomba.alive:
                continue
            goomba.vx = goomba.speed * goomba.direction
            goomba.vy = min(goomba.vy + GRAVITY, MAX_FALL_SPEED)
            hit_wall = self._move_horizontal(goomba)
            if hit_wall:
                goomba.direction *= -1
            self._move_vertical(goomba)
            if goomba.rect.top > SCREEN_HEIGHT + 100:
                goomba.alive = False

            if goomba.alive and self.player.rect.colliderect(goomba.rect):
                stomp = self.player.vy > 0 and self.player.rect.bottom - goomba.rect.top < 12
                if stomp:
                    goomba.alive = False
                    self.player.vy = -JUMP_SPEED * 0.6
                    self.play_sound("stomp")
                else:
                    self._lose_life()
                    return

        if self.player.rect.top > SCREEN_HEIGHT:
            self._lose_life()
            return

        if self.player.rect.colliderect(self.level.goalpost):
            self.state = "won"
            self.play_sound("win")

    def get_state(self):
        return {
            "tick": self.frame,
            "status": self.state,
            "lives": self.lives,
            "score": self.score,
            "coins_collected": self.coins_collected,
            "world": {
                "width": SCREEN_WIDTH,
                "height": SCREEN_HEIGHT,
                "ground_y": SCREEN_HEIGHT - GROUND_HEIGHT,
            },
            "player": {
                "x": self.player.rect.x,
                "y": self.player.rect.y,
                "w": self.player.rect.width,
                "h": self.player.rect.height,
                "vx": round(self.player.vx, 2),
                "vy": round(self.player.vy, 2),
                "on_ground": self.player.on_ground,
            },
            "solids": [
                {"x": block.x, "y": block.y, "w": block.width, "h": block.height}
                for block in self.level.solids
            ],
            "coins": [
                {
                    "x": coin.rect.x,
                    "y": coin.rect.y,
                    "w": coin.rect.width,
                    "h": coin.rect.height,
                    "collected": coin.collected,
                }
                for coin in self.level.coins
            ],
            "goombas": [
                {
                    "x": goomba.rect.x,
                    "y": goomba.rect.y,
                    "w": goomba.rect.width,
                    "h": goomba.rect.height,
                    "vx": round(goomba.vx, 2),
                    "vy": round(goomba.vy, 2),
                    "alive": goomba.alive,
                }
                for goomba in self.level.goombas
            ],
            "goalpost": {
                "x": self.level.goalpost.x,
                "y": self.level.goalpost.y,
                "w": self.level.goalpost.width,
                "h": self.level.goalpost.height,
            },
        }

    def draw(self, screen, fonts):
        screen.fill(COLOR_BG)

        for block in self.level.solids:
            color = COLOR_PLATFORM if block.height < GROUND_HEIGHT else COLOR_GROUND
            pygame.draw.rect(screen, color, block)

        for coin in self.level.coins:
            if not coin.collected:
                pygame.draw.circle(screen, COLOR_COIN, coin.rect.center, coin.rect.width // 2)

        for goomba in self.level.goombas:
            if goomba.alive:
                pygame.draw.rect(screen, COLOR_GOOMBA, goomba.rect)

        pygame.draw.rect(screen, COLOR_GOAL, self.level.goalpost)
        pygame.draw.rect(screen, COLOR_PLAYER, self.player.rect)

        hud = fonts["small"].render(
            f"Score: {self.score}  Lives: {self.lives}", True, COLOR_TEXT
        )
        screen.blit(hud, (10, 10))

        if self.state == "won":
            message = fonts["large"].render("You Win! Press R to Restart", True, COLOR_TEXT)
            screen.blit(
                message,
                message.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)),
            )
        elif self.state == "game_over":
            message = fonts["large"].render("Game Over! Press R to Restart", True, COLOR_TEXT)
            screen.blit(
                message,
                message.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)),
            )
