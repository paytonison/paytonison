# mario_clone.py
import random
import sys
from dataclasses import dataclass
import pygame as pg

# ---------- CONSTANTS ----------
WIDTH, HEIGHT = 800, 400
GROUND_Y = HEIGHT - 60
FPS = 60

PLAYER_WIDTH, PLAYER_HEIGHT = 40, 55
PLAYER_SPEED = 5
JUMP_VELOCITY = -12
GRAVITY = 0.6

COIN_SIZE = 20
ENEMY_SIZE = 30
SPAWN_EVENT = pg.USEREVENT + 1  # fires every 1.2 s

START_LIVES = 3
START_SCORE = 100  # tweak to taste
POINT_VALUE = 100  # +/- for coin / enemy


# ---------- DATA CLASSES ----------
@dataclass
class Sprite:
    rect: pg.Rect
    vy: float = 0  # vertical velocity (only for player)

    def draw(self, surface, color):
        pg.draw.rect(surface, color, self.rect)


# ---------- GAME STATE ----------
class Game:
    def __init__(self):
        pg.init()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption("Tiny-Mario")
        self.clock = pg.time.Clock()

        self.font = pg.font.SysFont(None, 30)

        self.player = Sprite(
            pg.Rect(100, GROUND_Y - PLAYER_HEIGHT, PLAYER_WIDTH, PLAYER_HEIGHT)
        )
        self.coins: list[Sprite] = []
        self.enemies: list[Sprite] = []

        self.lives = START_LIVES
        self.score = START_SCORE
        self.running = True

        pg.time.set_timer(SPAWN_EVENT, 1200)

    # ---------- SPAWN HELPERS ----------
    def spawn_coin(self):
        y = random.randint(GROUND_Y - 120, GROUND_Y - COIN_SIZE)
        rect = pg.Rect(WIDTH, y, COIN_SIZE, COIN_SIZE)
        self.coins.append(Sprite(rect))

    def spawn_enemy(self):
        rect = pg.Rect(WIDTH, GROUND_Y - ENEMY_SIZE, ENEMY_SIZE, ENEMY_SIZE)
        self.enemies.append(Sprite(rect))

    def spawn_left_enemy(self):
        rect = pg.Rect(0 - ENEMY_SIZE, GROUND_Y - ENEMY_SIZE, ENEMY_SIZE, ENEMY_SIZE)
        # Mark this enemy as coming from the left by adding an attribute
        sprite = Sprite(rect)
        sprite.from_left = True
        self.enemies.append(sprite)

    # ---------- MAIN LOOP ----------
    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)

        pg.quit()
        sys.exit()

    # ---------- INPUT ----------
    def handle_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
            elif event.type == SPAWN_EVENT:
                # 70 % coins, 30 % enemies
                if random.random() < 0.7:
                    self.spawn_coin()
                else:
                    self.spawn_enemy()

        keys = pg.key.get_pressed()
        if keys[pg.K_LEFT]:
            self.player.rect.x -= PLAYER_SPEED
        if keys[pg.K_RIGHT]:
            self.player.rect.x += PLAYER_SPEED
        if keys[pg.K_SPACE] or keys[pg.K_UP]:
            # jump if on ground
            if self.player.rect.bottom >= GROUND_Y:
                self.player.vy = JUMP_VELOCITY

    # ---------- UPDATE ----------
    def update(self):
        # apply gravity & move
        self.player.vy += GRAVITY
        self.player.rect.y += self.player.vy
        if self.player.rect.bottom >= GROUND_Y:
            self.player.rect.bottom = GROUND_Y
            self.player.vy = 0

        # move & cull coins/enemies
        for collection in (self.coins, self.enemies):
            for sprite in collection:
                # Move left-spawned enemies right, others left
                if getattr(sprite, "from_left", False):
                    sprite.rect.x += PLAYER_SPEED
                else:
                    sprite.rect.x -= PLAYER_SPEED
            # keep sprites still on screen
            collection[:] = [
                s for s in collection if 0 < s.rect.right and s.rect.left < WIDTH
            ]

        # coin collisions
        for coin in self.coins[:]:
            if self.player.rect.colliderect(coin.rect):
                self.coins.remove(coin)
                self.score += POINT_VALUE

        # enemy collisions
        for enemy in self.enemies[:]:
            if self.player.rect.colliderect(enemy.rect):
                self.enemies.remove(enemy)
                penalty = 500 if getattr(enemy, "from_left", False) else POINT_VALUE
                self.score -= penalty
                if self.score <= 0:
                    self.lives -= 1
                    if self.lives == 0:
                        self.running = False  # game over
                        return
                    self.score = START_SCORE

    def draw(self):
        self.screen.fill((92, 148, 252))  # sky blue
        pg.draw.rect(
            self.screen,
            (106, 170, 64),  # ground
            (0, GROUND_Y, WIDTH, HEIGHT - GROUND_Y),
        )

        # player, coins, enemies
        self.player.draw(self.screen, (255, 0, 0))  # red
        for coin in self.coins:
            coin.draw(self.screen, (255, 223, 0))  # gold
        for enemy in self.enemies:
            enemy.draw(self.screen, (0, 0, 0))  # black

        # UI
        ui = self.font.render(
            f"Score: {self.score}   Lives: {self.lives}", True, (255, 255, 255)
        )
        self.screen.blit(ui, (10, 10))

        pg.display.flip()


# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    Game().run()
