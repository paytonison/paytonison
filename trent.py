# super_mario_clone.py  ── pygame ≥ 2.2
"""
Controls
--------
←/→      walk
SPACE/↑  jump
ESC      quit
"""

import os, sys
import pygame as pg
from dataclasses import dataclass

# ───────── CONFIG ─────────
SCREEN_W, SCREEN_H = 960, 480
FPS = 60
TILE = 32  # 1 tile = 32 px
GRAVITY = 0.5
JUMP_V = -11
RUN_SPEED = 4.5

COIN_VALUE = 100
ENEMY_PENALTY = 100  # if enemy hits you
START_LIVES = 3
START_SCORE = 0

LEVEL = [
    "                                                                                                          F",
    "                                                                                                           ",
    "                                                                                                           ",
    "                              C                                                                            ",
    "                                                                                                           ",
    "                  ######                                                                                   ",
    "                                                                                                           ",
    "        C                                                                                                  ",
    "                          G                                                                                ",
    "#############################      ###########################      ###############################        ",
]

# Map legend → collision / entity factory will look at these
SOLID_TILES = {"#"}
COIN_CHR = "C"
ENEMY_CHR = "G"
FINISH_CHR = "F"

ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")


# ───────── HELPERS ─────────
def load_image(name, fallback_color, size):
    """Load image or return simple colored Surface of given size."""
    path = os.path.join(ASSET_DIR, name)
    surf = pg.Surface(size, pg.SRCALPHA)
    try:
        img = pg.image.load(path).convert_alpha()
        return pg.transform.scale(img, size)
    except FileNotFoundError:
        surf.fill(fallback_color)
        return surf


# ───────── SPRITES ─────────
@dataclass
class Entity:
    image: pg.Surface
    rect: pg.Rect
    vx: float = 0
    vy: float = 0
    alive: bool = True

    def draw(self, screen, cam_x):
        screen.blit(self.image, (self.rect.x - cam_x, self.rect.y))


# ───────── GAME ─────────
class MarioGame:
    def __init__(self):
        pg.init()
        self.scr = pg.display.set_mode((SCREEN_W, SCREEN_H))
        pg.display.set_caption("Mario-esque clone")
        self.clock = pg.time.Clock()
        self.font = pg.font.SysFont(None, 28)

        self.load_assets()
        self.reset()

    def load_assets(self):
        self.img_player = load_image("mario.png", (255, 0, 0), (TILE, int(TILE * 1.5)))
        self.img_coin = load_image("coin.png", (255, 213, 0), (TILE, TILE))
        self.img_enemy = load_image("goomba.png", (139, 69, 19), (TILE, TILE))
        self.img_brick = load_image("brick.png", (174, 105, 6), (TILE, TILE))
        self.img_flag = load_image("flag.png", (30, 197, 48), (TILE, TILE * 2))

    # ────── WORLD BUILD ──────
    def build_level(self):
        self.solids, self.coins, self.enemies = [], [], []
        self.finish = None

        offset_y = SCREEN_H - len(LEVEL) * TILE  # NEW: push map to bottom edge

        for row_idx, line in enumerate(LEVEL):
            for col_idx, ch in enumerate(line):
                x = col_idx * TILE
                y = offset_y + row_idx * TILE  # NEW: apply vertical offset
                if ch in SOLID_TILES:
                    self.solids.append(pg.Rect(x, y, TILE, TILE))
                elif ch == COIN_CHR:
                    self.coins.append(Entity(self.img_coin, pg.Rect(x, y, TILE, TILE)))
                elif ch == ENEMY_CHR:
                    enemy = Entity(self.img_enemy, pg.Rect(x, y, TILE, TILE))
                    enemy.vx = -1.2
                    self.enemies.append(enemy)
                elif ch == FINISH_CHR:
                    self.finish = Entity(
                        self.img_flag, pg.Rect(x, y - TILE, TILE, TILE * 2)
                    )

    # ────── STATE ──────
    def reset(self):
        self.build_level()
        spawn_x = 64
        spawn_y = SCREEN_H - TILE * 3
        self.player = Entity(
            self.img_player, pg.Rect(spawn_x, spawn_y, TILE, int(TILE * 1.5))
        )
        self.score = START_SCORE
        self.lives = START_LIVES
        self.camera = 0
        self.game_over = False
        self.level_complete = False

    # ────── MAIN LOOP ──────
    def run(self):
        while True:
            dt = self.clock.tick(FPS) / 16.666  # normalise to 60 fps steps
            self.handle_events()
            if not self.game_over and not self.level_complete:
                self.update(dt)
            self.draw()

    # ────── INPUT ──────
    def handle_events(self):
        for e in pg.event.get():
            if e.type == pg.QUIT:
                pg.quit()
                sys.exit()
            elif e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE:
                pg.quit()
                sys.exit()
            elif e.type == pg.KEYDOWN and (e.key in (pg.K_SPACE, pg.K_UP)):
                if self.on_ground(self.player):
                    self.player.vy = JUMP_V

    # ────── UPDATE ──────
    def update(self, dt):
        keys = pg.key.get_pressed()
        self.player.vx = (keys[pg.K_RIGHT] - keys[pg.K_LEFT]) * RUN_SPEED

        # Horizontal move + collide
        self.player.rect.x += self.player.vx
        self.collide_axis(self.player, axis=0)

        # Apply gravity
        self.player.vy += GRAVITY
        self.player.rect.y += self.player.vy
        self.collide_axis(self.player, axis=1)

        # Camera follows
        self.camera = max(0, self.player.rect.centerx - SCREEN_W // 3)

        # ===== ENEMIES =====
        for en in self.enemies[:]:
            en.rect.x += en.vx
            # simple bounce off bricks
            for solid in self.solids:
                if en.rect.colliderect(solid):
                    en.vx *= -1
                    en.rect.x += en.vx * 2
            # kill if off-screen far left
            if en.rect.right < 0:
                self.enemies.remove(en)

        # ===== COLLISIONS =====
        # coins
        for coin in self.coins[:]:
            if self.player.rect.colliderect(coin.rect):
                self.coins.remove(coin)
                self.score += COIN_VALUE

        # enemies
        for en in self.enemies[:]:
            if self.player.rect.colliderect(en.rect):
                if (
                    self.player.vy > 0
                    and self.player.rect.bottom - en.rect.top < TILE // 2
                ):
                    # stomp!
                    self.enemies.remove(en)
                    self.player.vy = JUMP_V * 0.6
                    self.score += COIN_VALUE  # reward for stomp
                else:
                    # got hit
                    self.score = max(0, self.score - ENEMY_PENALTY)
                    self.lives -= 1
                    # knockback & brief invuln: simple respawn
                    self.player.rect.topleft = (64, SCREEN_H - TILE * 3)
                    if self.lives <= 0:
                        self.game_over = True
                    break

        # finish line
        if self.finish and self.player.rect.colliderect(self.finish.rect):
            self.level_complete = True

    # ────── COLLISION RESOLUTION ──────
    def collide_axis(self, ent, axis):
        collidables = self.solids
        for solid in collidables:
            if ent.rect.colliderect(solid):
                if axis == 0:  # x resolve
                    if ent.vx > 0:
                        ent.rect.right = solid.left
                    elif ent.vx < 0:
                        ent.rect.left = solid.right
                    ent.vx = 0
                else:  # y resolve
                    if ent.vy > 0:
                        ent.rect.bottom = solid.top
                        ent.vy = 0
                    elif ent.vy < 0:
                        ent.rect.top = solid.bottom
                        ent.vy = 0

    def on_ground(self, ent):
        test_rect = ent.rect.move(0, 1)
        return any(test_rect.colliderect(s) for s in self.solids)

    # ────── RENDER ──────
    def draw(self):
        self.scr.fill((92, 148, 252))  # sky
        cam = self.camera

        # draw bricks
        for solid in self.solids:
            self.scr.blit(self.img_brick, (solid.x - cam, solid.y))

        # flag
        if self.finish:
            self.finish.draw(self.scr, cam)

        # entities
        for coin in self.coins:
            coin.draw(self.scr, cam)
        for en in self.enemies:
            en.draw(self.scr, cam)
        self.player.draw(self.scr, cam)

        # UI
        txt = f"SCORE {self.score:>6}   LIVES {self.lives}"
        surf = self.font.render(txt, True, (255, 255, 255))
        self.scr.blit(surf, (10, 10))

        if self.game_over:
            self.center_message("GAME OVER – press ESC")
        elif self.level_complete:
            self.center_message("YOU WIN! press ESC")

        pg.display.flip()

    def center_message(self, msg):
        surf = self.font.render(msg, True, (0, 0, 0))
        rect = surf.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2))
        self.scr.blit(surf, rect)


# ───────── ENTRY ─────────
if __name__ == "__main__":
    MarioGame().run()
