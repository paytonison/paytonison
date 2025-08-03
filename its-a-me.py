"""
Controls
--------
←/→      walk
SPACE/↑  jump
ESC      quit
"""

import os
import sys
import pygame as pg
from dataclasses import dataclass
from typing import List

# ───────── CONFIG ─────────
SCREEN_W, SCREEN_H = 960, 480
FPS = 60
TILE = 32
GRAVITY = 0.5
JUMP_V = -11
RUN_SPEED = 4.5

COIN_VALUE = 100
ENEMY_PENALTY = 100
START_LIVES = 3
START_SCORE = 0

LEVEL = [
    "                                                           F",
    "                                                           F",
    "                  #####                                    F",
    "                             C                             F",
    "        C                      C                           F",
    "              ##########            C      C               F",
    "                      G         G        G     C    ####   F",
    "############################  #### ######### ############  F",
]

SOLID_TILES = {"#"}
COIN_CHR = "C"
ENEMY_CHR = "G"
FINISH_CHR = "F"

ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")


# ───────── HELPERS ─────────
def load_image(name: str, fallback_color: tuple[int, int, int], size: tuple[int, int]) -> pg.Surface:
    """Load image or return simple colored Surface of given size."""
    path = os.path.join(ASSET_DIR, name)
    surf = pg.Surface(size, pg.SRCALPHA)
    if os.path.isfile(path):
        try:
            img = pg.image.load(path).convert_alpha()
            return pg.transform.scale(img, size)
        except pg.error:
            pass
    surf.fill(fallback_color)
    return surf


# ───────── DATA CLASSES ─────────
@dataclass(slots=True)
class Entity:
    image: pg.Surface
    rect: pg.Rect
    vx: float = 0.0
    vy: float = 0.0

    def draw(self, screen: pg.Surface, cam_x: int):
        screen.blit(self.image, (self.rect.x - cam_x, self.rect.y))


# ───────── GAME CLASS ─────────
class MarioGame:
    def __init__(self):
        pg.init()
        self.scr = pg.display.set_mode((SCREEN_W, SCREEN_H))
        pg.display.set_caption("Mario-esque clone")
        self.clock = pg.time.Clock()
        self.font = pg.font.SysFont(None, 28)

        self.load_assets()
        self.reset()

    # ────── ASSETS ──────
    def load_assets(self):
        self.img_player = load_image("mario.png", (255, 0, 0), (TILE, int(TILE * 1.5)))
        self.img_coin = load_image("coin.png", (255, 213, 0), (TILE, TILE))
        # Create simple brown square for enemies
        self.img_enemy = pg.Surface((TILE, TILE))
        self.img_enemy.fill((139, 69, 19))
        self.img_brick = load_image("brick.png", (174, 105, 6), (TILE, TILE))
        self.img_flag = load_image("flag.png", (30, 197, 48), (TILE, TILE * 2))

    # ────── WORLD BUILD ──────
    def build_level(self):
        self.solids: List[pg.Rect] = []
        self.coins: List[Entity] = []
        self.enemies: List[Entity] = []
        self.finish: Entity | None = None

        offset_y = SCREEN_H - len(LEVEL) * TILE

        for row_idx, line in enumerate(LEVEL):
            for col_idx, ch in enumerate(line):
                x = col_idx * TILE
                y = offset_y + row_idx * TILE
                if ch in SOLID_TILES:
                    self.solids.append(pg.Rect(x, y, TILE, TILE))
                elif ch == COIN_CHR:
                    self.coins.append(Entity(self.img_coin, pg.Rect(x, y, TILE, TILE)))
                elif ch == ENEMY_CHR:
                    enemy = Entity(self.img_enemy, pg.Rect(x, y, TILE, TILE), vx=-1.2)
                    self.enemies.append(enemy)
                elif ch == FINISH_CHR:
                    self.finish = Entity(
                            self.img_flag, pg.Rect(x, y - TILE, TILE, TILE * 2)
                            )
        # Setup flagpole rect for win detection
        if self.finish:
            pole_x = self.finish.rect.x + TILE // 2
            self.pole_rect = pg.Rect(pole_x - 2, 0, 4, SCREEN_H)
        self.build_background()
    # ────── BACKGROUND ──────
    def build_background(self):
        width = len(LEVEL[0]) * TILE
        self.level_surface = pg.Surface((width, SCREEN_H))
        self.level_surface.fill((92, 148, 252))
        for solid in self.solids:
            self.level_surface.blit(self.img_brick, (solid.x, solid.y))
        if self.finish:
            # Draw flagpole as vertical pole spanning full height
            pole_x = self.finish.rect.x + TILE // 2
            pole_width = 4
            pg.draw.rect(self.level_surface, (150, 150, 150), (pole_x - pole_width//2, 0, pole_width, SCREEN_H))
            # Draw flag at top of pole
            self.level_surface.blit(self.img_flag, (self.finish.rect.x, self.finish.rect.y))
        self.camera = 0
        self.game_over = False
        self.level_complete = False

    # ────── MAIN LOOP ──────
    def run(self):
        while True:
            dt = self.clock.tick(FPS) / 16.666  # scaled to 60 FPS base
            self.handle_events()
            if not self.game_over and not self.level_complete:
                self.update(dt)
            self.draw()

    # ────── INPUT ──────
    def handle_events(self):
        for e in pg.event.get():
            if e.type == pg.QUIT:
                self.quit()
            elif e.type == pg.KEYDOWN:
                if e.key == pg.K_ESCAPE:
                    self.quit()
                elif e.key in (pg.K_SPACE, pg.K_UP) and self.on_ground(self.player):
                    self.player.vy = JUMP_V

    # ────── UPDATE ──────
    def update(self, dt: float):
        keys = pg.key.get_pressed()
        self.player.vx = (keys[pg.K_RIGHT] - keys[pg.K_LEFT]) * RUN_SPEED
        # Horizontal movement + collision
        move_x = self.player.vx * dt
        self.player.rect.x += int(round(move_x))
        self.collide_axis(self.player, axis=0)

        # Vertical movement + gravity + collision
        self.player.vy += GRAVITY * dt
        move_y = self.player.vy * dt
        self.player.rect.y += int(round(move_y))
        self.collide_axis(self.player, axis=1)

        # Camera update - follow player horizontally
        max_cam = self.level_surface.get_width() - SCREEN_W
        self.camera = max(0, min(max_cam, self.player.rect.centerx - SCREEN_W // 2))
        for en in self.enemies[:]:
            en.rect.x += int(round(en.vx * dt))
            # bounce off bricks
            for solid in self.solids:
                if en.rect.colliderect(solid):
                    en.vx *= -1
                    en.rect.x += int(round(en.vx * dt * 2))
            if en.rect.right < 0:  # off-screen
                self.enemies.remove(en)

        # Coin collection
        for coin in self.coins[:]:
            if self.player.rect.colliderect(coin.rect):
                self.coins.remove(coin)
                self.score += COIN_VALUE

        # Enemy collisions
        for en in self.enemies[:]:
            if self.player.rect.colliderect(en.rect):
                if (
                        self.player.vy > 0
                        and self.player.rect.bottom - en.rect.top < TILE // 2
                        ):
                    # stomp
                    self.enemies.remove(en)
                    self.player.vy = JUMP_V * 0.6
                    self.score += COIN_VALUE
                else:
                    self.handle_enemy_hit()
                    break

        # Fell below screen
        if self.player.rect.top > SCREEN_H:
            self.lives -= 1
            if self.lives <= 0:
                self.game_over = True
            else:
                self.build_level()
                self.player.rect.topleft = (64, SCREEN_H - TILE * 3)
                self.player.vx = self.player.vy = 0
                self.camera = 0

        # Finish line (flagpole)
        if hasattr(self, 'pole_rect') and self.player.rect.colliderect(self.pole_rect):
            # Award points based on how high the player is on the pole
            touch_y = self.player.rect.bottom
            # Higher touch (lower y) yields more points
            factor = max(0.0, min(1.0, (SCREEN_H - touch_y) / SCREEN_H))
            added = int(factor * 1000)
            self.score += added
            self.level_complete = True

    # ────── COLLISIONS ──────
    def collide_axis(self, ent: Entity, axis: int):
        for solid in self.solids:
            if ent.rect.colliderect(solid):
                if axis == 0:
                    if ent.vx > 0:
                        ent.rect.right = solid.left
                    elif ent.vx < 0:
                        ent.rect.left = solid.right
                    ent.vx = 0
                else:
                    if ent.vy > 0:
                        ent.rect.bottom = solid.top
                    elif ent.vy < 0:
                        ent.rect.top = solid.bottom
                    ent.vy = 0

    def on_ground(self, ent: Entity) -> bool:
        # Check if entity is on the ground
        rect = ent.rect.copy()
        rect.y += 1
        for solid in self.solids:
            if rect.colliderect(solid):
                return True
        return False
    def reset(self):
        """Reset game to its initial state."""
        self.score = START_SCORE
        self.lives = START_LIVES
        self.build_level()
        # Initialize player
        player_y = SCREEN_H - TILE * 3
        self.player = Entity(
            self.img_player,
            pg.Rect(64, player_y, TILE, int(TILE * 1.5))
        )
        self.player.vx = self.player.vy = 0
        self.camera = 0

    def draw(self):
        # Draw background
        self.scr.blit(self.level_surface, (-self.camera, 0))
        cam = self.camera

        # Draw entities
        for coin in self.coins:
            coin.draw(self.scr, cam)
        # Draw enemies
        for en in self.enemies:
            en.draw(self.scr, cam)
        self.player.draw(self.scr, cam)
        if self.finish:
            self.finish.draw(self.scr, cam)

        # UI
        ui_text = f"SCORE {self.score:>6}   LIVES {self.lives}"
        ui_surf = self.font.render(ui_text, True, (255, 255, 255))
        self.scr.blit(ui_surf, (10, 10))

        if self.game_over:
            self.center_message("GAME OVER – press ESC")
        elif self.level_complete:
            # Victory overlay and message
            overlay = pg.Surface((SCREEN_W, SCREEN_H), pg.SRCALPHA)
            overlay.fill((0, 0, 0, 180))  # semi-transparent black
            self.scr.blit(overlay, (0, 0))
            # Big victory text
            big_font = pg.font.SysFont(None, 72)
            win_surf = big_font.render("YOU WIN!", True, (255, 215, 0))
            win_rect = win_surf.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 - 20))
            self.scr.blit(win_surf, win_rect)
            # Prompt to exit
            small_surf = self.font.render("Press ESC to quit", True, (255, 255, 255))
            small_rect = small_surf.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 + 40))
            self.scr.blit(small_surf, small_rect)

        pg.display.flip()

    # ────── HELPERS ──────
    def handle_enemy_hit(self):
        """Lose one life and points when colliding with an enemy."""
        self.score = max(0, self.score - ENEMY_PENALTY)
        self.lives -= 1
        if self.lives <= 0:
            self.game_over = True
        else:
            self.build_level()
            self.player.rect.topleft = (64, SCREEN_H - TILE * 3)
            self.player.vx = self.player.vy = 0
            self.camera = 0

    def center_message(self, msg: str):
        surf = self.font.render(msg, True, (0, 0, 0))
        rect = surf.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2))
        self.scr.blit(surf, rect)

    def quit(self):
        pg.quit()
        sys.exit()


# ───────── ENTRY ─────────
if __name__ == "__main__":
    MarioGame().run()
