use macroquad::prelude::*;

const TILE_SIZE: f32 = 32.0;
const PLAYER_SIZE: Vec2 = Vec2::new(24.0, 30.0);
const GOOMBA_SIZE: Vec2 = Vec2::new(24.0, 22.0);
const COIN_SIZE: f32 = 12.0;

const GRAVITY: f32 = 900.0;
const MOVE_SPEED: f32 = 200.0;
const JUMP_SPEED: f32 = 380.0;
const STOMP_BOUNCE: f32 = 260.0;
const GOOMBA_SPEED: f32 = 60.0;
const INVULN_TIME: f32 = 1.2;
const FALL_LIMIT_PADDING: f32 = 200.0;

const LEVEL: &str = r#"
........................................................................
........................................................................
....C..................G.....................C.......................F..
..................#####..................#####.........................
.............G..........................................................
..P..............................C..............................#####...
...............#####....................................................
...........................G..................C.........................
...........#####.................................#####..................
........................................................................
.............................#####......................................
........................................................................
########################################################################
"#;

#[derive(Clone)]
struct Goomba {
    pos: Vec2,
    vel: Vec2,
    dir: f32,
    alive: bool,
}

#[derive(Clone)]
struct Level {
    solids: Vec<Rect>,
    coins: Vec<Vec2>,
    goombas: Vec<Goomba>,
    goal: Rect,
    spawn: Vec2,
    width: usize,
    height: usize,
}

struct Player {
    pos: Vec2,
    vel: Vec2,
    on_ground: bool,
    lives: i32,
    score: i32,
    coins: i32,
    invuln: f32,
}

enum GameState {
    Playing,
    Won,
    GameOver,
}

struct Game {
    level: Level,
    player: Player,
    coins: Vec<Vec2>,
    goombas: Vec<Goomba>,
    state: GameState,
}

#[macroquad::main("Mario Rust")]
async fn main() {
    let base_level = parse_level();
    let mut game = Game::new(base_level.clone());

    loop {
        let dt = get_frame_time().min(0.05);

        if matches!(game.state, GameState::Playing) {
            game.update(dt);
        } else if is_key_pressed(KeyCode::R) {
            game = Game::new(base_level.clone());
        }

        game.draw();
        next_frame().await;
    }
}

impl Game {
    fn new(level: Level) -> Self {
        let player = Player {
            pos: level.spawn,
            vel: Vec2::ZERO,
            on_ground: false,
            lives: 3,
            score: 0,
            coins: 0,
            invuln: 0.0,
        };

        Self {
            coins: level.coins.clone(),
            goombas: level.goombas.clone(),
            level,
            player,
            state: GameState::Playing,
        }
    }

    fn update(&mut self, dt: f32) {
        self.handle_input();
        self.apply_physics(dt);
        self.update_goombas(dt);
        self.collect_coins();
        self.check_goal();
        self.check_enemy_collisions();
        self.check_fall_off();

        if self.player.invuln > 0.0 {
            self.player.invuln = (self.player.invuln - dt).max(0.0);
        }
    }

    fn handle_input(&mut self) {
        let mut input = 0.0;
        if is_key_down(KeyCode::Left) || is_key_down(KeyCode::A) {
            input -= 1.0;
        }
        if is_key_down(KeyCode::Right) || is_key_down(KeyCode::D) {
            input += 1.0;
        }

        self.player.vel.x = input * MOVE_SPEED;

        if self.player.on_ground
            && (is_key_pressed(KeyCode::Space)
                || is_key_pressed(KeyCode::Up)
                || is_key_pressed(KeyCode::W))
        {
            self.player.vel.y = -JUMP_SPEED;
            self.player.on_ground = false;
        }
    }

    fn apply_physics(&mut self, dt: f32) {
        self.player.vel.y += GRAVITY * dt;

        let (pos, vel, on_ground) = move_with_collisions(
            self.player.pos,
            PLAYER_SIZE,
            self.player.vel,
            &self.level.solids,
            dt,
        );

        self.player.pos = pos;
        self.player.vel = vel;
        self.player.on_ground = on_ground;
    }

    fn update_goombas(&mut self, dt: f32) {
        for goomba in &mut self.goombas {
            if !goomba.alive {
                continue;
            }

            goomba.vel.y += GRAVITY * dt;
            goomba.vel.x = GOOMBA_SPEED * goomba.dir;

            let (pos, vel, _on_ground) = move_with_collisions(
                goomba.pos,
                GOOMBA_SIZE,
                goomba.vel,
                &self.level.solids,
                dt,
            );

            let hit_wall = goomba.vel.x.abs() > 0.0 && vel.x.abs() <= 0.01;
            if hit_wall {
                goomba.dir *= -1.0;
            }

            goomba.pos = pos;
            goomba.vel = vel;
        }
    }

    fn collect_coins(&mut self) {
        let player_rect = rect_at(self.player.pos, PLAYER_SIZE);
        let before_bonus = self.player.coins / 10;
        let mut collected = 0;

        self.coins.retain(|coin| {
            let coin_rect = rect_at(*coin, Vec2::splat(COIN_SIZE));
            let hit = player_rect.overlaps(&coin_rect);
            if hit {
                collected += 1;
            }
            !hit
        });

        if collected > 0 {
            self.player.coins += collected;
            self.player.score += collected * 100;
            let after_bonus = self.player.coins / 10;
            if after_bonus > before_bonus {
                self.player.lives += after_bonus - before_bonus;
            }
        }
    }

    fn check_goal(&mut self) {
        let player_rect = rect_at(self.player.pos, PLAYER_SIZE);
        if player_rect.overlaps(&self.level.goal) {
            self.state = GameState::Won;
        }
    }

    fn check_enemy_collisions(&mut self) {
        let mut stomped = Vec::new();
        let player_rect = rect_at(self.player.pos, PLAYER_SIZE);

        for (idx, goomba) in self.goombas.iter_mut().enumerate() {
            if !goomba.alive {
                continue;
            }

            let goomba_rect = rect_at(goomba.pos, GOOMBA_SIZE);
            if !player_rect.overlaps(&goomba_rect) {
                continue;
            }

            let player_bottom = self.player.pos.y + PLAYER_SIZE.y;
            let stomp_window = goomba.pos.y + 4.0;

            if self.player.vel.y > 0.0 && player_bottom <= stomp_window {
                stomped.push(idx);
                self.player.vel.y = -STOMP_BOUNCE;
                self.player.score += 200;
            } else {
                self.hurt_player();
                return;
            }
        }

        for idx in stomped {
            if let Some(goomba) = self.goombas.get_mut(idx) {
                goomba.alive = false;
            }
        }
    }

    fn check_fall_off(&mut self) {
        let world_h = self.level.height as f32 * TILE_SIZE;
        if self.player.pos.y > world_h + FALL_LIMIT_PADDING {
            self.hurt_player();
        }
    }

    fn hurt_player(&mut self) {
        if self.player.invuln > 0.0 {
            return;
        }

        self.player.lives -= 1;
        if self.player.lives <= 0 {
            self.state = GameState::GameOver;
            return;
        }

        self.player.pos = self.level.spawn;
        self.player.vel = Vec2::ZERO;
        self.player.invuln = INVULN_TIME;
        self.player.on_ground = false;
    }

    fn draw(&self) {
        clear_background(Color::new(0.45, 0.75, 0.95, 1.0));

        let camera = make_camera(&self.level, self.player.pos + PLAYER_SIZE / 2.0);
        set_camera(&camera);

        for solid in &self.level.solids {
            draw_rectangle(solid.x, solid.y, solid.w, solid.h, Color::new(0.25, 0.55, 0.25, 1.0));
        }

        for coin in &self.coins {
            draw_circle(
                coin.x + COIN_SIZE / 2.0,
                coin.y + COIN_SIZE / 2.0,
                COIN_SIZE / 2.0,
                Color::new(0.95, 0.8, 0.2, 1.0),
            );
        }

        for goomba in &self.goombas {
            if !goomba.alive {
                continue;
            }
            draw_rectangle(
                goomba.pos.x,
                goomba.pos.y,
                GOOMBA_SIZE.x,
                GOOMBA_SIZE.y,
                Color::new(0.55, 0.35, 0.2, 1.0),
            );
        }

        draw_goal(&self.level.goal);

        let player_color = if self.player.invuln > 0.0 && (self.player.invuln * 12.0) as i32 % 2 == 0 {
            Color::new(1.0, 0.9, 0.9, 1.0)
        } else {
            Color::new(0.8, 0.15, 0.15, 1.0)
        };

        draw_rectangle(
            self.player.pos.x,
            self.player.pos.y,
            PLAYER_SIZE.x,
            PLAYER_SIZE.y,
            player_color,
        );

        set_default_camera();
        draw_ui(self);
    }
}

fn parse_level() -> Level {
    let mut solids = Vec::new();
    let mut coins = Vec::new();
    let mut goombas = Vec::new();
    let mut spawn = vec2(TILE_SIZE, TILE_SIZE * 2.0);
    let mut goal = Rect::new(0.0, 0.0, TILE_SIZE, TILE_SIZE * 2.0);
    let mut width = 0;
    let mut height = 0;

    for (y, line) in LEVEL.trim().lines().enumerate() {
        height = height.max(y + 1);
        width = width.max(line.chars().count());

        for (x, ch) in line.chars().enumerate() {
            let world_x = x as f32 * TILE_SIZE;
            let world_y = y as f32 * TILE_SIZE;

            match ch {
                '#' => solids.push(Rect::new(world_x, world_y, TILE_SIZE, TILE_SIZE)),
                'C' => {
                    let offset = (TILE_SIZE - COIN_SIZE) / 2.0;
                    coins.push(vec2(world_x + offset, world_y + offset));
                }
                'G' => goombas.push(Goomba {
                    pos: vec2(world_x, world_y + (TILE_SIZE - GOOMBA_SIZE.y)),
                    vel: Vec2::ZERO,
                    dir: -1.0,
                    alive: true,
                }),
                'P' => spawn = vec2(world_x, world_y + (TILE_SIZE - PLAYER_SIZE.y)),
                'F' => {
                    let goal_y = if world_y >= TILE_SIZE { world_y - TILE_SIZE } else { 0.0 };
                    goal = Rect::new(world_x, goal_y, TILE_SIZE, TILE_SIZE * 2.0);
                }
                _ => {}
            }
        }
    }

    Level {
        solids,
        coins,
        goombas,
        goal,
        spawn,
        width,
        height,
    }
}

fn move_with_collisions(
    pos: Vec2,
    size: Vec2,
    vel: Vec2,
    solids: &[Rect],
    dt: f32,
) -> (Vec2, Vec2, bool) {
    let mut pos = pos;
    let mut vel = vel;
    let mut on_ground = false;

    pos.x += vel.x * dt;
    let mut rect = rect_at(pos, size);
    for solid in solids {
        if rect.overlaps(solid) {
            if vel.x > 0.0 {
                pos.x = solid.x - size.x;
            } else if vel.x < 0.0 {
                pos.x = solid.x + solid.w;
            }
            vel.x = 0.0;
            rect.x = pos.x;
        }
    }

    pos.y += vel.y * dt;
    rect.y = pos.y;
    for solid in solids {
        if rect.overlaps(solid) {
            if vel.y > 0.0 {
                pos.y = solid.y - size.y;
                on_ground = true;
            } else if vel.y < 0.0 {
                pos.y = solid.y + solid.h;
            }
            vel.y = 0.0;
            rect.y = pos.y;
        }
    }

    (pos, vel, on_ground)
}

fn rect_at(pos: Vec2, size: Vec2) -> Rect {
    Rect::new(pos.x, pos.y, size.x, size.y)
}

fn make_camera(level: &Level, focus: Vec2) -> Camera2D {
    let world_w = level.width as f32 * TILE_SIZE;
    let world_h = level.height as f32 * TILE_SIZE;
    let screen_w = screen_width();
    let screen_h = screen_height();

    let mut cam_x = focus.x;
    let mut cam_y = focus.y;

    if world_w > screen_w {
        cam_x = cam_x.clamp(screen_w * 0.5, world_w - screen_w * 0.5);
    } else {
        cam_x = world_w * 0.5;
    }

    if world_h > screen_h {
        cam_y = cam_y.clamp(screen_h * 0.5, world_h - screen_h * 0.5);
    } else {
        cam_y = world_h * 0.5;
    }

    Camera2D {
        target: vec2(cam_x, cam_y),
        zoom: vec2(2.0 / screen_w, -2.0 / screen_h),
        ..Default::default()
    }
}

fn draw_goal(goal: &Rect) {
    draw_rectangle(goal.x + goal.w * 0.45, goal.y, goal.w * 0.1, goal.h, GRAY);
    draw_rectangle(goal.x + goal.w * 0.1, goal.y, goal.w * 0.4, goal.h * 0.3, RED);
}

fn draw_ui(game: &Game) {
    let state_text = match game.state {
        GameState::Playing => "",
        GameState::Won => "YOU WIN! Press R to restart",
        GameState::GameOver => "GAME OVER! Press R to restart",
    };

    draw_text(
        &format!(
            "Lives: {}   Coins: {}   Score: {}",
            game.player.lives, game.player.coins, game.player.score
        ),
        16.0,
        28.0,
        24.0,
        BLACK,
    );

    if !state_text.is_empty() {
        let text_dim = measure_text(state_text, None, 36, 1.0);
        draw_text(
            state_text,
            (screen_width() - text_dim.width) * 0.5,
            70.0,
            36.0,
            BLACK,
        );
    }
}
