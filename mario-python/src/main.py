import argparse
import json
import math
import os
from array import array

import pygame

from agents import HeuristicAgent, KeyboardAgent, ResponsesAgent
from browser_client import ActionStore, BrowserAgent, StateStore, start_browser_server
from game import Game, SCREEN_HEIGHT, SCREEN_WIDTH


def parse_action(action):
    if not action:
        return False, False, False
    action = action.lower().strip()
    if action in {"noop", "none"}:
        return False, False, False
    tokens = action.replace("+", " ").replace(",", " ").split()
    left = "left" in tokens
    right = "right" in tokens
    jump = "jump" in tokens
    if left and right:
        left = right = False
    return left, right, jump


class AgentController:
    def __init__(self, agent, min_interval):
        self.agent = agent
        self.min_interval = min_interval
        self.last_time = 0.0
        self.last_action = "noop"

    def get_action(self, now, state):
        if self.min_interval <= 0 or now - self.last_time >= self.min_interval:
            self.last_action = self.agent.decide(state)
            self.last_time = now
        return self.last_action


def make_beep(freq_hz, duration_s, volume=0.4, sample_rate=44100):
    length = int(sample_rate * duration_s)
    amplitude = int(32767 * max(0.0, min(volume, 1.0)))
    buf = array("h")
    for i in range(length):
        t = i / sample_rate
        value = int(amplitude * math.sin(2 * math.pi * freq_hz * t))
        buf.append(value)
    return pygame.mixer.Sound(buffer=buf.tobytes())


def init_audio():
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
    except pygame.error:
        return {}
    return {
        "coin": make_beep(880, 0.08, 0.35),
        "jump": make_beep(660, 0.06, 0.25),
        "stomp": make_beep(220, 0.08, 0.4),
        "hurt": make_beep(120, 0.14, 0.4),
        "win": make_beep(990, 0.2, 0.4),
    }


def build_agent(args):
    if args.agent == "keyboard":
        return KeyboardAgent()
    if args.agent == "heuristic":
        return HeuristicAgent()
    if args.agent == "responses":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("OPENAI_API_KEY is not set; using heuristic agent instead.")
            return HeuristicAgent()
        return ResponsesAgent(api_key, model=args.model)
    return KeyboardAgent()


def parse_args():
    parser = argparse.ArgumentParser(description="Mario clone with optional agent control.")
    parser.add_argument(
        "--agent",
        choices=["keyboard", "heuristic", "responses", "browser"],
        default="keyboard",
        help="Control mode for Mario.",
    )
    parser.add_argument(
        "--agent-rate",
        type=float,
        default=0.2,
        help="Seconds between agent decisions.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.2",
        help="Responses API model to use when --agent responses is selected.",
    )
    parser.add_argument(
        "--state-file",
        default="",
        help="Optional path to write the latest game state JSON.",
    )
    parser.add_argument(
        "--browser-port",
        type=int,
        default=8765,
        help="Port for the browser client when --agent browser is selected.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Frames per second for the game loop.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pygame.mixer.pre_init(44100, -16, 1, 512)
    pygame.init()
    pygame.display.set_caption("Mario Clone")

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    fonts = {
        "small": pygame.font.Font(None, 24),
        "large": pygame.font.Font(None, 36),
    }
    sfx = init_audio()
    game = Game(sfx=sfx)

    state_store = None
    server = None
    if args.agent == "browser":
        action_store = ActionStore()
        state_store = StateStore()
        try:
            server = start_browser_server(
                state_store, action_store, port=args.browser_port
            )
            print(f"Browser client available at http://127.0.0.1:{args.browser_port}/")
            agent = BrowserAgent(action_store)
        except OSError as exc:
            print(f"Failed to start browser client: {exc}. Falling back to keyboard.")
            agent = KeyboardAgent()
    else:
        agent = build_agent(args)

    agent_interval = 0 if args.agent in {"keyboard", "browser"} else args.agent_rate
    controller = AgentController(agent, agent_interval)
    state_dump_interval = max(args.agent_rate, 0.2) if args.state_file else 0
    last_state_dump = 0.0

    clock = pygame.time.Clock()
    running = True
    while running:
        now = pygame.time.get_ticks() / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r and game.state != "playing":
                    game = Game(sfx=sfx)

        state_before = game.get_state()
        action_str = controller.get_action(now, state_before)
        action = parse_action(action_str)
        game.update(action)
        state_after = game.get_state()

        if state_store:
            state_store.set(state_after)

        if args.state_file and now - last_state_dump >= state_dump_interval:
            with open(args.state_file, "w", encoding="utf-8") as handle:
                json.dump(state_after, handle)
            last_state_dump = now

        game.draw(screen, fonts)
        pygame.display.flip()
        clock.tick(args.fps)

    if server:
        server.shutdown()
        server.server_close()
    pygame.quit()


if __name__ == "__main__":
    main()
