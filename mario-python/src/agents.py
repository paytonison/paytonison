import json
import time
import urllib.error
import urllib.request

import pygame


class Agent:
    def decide(self, state):
        return "noop"


class KeyboardAgent(Agent):
    def decide(self, state):
        keys = pygame.key.get_pressed()
        parts = []
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            parts.append("left")
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            parts.append("right")
        if keys[pygame.K_SPACE] or keys[pygame.K_w] or keys[pygame.K_UP]:
            parts.append("jump")
        return "+".join(parts) if parts else "noop"


def _rect_collides(a, b):
    return not (
        a["x"] + a["w"] <= b["x"]
        or a["x"] >= b["x"] + b["w"]
        or a["y"] + a["h"] <= b["y"]
        or a["y"] >= b["y"] + b["h"]
    )


class HeuristicAgent(Agent):
    def __init__(self, jump_lookahead=40):
        self.jump_lookahead = jump_lookahead

    def decide(self, state):
        player = state["player"]
        solids = state.get("solids", [])
        goombas = [g for g in state.get("goombas", []) if g.get("alive")]

        if not player["on_ground"]:
            return "right"

        if self._gap_ahead(player, solids) or self._goomba_ahead(player, goombas):
            return "right+jump"

        return "right"

    def _gap_ahead(self, player, solids):
        foot = {
            "x": player["x"] + player["w"] + self.jump_lookahead,
            "y": player["y"] + player["h"] + 4,
            "w": 4,
            "h": 4,
        }
        return not any(_rect_collides(foot, block) for block in solids)

    def _goomba_ahead(self, player, goombas):
        for goomba in goombas:
            dx = goomba["x"] - player["x"]
            same_level = abs(goomba["y"] - player["y"]) < 20
            if 0 < dx < 60 and same_level:
                return True
        return False


class ResponsesAgent(Agent):
    def __init__(self, api_key, model="gpt-5.2", timeout=8):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.endpoint = "https://api.openai.com/v1/responses"
        self.system_prompt = (
            "You control Mario in a 2D platformer. "
            "Reply with one action only: left, right, jump, left+jump, right+jump, or noop."
        )
        self._last_error = 0.0

    def decide(self, state):
        payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": json.dumps(state)},
            ],
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read()
            result = json.loads(body.decode("utf-8"))
            action = self._extract_action(result)
            return action.strip().lower() if action else "noop"
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
            now = time.time()
            if now - self._last_error > 5:
                print("Responses API error; falling back to noop temporarily.")
                self._last_error = now
            return "noop"

    def _extract_action(self, result):
        if isinstance(result, dict):
            output_text = result.get("output_text")
            if output_text:
                return output_text
            for item in result.get("output", []):
                for content in item.get("content", []):
                    text = content.get("text")
                    if text:
                        return text
        return ""
