#!/usr/bin/env python3
import asyncio
import base64
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from agents import Agent, Runner, ComputerTool, ModelSettings
from agents.models.openai_responses import OpenAIResponsesModel
from agents.computer import AsyncComputer
from playwright.async_api import async_playwright, TimeoutError as PWTimeout
from openai import AsyncOpenAI
import argparse

# ---------- Helpers -----------------------------------------------------------

def default_chrome_user_data_root() -> str:
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Application Support/Google/Chrome")
    if sys.platform.startswith("win"):
        return os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Chrome", "User Data")
    # linux
    return os.path.expanduser("~/.config/google-chrome")

async def ensure_logged_in(page, login_timeout_sec: int) -> None:
    """Wait until the ChatGPT composer exists. If not, give user time to log in."""
    await page.goto("https://chatgpt.com/", wait_until="domcontentloaded")
    selectors = "form textarea, [contenteditable='true'], [data-testid*=composer]"
    try:
        await page.wait_for_selector(selectors, timeout=10_000)
        return  # already logged in
    except PWTimeout:
        print(f"Login required. Complete login in the opened Chrome window "
              f"(waiting up to {login_timeout_sec}s)…")
        await page.wait_for_selector(selectors, timeout=login_timeout_sec * 1000)
        return

# ---------- AsyncComputer backed by Playwright --------------------------------

@dataclass
class LocalPlaywrightComputer(AsyncComputer):
    width: int = 1280
    height: int = 900
    start_url: str = "https://chatgpt.com/"
    channel: str = "chrome"           # use system Chrome
    user_data_dir: str | None = None  # pass real Chrome profile root to reuse login

    _pw = None
    _context = None
    _page = None

    @property
    def environment(self):
        return "browser"

    @property
    def dimensions(self) -> tuple[int, int]:
        return (self.width, self.height)

    async def start(self):
        self._pw = await async_playwright().start()
        # If you reuse a real profile, make sure no regular Chrome is running.
        try:
            self._context = await self._pw.chromium.launch_persistent_context(
                user_data_dir=self.user_data_dir or "",
                channel=self.channel,
                headless=False,
                viewport={"width": self.width, "height": self.height},
                args=["--disable-blink-features=AutomationControlled"],
            )
        except Exception as e:
            msg = str(e).lower()
            if "another browser instance" in msg or "profile" in msg and "in use" in msg:
                raise RuntimeError(
                    "Chrome refused to start with that profile (is Chrome already running?). "
                    "Close all Chrome windows and try again."
                ) from e
            raise
        self._page = self._context.pages[0] if self._context.pages else await self._context.new_page()
        if self.start_url:
            await self._page.goto(self.start_url, wait_until="domcontentloaded")

    async def stop(self):
        if self._context:
            await self._context.close()
        if self._pw:
            await self._pw.stop()

    # ---- Computer actions required by AsyncComputer ----
    async def screenshot(self) -> str:
        png = await self._page.screenshot(full_page=False)
        return base64.b64encode(png).decode("utf-8")

    async def click(self, x: int, y: int, button: str) -> None:
        await self._page.mouse.click(x, y, button=button.lower())

    async def double_click(self, x: int, y: int) -> None:
        await self._page.mouse.dblclick(x, y)

    async def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        await self._page.mouse.move(x, y)
        await self._page.mouse.wheel(scroll_x, scroll_y)

    async def type(self, text: str) -> None:
        await self._page.keyboard.type(text)

    async def wait(self) -> None:
        await self._page.wait_for_timeout(800)

    async def move(self, x: int, y: int) -> None:
        await self._page.mouse.move(x, y)

    async def keypress(self, keys: list[str]) -> None:
        if not keys:
            return
        if len(keys) == 1:
            await self._page.keyboard.press(keys[0])
            return
        for k in keys[:-1]:
            await self._page.keyboard.down(k)
        await self._page.keyboard.press(keys[-1])
        for k in reversed(keys[:-1]):
            await self._page.keyboard.up(k)

    async def drag(self, path: list[tuple[int, int]]) -> None:
        if not path:
            return
        x0, y0 = path[0]
        await self._page.mouse.move(x0, y0)
        await self._page.mouse.down()
        for x, y in path[1:]:
            await self._page.mouse.move(x, y)
        await self._page.mouse.up()

# ---------- Main --------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Post to ChatGPT via Agents ComputerTool.")
    parser.add_argument("--say", default="Shall we play a game?", help="Message to send.")
    parser.add_argument("--reuse-profile", action="store_true",
                        help="Reuse your signed-in Chrome profile (close Chrome first).")
    parser.add_argument("--profile-dir", default=None,
                        help="Explicit Chrome user-data root to reuse (NOT the '/Default' subfolder).")
    parser.add_argument("--login-timeout", type=int, default=600,
                        help="Seconds to wait for manual login if needed.")
    args = parser.parse_args()

    user_data_dir = None
    if args.profile_dir:
        user_data_dir = os.path.expanduser(args.profile_dir)
    elif args.reuse_profile:
        user_data_dir = default_chrome_user_data_root()

    if user_data_dir:
        Path(user_data_dir).mkdir(parents=True, exist_ok=True)
        print(f"Using Chrome user-data dir: {user_data_dir}")

    computer = LocalPlaywrightComputer(
        start_url="https://chatgpt.com/",
        channel="chrome",
        user_data_dir=user_data_dir,
    )
    await computer.start()

    # Preflight: ensure we're past login gate (or wait for you to finish it)
    await ensure_logged_in(computer._page, args.login_timeout)

    model = OpenAIResponsesModel(
        model="computer-use-preview",
        openai_client=AsyncOpenAI(),
    )

    agent = Agent(
        name="Desktop operator",
        model=model,
        instructions=(
            "You are controlling a visible Chrome window. "
            "Go to a new or existing chat at chatgpt.com, click the message textbox, "
            f"type exactly: {args.say!r}, press Enter to send, then verify it appears. "
            "Finally reply with 'done'."
        ),
        tools=[ComputerTool(computer=computer, on_safety_check=lambda _: True)],
        model_settings=ModelSettings(truncation="auto"),  # required for computer-use
    )

    try:
        result = await Runner.run(agent, input="Proceed.")
        print("\nAGENT:", result.final_output)
    finally:
        await computer.stop()

if __name__ == "__main__":
    asyncio.run(main())

