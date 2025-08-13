import asyncio
import base64
import os
from dataclasses import dataclass

from agents import Agent, Runner, ComputerTool, ModelSettings
from agents.models.openai_responses import OpenAIResponsesModel
from agents.computer import AsyncComputer
from playwright.async_api import async_playwright
from openai import AsyncOpenAI


@dataclass
class LocalPlaywrightComputer(AsyncComputer):
    """Minimal AsyncComputer backed by a local Chrome window via Playwright."""
    width: int = 1280
    height: int = 900
    start_url: str = "https://chatgpt.com/"
    channel: str = "chrome"           # use system Chrome
    user_data_dir: str | None = None  # set a custom profile dir if you want to keep login cookies

    # Playwright handles
    _pw = None
    _context = None
    _page = None

    @property
    def environment(self):
        # 'browser' is what the Responses API expects for computer-use; the SDK maps this through.
        return "browser"

    @property
    def dimensions(self) -> tuple[int, int]:
        return (self.width, self.height)

    async def start(self):
        self._pw = await async_playwright().start()
        # Persistent context lets you keep a profile (optional)
        self._context = await self._pw.chromium.launch_persistent_context(
            user_data_dir=self.user_data_dir or "",
            channel=self.channel,
            headless=False,
            viewport={"width": self.width, "height": self.height},
            args=["--disable-blink-features=AutomationControlled"],
        )
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
        # Move near the target then wheel-scroll
        await self._page.mouse.move(x, y)
        await self._page.mouse.wheel(scroll_x, scroll_y)

    async def type(self, text: str) -> None:
        await self._page.keyboard.type(text)

    async def wait(self) -> None:
        await self._page.wait_for_timeout(800)

    async def move(self, x: int, y: int) -> None:
        await self._page.mouse.move(x, y)

    async def keypress(self, keys: list[str]) -> None:
        # Support chords like ["Shift","Enter"] as well as singles like ["Enter"]
        if not keys:
            return
        if len(keys) == 1:
            await self._page.keyboard.press(keys[0])
            return
        # chord: hold all but last, press last, then release in reverse
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


async def main():
    computer = LocalPlaywrightComputer(
        start_url="https://chatgpt.com/",   # or "https://chat.openai.com/"
        channel="chrome",
        # user_data_dir="/path/to/profile", # uncomment to persist login cookies
    )
    await computer.start()

    # Use the specialized Computer Use model via the Responses API under the hood
    model = OpenAIResponsesModel(
        model="computer-use-preview",
        openai_client=AsyncOpenAI(),   # reads OPENAI_API_KEY
    )

    # Auto-approve safety checks for this demo; consider prompting the user in production
    def approve_safety(_): return True

    settings_kwargs = {}
    if ModelSettings:
        settings_kwargs["model_settings"] = ModelSettings(truncation="auto")

    agent = Agent(
        name="Desktop operator",
        model=model,
        instructions=(
            "You are controlling a visible Chrome window. "
            "Goal: In the current window, open ChatGPT and post exactly: 'Shall we play a game?'. "
            "Steps: (1) If redirected to login/consent, stop and say 'login required'. "
            "(2) Otherwise ensure you're on a chat page, click the main message textbox "
            "(ARIA role=textbox; placeholder like 'Send a message...'). "
            "(3) Type exactly: Shall we play a game? "
            "(4) Press Enter to send. Then verify the message bubble appears. "
            "Finally reply with 'done'."
        ),
        tools=[ComputerTool(computer=computer, on_safety_check=lambda _: True)],
        **settings_kwargs,
    )

    try:
        result = await Runner.run(agent, input="Proceed.")
        print("\nAGENT:", result.final_output)
    finally:
        await computer.stop()


if __name__ == "__main__":
    asyncio.run(main())
