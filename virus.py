# GPT-5 (Thinking) is challenged to write a Python script that uses the OpenAI Agents SDK
# to find and launch Google Chrome on macOS, including handling URLs.

#!/usr/bin/env python3
import argparse
import os
import platform
import shlex
import subprocess
from pathlib import Path
from typing import Optional, Any

from agents import Agent, Runner, function_tool  # OpenAI Agents SDK

# --- Tool implementations ----------------------------------------------------

@function_tool
def find_chrome() -> str:
    """
    Locate Google Chrome.app on macOS. Returns the absolute path to the .app
    bundle or a short error string if not found.
    """
    if platform.system() != "Darwin":
        return "Error: this tool only supports macOS."

    # Common install locations
    candidates = [
        "/Applications/Google Chrome.app",
        str(Path.home() / "Applications/Google Chrome.app"),
    ]
    for c in candidates:
        if Path(c).exists():
            return c

    # Fallback: Spotlight (Finder search uses the same metadata index)
    try:
        # Search by bundle id to avoid Chromium/Canary collisions
        cmd = ["mdfind", "kMDItemCFBundleIdentifier == 'com.google.Chrome'"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        paths = [p.strip() for p in res.stdout.splitlines() if p.strip()]
        # Prefer the canonical .app bundle in /Applications if present
        for p in paths:
            if p.endswith("Google Chrome.app"):
                return p
        if paths:
            return paths[0]
        return "Error: Google Chrome not found by Spotlight."
    except Exception as e:
        return f"Error: Spotlight search failed: {e}"

@function_tool
def launch_chrome(path: Optional[str] = None, url: Optional[str] = None) -> str:
    """
    Launch Google Chrome. If 'path' is provided, use that bundle path;
    otherwise use the bundle id. Optionally open a URL in Chrome.
    """
    if platform.system() != "Darwin":
        return "Error: this tool only supports macOS."

    try:
        # Use bundle id if we weren't given an explicit .app path
        if not path:
            # open by bundle id works even if app moved
            cmd = ["open", "-b", "com.google.Chrome"]
            if url:
                cmd.append(url)
        else:
            # Launch the .app directly; pass URL if provided
            if url:
                cmd = ["open", "-a", path, url]
            else:
                cmd = ["open", "-a", path]

        subprocess.run(cmd, check=True)
        return f"OK: Launched Chrome ({'with URL ' + url if url else 'no URL'})"
    except subprocess.CalledProcessError as e:
        return f"Error: launch failed (exit {e.returncode}). Command: {shlex.join(cmd)}"
    except Exception as e:
        return f"Error: {e}"

# --- Agent definition --------------------------------------------------------

agent = Agent(
    name="MacOps",
    instructions=(
        "You are a local Mac automation agent. "
        "First, call 'find_chrome' to get the full path to Google Chrome. "
        "Then call 'launch_chrome' to launch it (include the path you found). "
        "If the user provided a URL, pass it to 'launch_chrome'. "
        "Report success or a clear error."
    ),
    model="gpt-5",
    tools=[find_chrome, launch_chrome],
)

# --- CLI runner --------------------------------------------------------------

def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set.")
        raise SystemExit(1)

    parser = argparse.ArgumentParser(description="Find and launch Google Chrome via OpenAI Agents SDK")
    parser.add_argument("--url", help="URL to open in Chrome", default=None)
    args = parser.parse_args()

    # Give the agent the task in natural language; it will decide tool calls.
    task = (
        "Find Google Chrome on this Mac and launch it."
        + (f" Open this URL afterward: {args.url}" if args.url else "")
    )

    # Run synchronously for a simple script experience
    result = Runner.run_sync(agent, input=task, max_turns=4)
    print(result.final_output)

if __name__ == "__main__":
    main()

