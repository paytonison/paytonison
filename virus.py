# GPT-5 (Thinking) is challenged to write a Python script that uses the OpenAI Agents SDK
# to find and launch Google Chrome on macOS, including handling URLs.

#!/usr/bin/env python3
import argparse
import json
import os
import platform
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from agents import Agent, Runner, function_tool  # OpenAI Agents SDK

# ---------- Existing Chrome helpers (kept) -----------------------------------

@function_tool
def find_chrome() -> str:
    """
    Locate Google Chrome.app on macOS. Returns the absolute path to the .app
    bundle or an error string if not found.
    """
    if platform.system() != "Darwin":
        return "Error: this tool only supports macOS."

    candidates = [
        "/Applications/Google Chrome.app",
        str(Path.home() / "Applications/Google Chrome.app"),
    ]
    for c in candidates:
        if Path(c).exists():
            return c

    try:
        res = subprocess.run(
            ["mdfind", "kMDItemCFBundleIdentifier == 'com.google.Chrome'"],
            capture_output=True, text=True, check=False
        )
        paths = [p.strip() for p in res.stdout.splitlines() if p.strip()]
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
    Launch Google Chrome (by bundle id or explicit path). Optionally open a URL.
    """
    if platform.system() != "Darwin":
        return "Error: this tool only supports macOS."
    try:
        if not path:
            cmd = ["open", "-b", "com.google.Chrome"]
            if url:
                cmd.append(url)
        else:
            cmd = ["open", "-a", path] + ([url] if url else [])
        subprocess.run(cmd, check=True)
        return f"OK: Launched Chrome ({'with URL ' + url if url else 'no URL'})"
    except subprocess.CalledProcessError as e:
        return f"Error: launch failed (exit {e.returncode}). Command: {shlex.join(cmd)}"
    except Exception as e:
        return f"Error: {e}"

# ---------- New tools: open ChatGPT + send message ---------------------------

@function_tool
def open_chatgpt(wait_seconds: int = 3) -> str:
    """
    Open ChatGPT in Chrome (launches Chrome if needed). Returns 'OK' or error.
    """
    if platform.system() != "Darwin":
        return "Error: macOS only."
    try:
        subprocess.run(
            ["open", "-b", "com.google.Chrome", "https://chat.openai.com/"],
            check=True
        )
        # Give the page a moment; the sender tool can also wait.
        return f"OK: ChatGPT opened (wait {wait_seconds}s before inject)."
    except subprocess.CalledProcessError as e:
        return f"Error: open failed (exit {e.returncode}). {e}"

@function_tool
def chatgpt_say(message: str, wait_seconds: int = 4) -> str:
    """
    Navigate the front Chrome tab to ChatGPT and submit a message.
    Requires you're already signed in to chat.openai.com in Chrome.
    Returns 'OK: ...' or a clear error.
    """
    if platform.system() != "Darwin":
        return "Error: macOS only."

    applescript = r'''
on run argv
  set theJS to item 1 of argv
  set waitSecs to (item 2 of argv) as integer
  set theURL to "https://chat.openai.com/"

  tell application "Google Chrome"
    if (count of windows) = 0 then
      make new window
    end if
    set theWin to front window
    if (count of tabs of theWin) = 0 then
      make new tab at theWin
    end if
    set theTab to active tab of theWin
    set URL of theTab to theURL
  end tell

  delay waitSecs

  tell application "Google Chrome"
    tell active tab of front window
      set r to execute javascript theJS
    end tell
  end tell

  if r is missing value then
    return "no-result"
  else
    return r
  end if
end run
'''

    # Robust JS: tries multiple selectors (textarea and contenteditable composer),
    # dispatches input, then submits form or clicks a send button.
    js_code = f"""
(() => {{
  const TEXT = {json.dumps(message)};
  function submitFromTextarea(ta) {{
    ta.focus();
    ta.value = TEXT;
    ta.dispatchEvent(new Event('input', {{ bubbles: true }}));
    const form = ta.closest('form');
    if (form) {{
      if (typeof form.requestSubmit === 'function') {{
        form.requestSubmit();
      }} else {{
        const btn = form.querySelector('button[type="submit"], button[aria-label*="Send"], button[data-testid*="send"]');
        if (btn) btn.click();
      }}
    }} else {{
      const btn = document.querySelector('button[type="submit"], button[aria-label*="Send"], button[data-testid*="send"]');
      if (btn) btn.click();
    }}
    return "ok-textarea";
  }}
  function submitFromContentEditable(ce) {{
    ce.focus();
    ce.textContent = TEXT;
    ce.dispatchEvent(new Event('input', {{ bubbles: true }}));
    const btn = document.querySelector('form button[type="submit"], button[aria-label*="Send"], button[data-testid*="send"]');
    if (btn) btn.click();
    return "ok-contenteditable";
  }}

  // Try common textarea patterns first
  let ta = document.querySelector('form textarea, textarea[aria-label*="message" i], textarea[placeholder*="message" i], textarea[placeholder*="send" i]');
  if (ta) return submitFromTextarea(ta);

  // Fallback: contenteditable composer
  let ce = document.querySelector('[contenteditable="true"][data-testid*="composer" i], [contenteditable="true"]');
  if (ce) return submitFromContentEditable(ce);

  return "no-input";
}})();
"""

    script_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".applescript", delete=False) as f:
            f.write(applescript)
            script_path = f.name

        proc = subprocess.run(
            ["osascript", script_path, js_code, str(wait_seconds)],
            capture_output=True, text=True, check=True
        )
        result = (proc.stdout or "").strip()
        if "no-input" in result:
            return "Error: Could not find ChatGPT input. Likely not signed in yet."
        return f"OK: {result or 'submitted'}"
    except subprocess.CalledProcessError as e:
        return f"Error: osascript failed: {e.stderr or e}"
    except Exception as e:
        return f"Error: {e}"
    finally:
        if script_path:
            try: os.remove(script_path)
            except Exception: pass

# ---------- Agent ------------------------------------------------------------

agent = Agent(
    name="MacOps",
    instructions=(
        "You are a local Mac automation agent. "
        "Goal: open ChatGPT in Chrome and submit a user-provided message. "
        "First call 'open_chatgpt' (or 'launch_chrome' with the URL). "
        "Then call 'chatgpt_say' with the exact message. "
        "If sending fails with 'no-input', report that the user likely needs to sign in."
    ),
    tools=[find_chrome, launch_chrome, open_chatgpt, chatgpt_say],
)

# ---------- CLI --------------------------------------------------------------

def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set.")
        raise SystemExit(1)

    parser = argparse.ArgumentParser(description="Open ChatGPT in Chrome and send a message via the OpenAI Agents SDK.")
    parser.add_argument("--say", default="Shall we play a game?", help="The exact message to send to ChatGPT.")
    parser.add_argument("--wait", type=int, default=4, help="Seconds to wait before JS injection (page load).")
    args = parser.parse_args()

    task = (
        f"Open ChatGPT in Chrome and send this exact message: {args.say!r}. "
        f"Wait ~{args.wait}s for the page to load before injecting."
    )

    # Pass wait as part of the request context; the agent will choose tools.
    result = Runner.run_sync(agent, input=task, max_turns=6)
    print(result.final_output)

if __name__ == "__main__":
    main()

