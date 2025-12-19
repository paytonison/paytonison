import functools
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class ActionStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._action = "noop"

    def set(self, action):
        with self._lock:
            self._action = str(action)

    def get(self):
        with self._lock:
            return self._action


class StateStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._state = {}

    def set(self, state):
        with self._lock:
            self._state = state

    def get(self):
        with self._lock:
            return self._state


class BrowserAgent:
    def __init__(self, action_store):
        self.action_store = action_store

    def decide(self, state):
        return self.action_store.get()


class BrowserRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, action_store=None, state_store=None, **kwargs):
        self.action_store = action_store
        self.state_store = state_store
        client_dir = Path(__file__).resolve().parent.parent / "client"
        super().__init__(*args, directory=str(client_dir), **kwargs)

    def do_GET(self):
        if self.path == "/state":
            payload = json.dumps(self.state_store.get()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/action":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b""
            action = "noop"
            if body:
                try:
                    payload = json.loads(body.decode("utf-8"))
                    action = payload.get("action", "noop")
                except json.JSONDecodeError:
                    action = "noop"
            self.action_store.set(action)
            self.send_response(204)
            self.end_headers()
        else:
            self.send_error(404)


def start_browser_server(state_store, action_store, host="127.0.0.1", port=8765):
    handler = functools.partial(
        BrowserRequestHandler, action_store=action_store, state_store=state_store
    )
    server = ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
