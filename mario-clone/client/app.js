const stateEl = document.getElementById("state");
const statusEl = document.getElementById("status");
const actionEl = document.getElementById("action");
const tickEl = document.getElementById("tick");

const pressed = new Set();

function sendAction(action) {
  actionEl.textContent = action;
  fetch("/action", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action }),
  }).catch(() => {});
}

function currentAction() {
  const left = pressed.has("left");
  const right = pressed.has("right");
  const jump = pressed.has("jump");
  if (left && right) {
    return jump ? "jump" : "noop";
  }
  if (left && jump) return "left+jump";
  if (right && jump) return "right+jump";
  if (left) return "left";
  if (right) return "right";
  if (jump) return "jump";
  return "noop";
}

function updateFromKeys() {
  sendAction(currentAction());
}

function keyToAction(event) {
  switch (event.code) {
    case "ArrowLeft":
    case "KeyA":
      return "left";
    case "ArrowRight":
    case "KeyD":
      return "right";
    case "ArrowUp":
    case "KeyW":
    case "Space":
      return "jump";
    default:
      return "";
  }
}

window.addEventListener("keydown", (event) => {
  const action = keyToAction(event);
  if (!action) return;
  event.preventDefault();
  pressed.add(action);
  updateFromKeys();
});

window.addEventListener("keyup", (event) => {
  const action = keyToAction(event);
  if (!action) return;
  event.preventDefault();
  pressed.delete(action);
  updateFromKeys();
});

document.getElementById("left-btn").addEventListener("click", () => sendAction("left"));
document.getElementById("right-btn").addEventListener("click", () => sendAction("right"));
document.getElementById("jump-btn").addEventListener("click", () => sendAction("jump"));
document.getElementById("noop-btn").addEventListener("click", () => sendAction("noop"));

function pollState() {
  fetch("/state")
    .then((response) => response.json())
    .then((data) => {
      stateEl.textContent = JSON.stringify(data, null, 2);
      statusEl.textContent = data.status || "-";
      tickEl.textContent = data.tick ?? "-";
    })
    .catch(() => {
      stateEl.textContent = "Waiting for state...";
    });
}

pollState();
setInterval(pollState, 200);
