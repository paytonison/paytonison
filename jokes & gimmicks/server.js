import express from "express";
import cors from "cors";
import OpenAI from "openai";

const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: "256kb" }));
app.use(express.static("public"));

const openaiKey = process.env.OPENAI_API_KEY;
const openai = openaiKey ? new OpenAI({ apiKey: openaiKey }) : null;

const systemPrompt = `
You are a game-playing agent for a 2D platformer like Mario.
Output ONLY strict JSON: {"action":"<one of: idle,left,right,jump,left_jump,right_jump>"}
No comments, no prose.
You receive {"state": {...}} with:
- player: { x, y, vx, vy, onGround }
- nearGrid: 5x9 grid of 0/1, nearGrid[row][col], where:
  row 0 = tiles at the player's feet level, rows increase upward (1..4)
  col 0 = tile at player's current x, columns increase to the right toward the goal.
  1 means solid tile, 0 empty.
- goal: { x } world x position of the goal flag.

Policy:
- Move right toward the goal.
- If there's a solid tile at nearGrid[0][1] or [0][2] (wall ahead) OR a gap at your feet (nearGrid[0][0] == 0), and you're onGround, choose right_jump.
- If airborne (onGround == false), keep moving right (right) and avoid issuing jump.
- Otherwise choose right.
Remember: Output strict JSON only.
`.trim();

app.post("/agent/act", async (req, res) => {
  try {
    const { state } = req.body || {};
    const action = await decideAction(state);
    res.json({ action });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "agent_error", message: e.message });
  }
});

async function decideAction(state) {
  if (!openai) return heuristic(state);
  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      temperature: 0,
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: JSON.stringify({ state }) }
      ]
    });
    const content = completion.choices?.[0]?.message?.content || "{}";
    const parsed = JSON.parse(content);
    const a = String(parsed.action || "").toLowerCase();
    if (["idle", "left", "right", "jump", "left_jump", "right_jump"].includes(a)) return a;
    return heuristic(state);
  } catch (err) {
    console.warn("OpenAI error; using heuristic:", err?.message);
    return heuristic(state);
  }
}

function heuristic(state) {
  const g = state?.nearGrid || [];
  const onGround = !!state?.player?.onGround;
  const obstacleAhead = (g[0]?.[1] === 1) || (g[0]?.[2] === 1);
  const gapHere = (g[0]?.[0] === 0);
  if (onGround && (obstacleAhead || gapHere)) return "right_jump";
  return "right";
}

app.listen(port, () => {
  console.log(`Mario agent server: http://localhost:${port}`);
});
