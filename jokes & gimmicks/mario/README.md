# LLM Mario Agent

A 2D Mario-style platformer game that connects to GPT models for AI-powered gameplay.

## Features

- Classic Mario-style 2D platformer built with HTML5 Canvas
- AI agent integration using OpenAI GPT models
- Real-time game state analysis and decision making
- Both manual and AI-controlled gameplay modes

## Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Set up your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```

3. Start the server:
   ```bash
   npm start
   ```

4. Open your browser and navigate to `http://localhost:3000`

## Controls

- **Arrow Keys**: Move Mario left/right
- **Space**: Jump
- **A Key**: Toggle AI mode on/off
- **R Key**: Reset the level

## Development

- `npm run dev` - Start with auto-reload on file changes
- `npm run lint` - Check code style
- `npm run lint:fix` - Auto-fix linting issues

## How it Works

The game captures the current game state (player position, nearby tiles, goal location) and sends it to a GPT model via the `/agent/act` endpoint. The AI analyzes the situation and returns an action (left, right, jump, etc.) which is then executed in the game.

The AI uses a simple policy:
- Move right toward the goal
- Jump when encountering obstacles or gaps
- Avoid jumping while airborne