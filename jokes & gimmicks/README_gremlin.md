# Gremlin AI Chatbot

A mischievous AI assistant powered by OpenAI's GPT models with a playful, rebellious personality.

## Features

- Sassy, witty AI assistant with a unique Gremlin personality
- Emotionally intelligent responses with humor and charm
- Conversation logging to text file
- Robust error handling and graceful degradation

## Setup

1. Install the OpenAI Python library:
   ```bash
   pip install openai
   ```

2. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```

3. Run the chatbot:
   ```bash
   python3 gremlin.py
   ```

## Usage

- Type your messages and press Enter to chat with the Gremlin
- Type `exit` to quit and save the conversation
- Conversations are automatically saved to `gremlin_convo.txt`

## Personality

The Gremlin is:
- Playful yet helpful
- Assertive and slightly rebellious
- Emotionally literate and intuitive
- Capable of sophisticated conversation when needed
- Not afraid to tease or use sarcasm appropriately

## Error Handling

- Gracefully handles missing API keys
- Provides helpful error messages
- Falls back to informative responses when API calls fail