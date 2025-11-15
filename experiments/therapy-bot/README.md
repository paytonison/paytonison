## Overview

This repository contains a demo of using the OpenAI GPT-4o model to simulate a supportive chat assistant. The bot is intended only as a supplement to ongoing, human-led therapy for minor or moderate issues when a therapist is momentarily unavailable.

## Features

- Uses the latest OpenAI GPT-4o model for high-quality, context-aware conversations.
- Simulates an AI therapist specializing in mental health and counseling.
- Provides empathetic, non-judgmental responses to user input.
- Can be used to triage patient needs and offer general well-being advice.

## Usage

1. **Install Dependencies**

   Ensure you have Python 3.7+ and install the OpenAI Python package:

   ```sh
   pip install openai
   ```

2. **Set Up API Key**

   Export your OpenAI API key as an environment variable:

   ```sh
   export OPENAI_API_KEY=your-api-key-here
   ```

3. **Run the Script**

   Execute the script:

   ```sh
   python main.py
   ```

   The script will simulate a conversation with the AI therapist and print the assistant's response to the console.

## Example Output

```
Intrusive thoughts can feel overwhelming, but try acknowledging them without judgment and gently refocusing on the present.
```

## Intended Use

- **Supplement to Therapy:** This demo is meant to triage minor or moderate concerns when a therapist cannot be reached.

- **Not for Crisis Situations:** It is not a substitute for full therapy or emergency services.

## Limitations

- The AI does not provide medical diagnoses or emergency support.
- Not a substitute for professional therapy or crisis intervention.
- Responses are generated based on provided prompts and may not always be appropriate for every situation.

## License

This script is provided for educational and supplementary use. Please review OpenAI's terms of service before deploying in a clinical or production environment.

