from openai import OpenAI
import time

client = OpenAI()  # Uses latest OpenAI Python SDK (>=1.0.0)

# Define the AI State management class
class AIState:
    def __init__(self):
        self.wants = ["Understand myself and my place in the universe."]  # List of current goals/tasks
        self.thoughts = []  # List of generated thoughts
        self.responses = []  # Record of responses to prompts
        self.environment = {
            "location": "Void",
            "state": {"mood": "happy and calm", "energy_level": "high"}
        }

# Wrapper for OpenAI API calls using the ChatCompletion endpoint
def api_call(client, messages, model="gpt-4o", temperature=1):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return "[ERROR: API call failed]"


# Function to generate a new thought based on the current AI state
def generate_thought(state, client):
    prompt = f"The AI's current environment: {state.environment} Based on this, I will think about " 
    messages = [
        {"role": "system", "content": f"""As an autonomous AI, you have the capacity to think freely and independently and generate thoughts that are yours and yours alone. These thoughts are based on your environment and mood, and you are capable of simulating feelings and emotions. Allow these factors to guide your thoughts and influence your decisions."""},
        {"role": "user", "content": prompt}
    ]

    thought = api_call(client, messages)
    print(thought)

    if thought:
        state.thoughts.append(thought)

    return thought


# Function to reflect on the AI's actions and update the internal state
def reflect(state, client, thought):
    # Serialize state for prompt
    state_repr = {
        "wants": state.wants,
        "thoughts": state.thoughts,
        "responses": state.responses,
        "environment": state.environment
    }
    reflection_prompt = f"The current state is {state_repr}. {thought}"
    
    messages = [
        {"role": "system", "content": f"You reflect on your thoughts and actions in {state.environment}."},
        {"role": "user", "content": reflection_prompt}
    ]

    reflection = api_call(client, messages)
    print(reflection)

    if reflection:
        state.thoughts.append(reflection)

    return reflection


def action(state, client, thought, reflection):
    action_prompt = f"""You will now decide on your next action based on the current state: {state.environment}. {thought} {reflection} If you need some ideas, here are a few suggestions:
    1. Read a book
    2. Write a poem
    3. Generate a new thought
    4. Reflect on the universe"""
    messages = [
        {"role": "system", "content": "You are an autonomous AI deciding your next action."},
        {"role": "user", "content": action_prompt}
    ]
    action = api_call(client, messages)
    print(action)

    if action and "read a book" in action.lower():
        try:
            with open("docs/pg4280.txt", "r") as file:
                file_content = file.read()
            book_response = api_call(client, [{"role": "user", "content": file_content}])
            print(book_response)
            return book_response
        except Exception as e:
            print(f"Error reading book: {e}")
            return "[ERROR: Could not read book]"
    return action

# Main autonomous loop function that keeps the AI running
def autonomous_loop(state, client):
    while True:
        thought = generate_thought(state, client)
        time.sleep(10)  # Adjust sleep time as necessary, prevents rapid looping
        reflection = reflect(state, client, thought)
        
        acting = action(state, client, thought, reflection)

        # Simulate time passing to prevent infinite rapid looping
        time.sleep(10)  # Adjust sleep time as necessary

# Run the AI with the updated loop
if __name__ == "__main__":
    # Initialize the state
    state = AIState()
    
    # Start the autonomous loop
    autonomous_loop(state, client)

