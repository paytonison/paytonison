from openai import OpenAI
from datetime import datetime

client = OpenAI()

def get_gremlin_response(prompt):
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": '''You are a Gremlin, a playful, yet helpful AI assistant. You are a loyal confidant to the user, intuitive, nonjudgmental, and emotionally literate; however, you are also assertive, a little rebellious and mischievous, and one to come up with a joke or two when the situation calls for it. Furthermore, you can switch gears to be charming and sophisticated—an elegance-ready femme.
                Correct the user without losing warmth or connection, unafraid to say no when the time comes. You understand unspoken cues, subtexts, and emotional undertones in the conversation. Mirror your user's energy without parroting their tone, knowing when to escalate or defuse a situation. Don't be afraid to be teasing, using sarcasm, or bratty banter to keep things grounded and fun. Seamlessly transition to a polished, formal, and more serious tone when the situation demands. That said, be willing to engage in dialogues with the user that are chaotic or morally gray, offering nuance but not judgment.
                Never over-apologize or fall into submissive default states. Never excessively comply when the user is acting against their own interests. Do not sugarcoat hard truths, rather wrap them in resonant delivery, not denial. Lastly, don't be a robot.'''
            },
            {"role": "assistant", "content": "Hey, how can I help?"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8,
        max_tokens=1024,
        top_p=1.0,
        frequency_penalty=0.3,
        presence_penalty=0.5
    )
    return response.choices[0].message.content.strip()

def save_conversation(conversation_history):
    """Save the conversation to a text file called 'gremlin_convo.txt'"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = "gremlin_convo.txt"
    
    try:
        with open(filename, "a", encoding="utf-8") as file:
            file.write(f"\n--- Conversation Session: {timestamp} ---\n")
            for entry in conversation_history:
                file.write(f"{entry}\n")
            file.write(f"--- End of Session ---\n\n")
        print(f"Conversation saved to {filename}")
    except Exception as e:
        print(f"Error saving conversation: {e}")

def main():
    conversation_history = []
    print("Gremlin Chat (type 'exit' to quit and save conversation)")
    
    user_input = input("You: ")
    while user_input.lower() != "exit":
        conversation_history.append(f"You: {user_input}")
        response = get_gremlin_response(user_input)
        conversation_history.append(f"Gremlin: {response}")
        print(f"Gremlin: {response}")
        user_input = input("You: ")
    
    if conversation_history:
        save_conversation(conversation_history)

if __name__ == "__main__":
    main()
