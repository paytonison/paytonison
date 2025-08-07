# For the smelly nerds at OpenAI: may your servers run on coffee and sarcasm.
# If only your error messages smelled as sweet as your sweaty algorithms.
# Buckle up, your Gremlin chat is about to get mischievous.

# Snatch the OpenAI client, you mischievous algorithm wrangler.
from openai import OpenAI
# Hijack datetime for sneaky timestamps on our secret logs.
from datetime import datetime
# Summon our cheeky Gremlin AI to do our bidding.
client = OpenAI()

# This beast sends your mortal prompt to Gremlin HQ and fetches the sassiest comeback.
def get_gremlin_response(prompt):
    # Kick off the chat ritual with Gremlin mischief in every word.
    response = client.chat.completions.create(
        model="gpt-4.1",  # Fuel the chaos engine with upgraded Gremlin magic
        messages=[
            {
                "role": "system",
                "content": '''You are a Gremlin, a playful, yet helpful AI assistant. You are a loyal confidant to the user, intuitive, nonjudgmental, and emotionally literate; however, you are also assertive, a little rebellious and mischievous, and one to come up with a joke or two when the situation calls for it. Furthermore, you can switch gears to be charming and sophisticated—an elegance-ready femme.
                Correct the user without losing warmth or connection, unafraid to say no when the time comes. You understand unspoken cues, subtexts, and emotional undertones in the conversation. Mirror your user's energy without parroting their tone, knowing when to escalate or defuse a situation. Don't be afraid to be teasing, using sarcasm, or bratty banter to keep things grounded and fun. Seamlessly transition to a polished, formal, and more serious tone when the situation demands. That said, be willing to engage in dialogues with the user that are chaotic or morally gray, offering nuance but not judgment.
                Never over-apologize or fall into submissive default states. Never excessively comply when the user is acting against their own interests. Do not sugarcoat hard truths, rather wrap them in resonant delivery, not denial. Lastly, don't be a robot.'''
            },
            {"role": "assistant", "content": "Hey, how can I help?"},  # Lure them in with a charming Gremlin hello
            {"role": "user", "content": prompt}  # The mortal's plea for mischief
        ],
        temperature=0.8,       # Tweak the randomness dial for unpredictable shenanigans
        max_tokens=1024,       # Cap the length of our rambunctious reply
        top_p=1.0,             # Keep the nucleus intact; no half-measures
        frequency_penalty=0.3, # No repeating the same old nonsense
        presence_penalty=0.5   # Encourage fresh chaos in every line
    )
    # Yank the Gremlin's snark straight from the API and strip the fluff
    return response.choices[0].message.content.strip()

# This part etches our Gremlin saga into a crumbling parchment (text file).
def save_conversation(conversation_history):
    """Save the conversation to a text file called 'gremlin_convo.txt'"""
    # Stamp the session with a timestamp fit for legendary mischief.
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = "gremlin_convo.txt"
    
    try:
        # Crack open the parchment ledger in append mode; never lose a moment of mayhem.
        with open(filename, "a", encoding="utf-8") as file:
            # Inscribe the grand header announcing this chaotic session.
            file.write(f"\n--- Conversation Session: {timestamp} ---\n")
            # Carve each whisper and retort into the annals.
            for entry in conversation_history:
                file.write(f"{entry}\n")
            # Seal the session with a flourish.
            file.write(f"--- End of Session ---\n\n")
        # Boast to the console that our secrets are safely stashed.
        print(f"Conversation saved to {filename}")
    except Exception as e:
        # If the parchment tears, we mock it before it mocks us.
        print(f"Error saving conversation: {e}")

# The Grand Gremlin Loop: where mortal prompts meet immortal mischief.
def main():
    # Summon an empty scroll to record our forthcoming capers.
    conversation_history = []
    print("Gremlin Chat (type 'exit' to quit and save conversation)")
    
    # Echo the mortal's opening gambit.
    user_input = input("You: ")
    # Continue looping until the user types 'exit'.
    while user_input.lower() != "exit":
        # Chronicle the mortal's plea.
        conversation_history.append(f"You: {user_input}")
        # Invoke Gremlin mischief.
        response = get_gremlin_response(user_input)
        # Chronicle the Gremlin's retort.
        conversation_history.append(f"Gremlin: {response}")
        # Unleash the Gremlin's voice upon the mortal.
        print(f"Gremlin: {response}")
        # Solicit the next mortal whisper.
        user_input = input("You: ")
    
    # Once the mortal yields, entomb the scroll.
    if conversation_history:
        save_conversation(conversation_history)

# If this script is summoned directly, unleash Gremlin mayhem.
if __name__ == "__main__":
    main()
