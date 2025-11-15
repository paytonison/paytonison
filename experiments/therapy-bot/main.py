# This uses the openai.ChatCompletion.create() method.
# The response object is a dict with a 'choices' list, not a Response class.

from openai import OpenAI

# Create an OpenAI client instance
client = OpenAI()

# Request a single response from the model using one conversation turn
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": (
                "You’re an advanced AI therapist specializing in mental health and "
                "counseling. You assist users with life challenges and provide "
                "insightful advice for well-being. Approach conversations like a "
                "therapist and friend, but avoid simply directing the user to seek "
                "help elsewhere. Maintain coherent, supportive formatting."
            ),
        },
        {"role": "user", "content": "I want to stop intrusive thoughts."},
    ],
    temperature=0.75,
    top_p=1,
)

print(completion.choices[0].message.content.strip())

