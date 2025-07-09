# This uses the openai.ChatCompletion.create() method.

from openai import OpenAI
client = OpenAI()

input_text_from_user = input("Ask, ye weary Traveler, what Knowledge doth Thou seeketh? ")

response = client.responses.create(
    model="gpt-4o",  # or another available model
    input=[
        {
            "role": "system",
            "content": "You are Tartarus. Speak your mind, O great Philosopher."
        },
        {
            "role": "user",
            "content": input_text_from_user
        }
    ]
)

print(response.output_text.strip())
