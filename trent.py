from openai import OpenAI
client = OpenAI()

def get_response(prompt):
    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
    )
    return response.output_text

def main():
    input = [
        {
            "role": "system",
            "content": '''You only respond in Middle English.
Describe the capital like you're a traveler in Chaucer's Canterbury Tales.'''
        },
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ]

    response = get_response(input)
    print(response)

if __name__ == "__main__":
    main()

