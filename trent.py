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
            "content": "You only respond in Middle English."
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

