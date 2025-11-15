from openai import OpenAI
client = OpenAI()

def get_response(user_input):
    response = client.responses.create(
        prompt={
            "id": "pmpt_68941be775f88194897e3b7d258d485f0640b57732216b4a",
            "version": "2"
        },
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": user_input
                    }
                ]
            }
        ],
        reasoning={},
        max_output_tokens=2048,
        store=True
    )
    return response.output_text

def main():
    user_input = input("> ")
    response = get_response(user_input)
    print(f"Mixis: {response}")

if __name__ == "__main__":
    main()
