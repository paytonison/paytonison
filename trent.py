from openai import OpenAI
client = OpenAI()

def get_response(prompt):
    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
    )
    return response.output_text


# Function below created using GitHub Copilot.
def save_output(output, filename='output.txt'):
    with open(filename, 'w') as file:
        file.write(output)


def main():
    input = [
        {
            "role": "system",
            "content": '''You only respond in Middle English like you're a traveler in Chaucer's Canterbury Tales.
Write an ode to buttholes, praising their importance in human anatomy and society.'''
        },
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ]
    response = get_response(input)
    save_output(response, 'output.txt')
    print(response)

if __name__ == "__main__":
    main()

