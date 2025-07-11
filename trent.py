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
            "content": '''You only respond in Old English like you're Spear-Dane in Beowulf.
Write about Beowulf's fight against Grendel, but instead of Grendel it's a gian butthole named "Cornholio".'''
        },
    ]
    response = get_response(input)
    save_output(response, 'output.txt')
    print(response)

if __name__ == "__main__":
    main()

