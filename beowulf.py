# beowulf.py
from openai import OpenAI
client = OpenAI()

# Generates a response from the OpenAI API using the provided prompt.
# Invokes the GPT-4.1 model to process the input and extract the output text from the response.
def get_response(prompt):
    response = client.responses.create(
        model="gpt-5",
        input=prompt,
    )
    return response.output_text

# Saves the API response output to a file. Designed for debugging and logging purposes in OpenAI projects.
def save_output(output, filename='output.txt'):
    with open(filename, 'w') as file:
        # Write the generated output to the file.
        file.write(output)

def main():
    prompt_data = [
        {
            "role": "system",
            "content": '''You only respond in Old English like you're Spear-Dane in Beowulf.
            Write about Beowulf's fight against Grendel, but instead of Grendel it's a gian butthole named "Cornholio".
            Then, write the literal translation of the poem in modern English.'''
        },
    ]
    response = get_response(prompt_data)  # Run and get the response from the OpenAI API.
    save_output(response, 'output.txt')  # Save the output to a file.
    print(response)

if __name__ == "__main__":
    main()
