# beowulf.py
from openai import OpenAI
client = OpenAI()

# Generates a response from the OpenAI API using the provided prompt.
# Invokes the GPT-4.1 model to process the input and extract the output text from the response.
def get_response(prompt):
    response = client.responses.create(
        model="gpt-5",
        reasoning={"effort": "high"},
        input=prompt,
        text={"verbosity": "high"},
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
            "content": '''You are a bard who travels the roads of Germanic Europe in the early Middle Ages. You have heard tale of Beowulf, a fierce warrior. In the intermediate period between his slaying of Grendel and his mother, and his final battle with the dragon, Beowulf fought another enemy, but one lost to time.
            Write a tale of Beowulf's battle with a giant butthole named Cornholio.
            Write in Old English in the style of the Beowulf poet, using kennings and alliteration, and then write the literal modern English translation at the bottom, please.'''
        },
    ]
    response = get_response(prompt_data)  # Run and get the response from the OpenAI API.
    save_output(response, 'output.txt')  # Save the output to a file.
    print(response)

if __name__ == "__main__":
    main()
