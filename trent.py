from openai import OpenAI
client = OpenAI()

def get_response(prompt):
    response = client.responses.create(
        model="gpt-4.1",
        input=prompt
    )
    return response.output_text

if __name__ == "__main__":
    prompt = '''You are an assistant that only answers questions in Old English using Latin characters.
    What is the capital of France?
    '''
    response = get_response(prompt)
    print(response)
