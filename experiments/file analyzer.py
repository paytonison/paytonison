import os
from openai import OpenAI

client = OpenAI()

def file_analyzer(file_to_analyze):
    with open(file_to_analyze, 'r') as f:
        data = f.read()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a file analyzer."},
            {"role": "user", "content": f"Is this file a document? {data}"},
        ],
        max_tokens=10,
        temperature=0.0,
    )
    
    if "yes" in response.choices[0].message.content.strip().lower():
        os.open(file_to_analyze)
        file.close()
        return True, print("This is a document.")
    else:
        file.close()
        return False, print("This is not a document.")


def count_files(directory):
    num_files = len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
    return num_files


while num_files >= 0:
    count_files("/Users/paytonison")
    num_files += 1
