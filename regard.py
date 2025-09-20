from openai import OpenAI
client = OpenAI()

response = client.responses.create(
        model="gpt-5",
        reasoning={"effort": "high"},
        input="Write Eminem's 'Rap God' in the style of Shakespeare.")

print(response.output_text)
