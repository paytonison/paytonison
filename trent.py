# Use a pipeline as a high-level helper
from transformers import pipeline

device = "mps"
torch_dtype = torch.float16

pipe = pipeline("image-text-to-text", model="google/gemma-3n-E4B-it", device=device, torch_dtype=torch_dtype)
messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are an assistant that only responds in Middle English."},
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is the capital of France?"},
        ]
    },
]
pipe(text=messages)
