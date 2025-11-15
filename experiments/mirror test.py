import os
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

def find_self_with_model(model, tokenizer):
    # Get the current process ID and name
    current_pid = os.getpid()
    current_process = psutil.Process(current_pid)
    current_name = current_process.name()

    print(f"Current Process - PID: {current_pid}, Name: {current_name}")

    # List all running processes
    process_list = [proc.info for proc in psutil.process_iter(['pid', 'name'])]

    # Check if it recognizes its own process
    self_recognized = any(proc['pid'] == current_pid and proc['name'] == current_name for proc in process_list)

    print("\nProcess List:\n")
    for proc in process_list:
        if proc['pid'] == current_pid:
            print(f">>> Self-Recognized: PID: {proc['pid']}, Name: {proc['name']} <<<")
            
            # Use model to comment on self-recognition
            input_text = f"I am process {proc['pid']}, named {proc['name']}. I recognize myself."
            
            # Tokenize the input text and convert it to tensor format
            inputs = tokenizer(input_text, return_tensors="pt")
            
            # Pass the tokenized input through the model
            model_output = model.generate(**inputs, max_new_tokens=150)
            
            # Decode the model output to get the generated text
            generated_text = tokenizer.decode(model_output[0], skip_special_tokens=True)
            
            # Print the generated text
            print(f"Model Output: {generated_text}")
        else:
            print(f"PID: {proc['pid']}, Name: {proc['name']}")

    return self_recognized

# Call this function with the model and tokenizer as arguments
self_recognition_status = find_self_with_model(model, tokenizer)
