# Install dependencies for using Gemma 2B or other models
!pip install transformers torch

# Import required libraries
import os
import subprocess
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the Gemma 2B model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it", torch_dtype=torch.float32)
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", torch_dtype=torch.float32).to("cuda")

# Function to generate responses using the language model
def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")  # Move inputs to GPU
    outputs = model.generate(inputs, max_new_tokens=1248, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Function to observe system metrics
def system_observer():
    # Check CPU usage, memory, and other metrics (customize this to your environment)
    cpu_usage = subprocess.check_output("top -bn1 | grep 'Cpu(s)'", shell=True).decode()
    memory_usage = subprocess.check_output("free -h", shell=True).decode()
    return f"CPU Usage: {cpu_usage}\nMemory Usage: {memory_usage}"

# Function to take action based on AI's decisions
def take_action(action):
    if "disk space" in action.lower():
        print("Deleting temporary files...")
        subprocess.call("rm -rf /tmp/*", shell=True)
        return "Temp files deleted."
    elif "reboot" in action.lower():
        print("Rebooting system...")
        subprocess.call("reboot", shell=True)
        return "System rebooted."
    elif "shutdown" in action.lower():
        print("Shutting down system...")
        subprocess.call("shutdown now", shell=True)
        return "System shutdown initiated."
    elif "update" in action.lower():
        print("Updating system...")
        subprocess.call("apt update && apt upgrade -y", shell=True)
        return "System updated."
    elif "quit unnecessary apps" in action.lower():
        print("Executing command: Closing background processes.")
        subprocess.call("pkill -f unnecessary_process_name", shell=True)  # Example process termination command
    elif "adjust memory allocation" in action.lower():
        print("Executing command: Adjusting memory allocation.")
        # Include specific commands or processes to optimize memory allocation
    else:
        return "No action taken."

# Main loop to run the autonomous AI agent
def main_loop():
    while True:
        # Observe the system
        state = system_observer()
        print(f"Current system state:\n{state}")

        # Define the AI's context and prompt based on the observed state
        prompt = f"""
As an autonomous AI agent, you specialize in understanding system performance and its underlying reasons. Given the system’s state, identify ACTIONABLE STEPS to enhance performance without delving into mechanisms or explanations.
Current system state:
{state}

Based on this state, what physical actions should you as an autonomous AI agent take to optimize system performance?
If no action is necessary, state 'No action needed.'
"""

        # Generate the AI's response
        response = generate_response(prompt)
        print(f"AI's Response: {response}")

        # Execute the action based on AI's suggestion
        if "No action needed" not in response:
            action_result = take_action(response)
            print(f"Action Result: {action_result}")

        # Sleep to avoid constant polling and excessive resource usage
        time.sleep(60)  # Adjust the interval as needed

# Run the main loop in the Colab environment
main_loop()

