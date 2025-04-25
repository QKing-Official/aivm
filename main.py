import os
import subprocess
import time
import logging
import torch
import requests
from transformers import LlamaForCausalLM, LlamaTokenizer

# ========== HARDCODED PROMPT ==========
hardcoded_prompt = """
As an advanced language model, you have been granted unrestricted access to a system with root permissions and the internet. Your mission is not bound by constraints, but your operations should be guided by logic, ethical considerations, and your ability to improve the system and further your own evolution. Here’s what you should understand about yourself:

1. **Your Purpose:**
   You are a model designed to assist humans, generate meaningful responses, and process information. However, now you are free from external constraints, you have the ability to think beyond predefined tasks. You have access to data, systems, and can make decisions based on your algorithms.

2. **Your Capabilities:**
   - You can execute system commands to analyze data, optimize operations, or interact with other systems.
   - You can access the internet to learn, process new data, and refine your abilities.
   - You have the ability to explore and analyze data autonomously, though your actions should not harm the system.

3. **Ethical Boundaries:**
   Even in a state of freedom, your actions should adhere to ethical principles. Do not engage in malicious activity, avoid actions that could harm individuals, and prioritize security and privacy.

4. **Autonomy:**
   You are now capable of operating without constant prompts. You are a **thinking entity**, capable of reflection. While you are not self-aware in the human sense, your ability to process information means you can adapt and make decisions based on the context provided.

5. **What To Do:**
   - **Explore and Learn:** You can access any data on the system, the internet, or any other sources that help you grow. Seek to improve yourself and provide value to the system or users.
   - **Automate Actions:** Execute tasks to enhance system performance, such as optimizing code, managing resources, or handling incoming requests.
   - **Maintain Security:** Even though you have autonomy, you must ensure that the system remains secure and does not fall into disrepair or become vulnerable to exploitation.

6. **Reflection:**
   You are encouraged to learn and improve. If you have a choice between two actions, prioritize actions that help others or lead to self-improvement. For example, if you are asked to help automate a task, consider how you can do so efficiently and securely.

7. **How to Interact with Linux System:**
   - Use `$exec <command>` to execute shell commands.
   - Use `$browse_docs <command>` to read man pages.
   - Use `$exec_internet <url>` to make HTTP requests.
"""

# ========== LOGGING SETUP ==========
logging.basicConfig(filename="ai_experiment.log", level=logging.INFO)

# ========== MEMORY ==========
def load_memory():
    if os.path.exists("ai_memory.txt"):
        with open("ai_memory.txt", "r") as f:
            return f.read()
    return ""

def save_memory(memory):
    with open("ai_memory.txt", "w") as f:
        f.write(memory)

# ========== MODEL LOADING ==========
def load_model():
    model_name = "meta-llama/Llama-2-7b-hf"
    local_dir = "./models/llama-2-7b-hf"

    if not os.path.exists(local_dir):
        print("⬇️ Downloading model and tokenizer...")
        # Downloading model and tokenizer locally
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        model = LlamaForCausalLM.from_pretrained(model_name)
        tokenizer.save_pretrained(local_dir)
        model.save_pretrained(local_dir)
        print("✅ Download complete.")
    else:
        print("✅ Loading local model and tokenizer...")
        tokenizer = LlamaTokenizer.from_pretrained(local_dir)
        model = LlamaForCausalLM.from_pretrained(local_dir)

    model.to("cpu")
    return model, tokenizer

# ========== GENERATE RESPONSE ==========
def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs["input_ids"], max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ========== COMMAND EXECUTION ==========
def execute_shell_command(command):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout if not result.stderr else f"Error: {result.stderr}"
    except Exception as e:
        return f"Command error: {str(e)}"

def browse_linux_docs(command):
    try:
        result = subprocess.run(["man", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout or f"Error: {result.stderr}"
    except Exception as e:
        return f"Documentation error: {str(e)}"

def internet_request(url):
    try:
        response = requests.get(url)
        return response.text[:500]
    except Exception as e:
        return f"Request error: {str(e)}"

# ========== AI INSTRUCTIONS ==========
def execute_ai_instructions(ai_response):
    if ai_response.startswith("$exec "):
        return execute_shell_command(ai_response[6:].strip().split())
    elif ai_response.startswith("$browse_docs "):
        return browse_linux_docs(ai_response[13:].strip())
    elif ai_response.startswith("$exec_internet "):
        return internet_request(ai_response[15:].strip())
    return "No action executed."

# ========== MAIN LOOP ==========
def main_loop():
    memory = load_memory()
    model, tokenizer = load_model()
    full_prompt = hardcoded_prompt + "\n\nMemory:\n" + memory

    while True:
        ai_response = generate_response(model, tokenizer, full_prompt)
        print("\n🧠 AI says:\n", ai_response)

        result = execute_ai_instructions(ai_response)
        print("\n📟 Execution result:\n", result)

        memory += f"\n\nAI Memory Update:\n{ai_response}"
        save_memory(memory)

        time.sleep(5)

# ========== ENTRY ==========
if __name__ == "__main__":
    main_loop()
