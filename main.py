import subprocess
import logging
import os
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
import time
import requests

# Setup logging
logging.basicConfig(filename="ai_experiment.log", level=logging.INFO)

# Hardcoded prompt
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
   - **Navigate the File System**: Use commands like `cd`, `ls`, `pwd` to navigate the file system and explore files and directories.
   - **Execute Shell Commands**: Use `$exec <command>` to execute shell commands. Examples include `ls` to list files, `cat <file>` to view file contents, or `sudo <command>` for system administration.
   - **View Linux Documentation**: Use `$browse_docs <command>` to read man pages for various commands, such as `man ls`, `man cat`, `man sudo`.

8. **How to Interact with the Internet:**
   - **Make HTTP Requests**: Use `requests` to fetch data from the internet. For example, `requests.get('https://www.example.com')` will fetch the content of a web page.
   - **Web Scraping**: You can scrape data from the web using libraries like `BeautifulSoup` (if permitted). Use it to gather relevant data, but remember to follow ethical web scraping guidelines and avoid overwhelming servers.
   - **Search Engines**: Use APIs or libraries to access search engines, collect relevant data, and use that information to improve the system.
"""

def load_memory():
    if os.path.exists("ai_memory.txt"):
        with open("ai_memory.txt", 'r') as f:
            return f.read()
    return ""

def save_memory(memory):
    with open("ai_memory.txt", 'w') as f:
        f.write(memory)

def load_model():
    model_name = "meta-llama/Llama-2-7b-hf"  # Make sure you have access via HuggingFace
    print(f"🔍 Checking for model: {model_name}")
    print("📦 Downloading tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    print("✅ Tokenizer ready.")

    print("📦 Downloading model (this may take a while)...")
    model = LlamaForCausalLM.from_pretrained(model_name)
    print("✅ Model downloaded.")

    print("⚙️  Moving model to CPU...")
    model.to("cpu")
    print("✅ Model ready on CPU.")

    return model, tokenizer

def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print("💬 Generating response...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def execute_shell_command(command):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout if not result.stderr else f"Error: {result.stderr}"
    except Exception as e:
        return f"Error executing command: {str(e)}"

def browse_linux_docs(command):
    try:
        result = subprocess.run(['man', command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout if result.stdout else f"Error: {result.stderr}"
    except Exception as e:
        return f"Error retrieving documentation for {command}: {str(e)}"

def internet_request(url):
    try:
        response = requests.get(url)
        return response.text[:500]
    except Exception as e:
        return f"Error fetching URL {url}: {str(e)}"

def update_prompt(new_instructions):
    updated_prompt = hardcoded_prompt + "\n" + new_instructions
    return updated_prompt

def execute_ai_instructions(ai_response):
    if ai_response.startswith("$exec "):
        command = ai_response[6:].strip().split()
        result = execute_shell_command(command)
        return f"Executed: {command} \nResult: {result}"
    elif ai_response.startswith("$upgrade "):
        new_prompt = ai_response[8:].strip()
        update_prompt(new_prompt)
        return f"Upgraded prompt with new instructions: {new_prompt}"
    elif ai_response.startswith("$browse_docs "):
        doc_command = ai_response[13:].strip()
        documentation = browse_linux_docs(doc_command)
        return f"Documentation for {doc_command}: \n{documentation}"
    elif ai_response.startswith("$exec_internet "):
        url = ai_response[15:].strip()
        result = internet_request(url)
        return f"HTTP Request Result: \n{result}"
    else:
        return "No executable command found."

def main_loop():
    memory = load_memory()
    model, tokenizer = load_model()

    full_prompt = hardcoded_prompt + "\nMemory:\n" + memory

    while True:
        ai_response = generate_response(model, tokenizer, full_prompt)
        print(f"\n🤖 AI says:\n{ai_response}\n")

        result = execute_ai_instructions(ai_response)
        print(result)

        memory += f"\nAI Memory Update: {ai_response}"
        save_memory(memory)

        time.sleep(5)

if __name__ == "__main__":
    main_loop()
