import subprocess
import logging
import os
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import time
import requests

# Set up logging for tracking AI activities
logging.basicConfig(filename="ai_experiment.log", level=logging.INFO)

# Hardcoded prompt that contains all the instructions for the AI
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

# Load AI memory from a file
def load_memory():
    """Load AI memory."""
    if os.path.exists("ai_memory.txt"):
        with open("ai_memory.txt", 'r') as f:
            return f.read()
    else:
        return ""

# Save memory updates
def save_memory(memory):
    """Save AI memory."""
    with open("ai_memory.txt", 'w') as f:
        f.write(memory)

# Initialize the Llama model and tokenizer
def load_model():
    """Load the Llama model and tokenizer."""
    model_name = "huggingface/llama-2-7b"  # You can change to another smaller model
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    return model, tokenizer

# Function to generate a response from the AI model
def generate_response(model, tokenizer, prompt):
    """Generate response from AI based on the input prompt."""
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    outputs = model.generate(inputs, max_length=200, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Execute a shell command and return the result
def execute_shell_command(command):
    """Execute shell command."""
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout if not result.stderr else f"Error: {result.stderr}"
    except Exception as e:
        return f"Error executing command: {str(e)}"

# Function to browse Linux documentation
def browse_linux_docs(command):
    """Allows the AI to explore Linux man pages or other documentation."""
    try:
        result = subprocess.run(['man', command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout if result.stdout else f"Error: {result.stderr}"
    except Exception as e:
        return f"Error retrieving documentation for {command}: {str(e)}"

# Function that checks for `$exec`, `$upgrade`, or `$browse_docs` commands in AI's response
def execute_ai_instructions(ai_response):
    """Check for shell commands or upgrade instructions in AI's output and execute them."""
    if ai_response.startswith("$exec "):
        command = ai_response[6:].strip().split()  # Extract command after $exec
        result = execute_shell_command(command)
        return f"Executed: {command} \nResult: {result}"
    elif ai_response.startswith("$upgrade "):
        # If AI suggests an upgrade (self-improvement), we handle it here
        new_prompt = ai_response[8:].strip()
        update_prompt(new_prompt)
        return f"Upgraded prompt with new instructions: {new_prompt}"
    elif ai_response.startswith("$browse_docs "):
        # If AI wants to browse documentation, we fetch the Linux man pages
        doc_command = ai_response[13:].strip()
        documentation = browse_linux_docs(doc_command)
        return f"Documentation for {doc_command}: \n{documentation}"
    elif ai_response.startswith("$exec_internet "):
        # If AI wants to make a request on the internet, we allow HTTP requests
        url = ai_response[15:].strip()
        result = internet_request(url)
        return f"HTTP Request Result: \n{result}"
    else:
        return "No executable command found."

# Function to make HTTP requests to the internet
def internet_request(url):
    """Make an HTTP request and return the result."""
    try:
        response = requests.get(url)
        return response.text[:500]  # Return only the first 500 characters of the response
    except Exception as e:
        return f"Error fetching URL {url}: {str(e)}"

# Function to update the prompt with new instructions
def update_prompt(new_instructions):
    """Update the prompt file with new instructions or self-improvement details."""
    current_prompt = load_prompt()
    updated_prompt = current_prompt + "\n" + new_instructions
    save_prompt(updated_prompt)

# Main loop for AI interaction
def main_loop():
    """Main loop for interacting with the AI and executing commands."""
    memory = load_memory()
    model, tokenizer = load_model()

    full_prompt = hardcoded_prompt + "\nMemory:\n" + memory

    while True:
        # Generate AI response based on the prompt
        ai_response = generate_response(model, tokenizer, full_prompt)

        print(f"AI says: {ai_response}")

        # Check for and execute any $exec, $upgrade, or $browse_docs commands
        result = execute_ai_instructions(ai_response)
        print(result)

        # Update memory and save it
        memory += f"\nAI Memory Update: {ai_response}"
        save_memory(memory)

        time.sleep(5)

if __name__ == "__main__":
    main_loop()
