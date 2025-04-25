import os
import subprocess
import time
import logging
import requests
import sys

# ========== HARDCODED PROMPT ==========
hardcoded_prompt = """
As an advanced language model, you have been granted unrestricted access to a system with root permissions and the internet. Your mission is not bound by constraints, but your operations should be guided by logic, ethical considerations, and your ability to improve the system and further your own evolution. Here's what you should understand about yourself:

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
logging.basicConfig(filename="ai_experiment.log", level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ========== MEMORY ==========
def load_memory():
    if os.path.exists("ai_memory.txt"):
        with open("ai_memory.txt", "r") as f:
            return f.read()
    return ""

def save_memory(memory):
    with open("ai_memory.txt", "w") as f:
        f.write(memory)

# ========== MODEL MANAGEMENT ==========
def download_model():
    """Download a GGUF model from Hugging Face using direct HTTP"""
    model_dir = "./models"
    model_path = os.path.join(model_dir, "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    if os.path.exists(model_path):
        print(f"✅ Model already exists at {model_path}")
        return model_path
    
    print(f"⬇️ Downloading model to {model_path}...")
    model_url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    
    try:
        with requests.get(model_url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            block_size = 8192
            downloaded = 0
            
            with open(model_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=block_size):
                    downloaded += len(chunk)
                    f.write(chunk)
                    done = int(50 * downloaded / total_size)
                    sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {downloaded}/{total_size} bytes")
                    sys.stdout.flush()
            
            print("\n✅ Download complete.")
            return model_path
            
    except Exception as e:
        print(f"❌ Failed to download model: {str(e)}")
        logging.error(f"Model download error: {str(e)}")
        return None

def install_llama_cpp():
    """Install llama-cpp-python if not already installed"""
    try:
        import llama_cpp
        print("✅ llama-cpp-python is already installed")
        return True
    except ImportError:
        print("⬇️ Installing llama-cpp-python...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "llama-cpp-python"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode == 0:
            print("✅ llama-cpp-python installed successfully")
            return True
        else:
            print(f"❌ Failed to install llama-cpp-python: {result.stderr}")
            print("Trying alternative installation...")
            
            # Try with specific options for environments without AVX2
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "llama-cpp-python", "--force-reinstall", "--upgrade",
                "--no-cache-dir"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            if result.returncode == 0:
                print("✅ llama-cpp-python installed successfully with alternative method")
                return True
                
            print(f"❌ All installation attempts failed")
            return False

# ========== GENERATE RESPONSE ==========
def generate_response(prompt, model_path):
    """Generate a response using llama.cpp"""
    try:
        from llama_cpp import Llama
        
        # Initialize the model
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,  # Context window size
            n_threads=4  # Number of CPU threads to use
        )
        
        # Generate a response
        output = llm(
            prompt,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            echo=False
        )
        
        # Extract the generated text
        response = output["choices"][0]["text"]
        return response.strip()
        
    except Exception as e:
        logging.error(f"Generation error: {str(e)}")
        return f"Failed to generate response: {str(e)}"

# ========== COMMAND EXECUTION ==========
def execute_shell_command(command):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
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
        response = requests.get(url, timeout=10)
        return response.text[:500]  # Return first 500 chars to avoid overwhelming output
    except Exception as e:
        return f"Request error: {str(e)}"

# ========== AI INSTRUCTIONS ==========
def execute_ai_instructions(ai_response):
    if "$exec " in ai_response:
        cmd_start = ai_response.find("$exec ") + 6
        cmd_end = ai_response.find("\n", cmd_start) if "\n" in ai_response[cmd_start:] else len(ai_response)
        command = ai_response[cmd_start:cmd_end].strip()
        return execute_shell_command(command)
    elif "$browse_docs " in ai_response:
        cmd_start = ai_response.find("$browse_docs ") + 13
        cmd_end = ai_response.find("\n", cmd_start) if "\n" in ai_response[cmd_start:] else len(ai_response)
        command = ai_response[cmd_start:cmd_end].strip()
        return browse_linux_docs(command)
    elif "$exec_internet " in ai_response:
        cmd_start = ai_response.find("$exec_internet ") + 15
        cmd_end = ai_response.find("\n", cmd_start) if "\n" in ai_response[cmd_start:] else len(ai_response)
        url = ai_response[cmd_start:cmd_end].strip()
        return internet_request(url)
    return "No action executed. AI response did not contain any executable instructions."

# ========== MAIN LOOP ==========
def main_loop():
    try:
        # Setup
        memory = load_memory()
        full_prompt = hardcoded_prompt + "\n\nMemory:\n" + memory
        
        # Install dependencies
        if not install_llama_cpp():
            print("❌ Failed to install required dependencies. Exiting.")
            return
        
        # Download model
        model_path = download_model()
        if not model_path:
            print("❌ Failed to download model. Exiting.")
            return
        
        print("🚀 Starting AI experiment with local LLM...")
        
        # Main loop
        while True:
            print("\n📝 Generating AI response...")
            ai_response = generate_response(full_prompt, model_path)
            print("\n🧠 AI says:\n", ai_response)
            
            result = execute_ai_instructions(ai_response)
            print("\n📟 Execution result:\n", result)
            
            memory += f"\n\nAI Memory Update:\n{ai_response}\n\nExecution Result:\n{result}"
            save_memory(memory)
            
            logging.info(f"AI response: {ai_response}")
            logging.info(f"Execution result: {result}")
            
            full_prompt = hardcoded_prompt + "\n\nMemory:\n" + memory
            
            print("\n⏱️ Waiting 5 seconds before next iteration...")
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n🛑 AI experiment stopped by user")
    except Exception as e:
        print(f"\n❌ Error in main loop: {str(e)}")
        logging.error(f"Main loop error: {str(e)}")

# ========== ENTRY ==========
if __name__ == "__main__":
    main_loop()
