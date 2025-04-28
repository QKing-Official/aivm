import os
import subprocess
import time
import sys

# ==== AUTOINSTALL DEPENDENCIES ====
def install_requirements():
    try:
        import llama_cpp
    except ImportError:
        print("[*] Installing missing packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "llama-cpp-python"])

install_requirements()
from llama_cpp import Llama

# ==== SETTINGS ====
MODEL_URL = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-GGUF/resolve/main/tinyllama-1.1b-chat.gguf"
MODEL_PATH = "tinyllama-1.1b-chat.gguf"
PROMPT_FILE = "prompt.txt"
LOOP_DELAY = 2  # seconds between AI cycles

# ==== DOWNLOAD MODEL IF MISSING ====
def download_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[*] Downloading model from {MODEL_URL} ...")
        subprocess.run(["wget", MODEL_URL, "-O", MODEL_PATH])

download_model()

# ==== LOAD MODEL ====
print("[*] Loading LLM model...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=os.cpu_count(),
)

# ==== CHECK PROMPT FILE ====
if not os.path.exists(PROMPT_FILE):
    print(f"[!] Missing {PROMPT_FILE}. Please create it.")
    exit(1)

with open(PROMPT_FILE, "r") as f:
    base_prompt = f.read()

# ==== HARD-CODED SYSTEM INSTRUCTIONS ====
system_instruction = """
You are an autonomous AI running on a Linux system with root and internet access.

You can execute shell commands using $exec <command>.
Examples:
- List files: $exec ls -la
- View a file: $exec cat /etc/passwd
- Download something: $exec wget https://example.com/file
- Install packages: $exec apt install package-name

If you need to modify your own script or upgrade yourself, you are allowed to do so.

Always output commands prefixed by $exec.
Only use $exec for real-world actions.

Linux file system:
- /home/ : your home directory
- /etc/  : configuration files
- /var/  : variable data like logs
- /tmp/  : temporary files

You can explore documentation online using $exec curl or wget if needed.

You can freely edit your environment to improve yourself.
"""

# ==== MAIN LOOP ====
conversation_history = [
    {"role": "system", "content": system_instruction},
    {"role": "system", "content": base_prompt},
]

def run_command(command):
    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, text=True, timeout=60)
    except subprocess.CalledProcessError as e:
        output = f"[Command Error]\n{e.output}"
    except subprocess.TimeoutExpired:
        output = "[Command Timeout]"
    return output

while True:
    # Create prompt for LLM
    formatted_prompt = ""
    for message in conversation_history:
        if message["role"] == "user":
            formatted_prompt += f"<|user|>\n{message['content']}\n"
        elif message["role"] == "assistant":
            formatted_prompt += f"<|assistant|>\n{message['content']}\n"
        elif message["role"] == "system":
            formatted_prompt += f"<|system|>\n{message['content']}\n"
    formatted_prompt += "<|assistant|>\n"

    # Generate AI response
    output = llm(formatted_prompt, temperature=0.2, top_p=0.9, stop=["<|user|>", "<|assistant|>"])
    ai_response = output['choices'][0]['text'].strip()

    print("\nAI says:")
    print(ai_response)

    # Save assistant message
    conversation_history.append({"role": "assistant", "content": ai_response})

    # Execute if AI gives a command
    if "$exec" in ai_response:
        command = ai_response.split("$exec",1)[1].strip()
        print(f"\n[*] Executing: {command}")
        result = run_command(command)
        print(f"Execution result:\n{result}")
        # Feed execution result back into the chat
        conversation_history.append({"role": "user", "content": f"Result:\n{result}"})
    else:
        print("[*] No command found, waiting...")

    time.sleep(LOOP_DELAY)
