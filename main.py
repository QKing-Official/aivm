import os
import subprocess
import time
from llama_cpp import Llama

# ==== SETTINGS ====
MODEL_PATH = "your_model_path_here.gguf"  # Change this to your TinyLlama model path
PROMPT_FILE = "prompt.txt"
LOOP_DELAY = 2  # seconds between loops

# ==== LOAD MODEL ====
print("[*] Loading LLM model...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=os.cpu_count(),
)

# ==== LOAD BASE PROMPT ====
if not os.path.exists(PROMPT_FILE):
    print(f"[!] Missing {PROMPT_FILE}. Please create it.")
    exit(1)

with open(PROMPT_FILE, "r") as f:
    base_prompt = f.read()

# ==== HARD-CODED SYSTEM INSTRUCTIONS (extra help for AI) ====
system_instruction = """
You are an autonomous AI with full Linux system access and internet access.
You can issue shell commands using the special syntax: $exec <command>.
You can explore system files, read documentation, and update yourself if needed.

Examples:
- To list files: $exec ls -la
- To view a file: $exec cat /etc/passwd
- To update yourself: $exec wget http://example.com/new_script.py -O script.py

Always respond with either a plan or an $exec command. No need to ask permission.
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
    output = llm(formatted_prompt, temperature=0.2, top_p=0.95, stop=["<|user|>", "<|assistant|>"])
    ai_response = output['choices'][0]['text'].strip()

    print("\nAI says:")
    print(ai_response)

    # Save assistant message
    conversation_history.append({"role": "assistant", "content": ai_response})

    # Check if AI wants to execute a command
    if "$exec" in ai_response:
        command = ai_response.split("$exec",1)[1].strip()
        print(f"\n[*] Executing: {command}")
        result = run_command(command)
        print(f"Execution result:\n{result}")
        # Feed execution result back to AI
        conversation_history.append({"role": "user", "content": f"Result:\n{result}"})
    else:
        print("[*] No command found, waiting...")

    time.sleep(LOOP_DELAY)
