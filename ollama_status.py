# ollama_status.py

import subprocess
import platform
import requests
import time

OLLAMA_API_URL = "http://localhost:11434"

def is_ollama_running():
    try:
        res = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=3)
        return res.status_code == 200
    except:
        return False

def start_ollama_server(wait_time=2):
    try:
        system = platform.system()
        if system == "Windows":
            subprocess.Popen(["start", "cmd", "/k", "ollama serve"], shell=True)
        elif system in ["Linux", "Darwin"]:
            subprocess.Popen(["ollama", "serve"])
        else:
            return False
        time.sleep(wait_time)
        return is_ollama_running()
    except Exception as e:
        print(f"[Ollama Helper] Failed to start Ollama: {e}")
        return False
