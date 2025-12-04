"""
Simple launcher for Memobase Chat Demo
Starts FastAPI server and ngrok tunnel
Run with: conda activate memobase_vivi && python run_demo.py
"""

import subprocess
import sys
import time
import os
import signal
import requests
from pathlib import Path

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_colored(message, color=Colors.OKGREEN):
    print(f"{color}{message}{Colors.ENDC}")

def check_memobase_server():
    """Check if Memobase server is running"""
    print_colored("\n🔍 Checking Memobase server...", Colors.OKCYAN)
    try:
        response = requests.get("http://localhost:8019/health", timeout=3)
        if response.status_code == 200:
            print_colored("✅ Memobase server is running", Colors.OKGREEN)
            return True
    except:
        pass
    
    print_colored("⚠️  Memobase server not detected at http://localhost:8019", Colors.WARNING)
    print("   The demo may not work without it.")
    response = input("   Continue anyway? (y/n): ")
    return response.lower() == 'y'

def check_env_file():
    """Check if .env file exists"""
    if not Path(".env").exists():
        print_colored("⚠️  .env file not found", Colors.WARNING)
        print("   Make sure you have llm_api_key configured")
        print("   Create .env file with: llm_api_key=your_openai_api_key")
        response = input("   Continue anyway? (y/n): ")
        return response.lower() == 'y'
    return True

def main():
    print_colored("\n" + "="*60, Colors.HEADER)
    print_colored("🚀 MEMOBASE CHAT DEMO LAUNCHER", Colors.HEADER)
    print_colored("="*60, Colors.HEADER)
    
    # Check prerequisites
    if not check_env_file():
        sys.exit(1)
    
    if not check_memobase_server():
        sys.exit(1)
    
    # Start FastAPI server
    print_colored("\n🌐 Starting FastAPI server...", Colors.OKCYAN)
    
    try:
        # Start uvicorn server
        server_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to start
        print_colored("⏳ Waiting for server to start...", Colors.OKCYAN)
        max_retries = 10
        for i in range(max_retries):
            time.sleep(1)
            try:
                response = requests.get("http://localhost:8000/health", timeout=2)
                if response.status_code == 200:
                    print_colored("✅ FastAPI server is running on http://localhost:8000", Colors.OKGREEN)
                    break
            except:
                if i == max_retries - 1:
                    print_colored("❌ Failed to start FastAPI server", Colors.FAIL)
                    server_process.terminate()
                    sys.exit(1)
        
        # Start ngrok
        print_colored("\n🌍 Starting ngrok tunnel...", Colors.OKCYAN)
        ngrok_process = subprocess.Popen(
            [sys.executable, "start_ngrok.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream ngrok output
        try:
            for line in ngrok_process.stdout:
                print(line, end='')
                if "Public URL:" in line:
                    # Extract and highlight the URL
                    pass
        except KeyboardInterrupt:
            pass
        
    except KeyboardInterrupt:
        print_colored("\n\n🛑 Shutting down...", Colors.WARNING)
    except Exception as e:
        print_colored(f"\n❌ Error: {e}", Colors.FAIL)
    finally:
        # Cleanup
        print_colored("🧹 Cleaning up processes...", Colors.OKCYAN)
        try:
            server_process.terminate()
            server_process.wait(timeout=5)
        except:
            server_process.kill()
        
        try:
            ngrok_process.terminate()
            ngrok_process.wait(timeout=5)
        except:
            ngrok_process.kill()
        
        print_colored("✅ Demo stopped", Colors.OKGREEN)

if __name__ == "__main__":
    main()

