"""
Auto launcher for Memobase Chat Demo - No prompts
Starts FastAPI server and ngrok tunnel automatically
Run with: conda activate memobase_vivi && python run_demo_auto.py
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

def main():
    print_colored("\n" + "="*60, Colors.HEADER)
    print_colored("🚀 MEMOBASE CHAT DEMO LAUNCHER", Colors.HEADER)
    print_colored("="*60, Colors.HEADER)
    
    # Check Memobase server (warning only, don't stop)
    print_colored("\n🔍 Checking Memobase server...", Colors.OKCYAN)
    try:
        response = requests.get("http://localhost:8019/health", timeout=3)
        if response.status_code == 200:
            print_colored("✅ Memobase server is running", Colors.OKGREEN)
        else:
            print_colored("⚠️  Memobase server responded but may not be healthy", Colors.WARNING)
    except:
        print_colored("⚠️  Memobase server not detected - continuing anyway...", Colors.WARNING)
    
    # Check .env file (warning only)
    if not Path(".env").exists():
        print_colored("⚠️  .env file not found - make sure environment variables are set", Colors.WARNING)
    
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
        max_retries = 15
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
        print_colored("="*60, Colors.HEADER)
        
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
                sys.stdout.flush()
        except KeyboardInterrupt:
            pass
        
    except KeyboardInterrupt:
        print_colored("\n\n🛑 Shutting down...", Colors.WARNING)
    except Exception as e:
        print_colored(f"\n❌ Error: {e}", Colors.FAIL)
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print_colored("\n🧹 Cleaning up processes...", Colors.OKCYAN)
        try:
            server_process.terminate()
            server_process.wait(timeout=5)
        except:
            try:
                server_process.kill()
            except:
                pass
        
        try:
            ngrok_process.terminate()
            ngrok_process.wait(timeout=5)
        except:
            try:
                ngrok_process.kill()
            except:
                pass
        
        print_colored("✅ Demo stopped", Colors.OKGREEN)

if __name__ == "__main__":
    main()

