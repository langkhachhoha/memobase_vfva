#!/usr/bin/env python3
"""
Simple Local Launcher for Memobase Chat Demo
No ngrok - just runs FastAPI server locally

Run with:
  conda activate memobase_vivi && python run_local.py
"""

import subprocess
import sys
import time
import requests
from pathlib import Path

# Colors
GREEN = '\033[92m'
RED = '\033[91m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
BOLD = '\033[1m'
END = '\033[0m'

def main():
    print(f"\n{BOLD}{'='*60}")
    print(f"🚀 MEMOBASE CHAT DEMO - LOCAL MODE")
    print(f"{'='*60}{END}\n")
    
    # Check .env
    print(f"{CYAN}[1/3] Checking .env file...{END}", end=" ", flush=True)
    if not Path(".env").exists():
        print(f"{RED}❌{END}")
        print(f"   Create .env with: llm_api_key=your_openai_api_key")
        sys.exit(1)
    print(f"{GREEN}✅{END}")
    
    # Check Memobase server
    print(f"{CYAN}[2/3] Checking Memobase server...{END}", end=" ", flush=True)
    try:
        r = requests.get("http://localhost:8019/api/v1/healthcheck", timeout=2)
        if r.status_code == 200 and r.json().get("errno") == 0:
            print(f"{GREEN}✅{END}")
        else:
            print(f"{RED}❌{END}")
            print(f"   Health check failed")
            sys.exit(1)
    except:
        print(f"{RED}❌{END}")
        print(f"   Start Memobase server: cd src/server && docker-compose up -d")
        sys.exit(1)
    
    # Start FastAPI
    print(f"{CYAN}[3/3] Starting FastAPI server...{END}", end=" ", flush=True)
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app_simple:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Wait for server
    for i in range(20):
        time.sleep(0.5)
        try:
            if requests.get("http://localhost:8000/health", timeout=0.5).status_code == 200:
                print(f"{GREEN}✅{END}")
                break
        except:
            if i > 0 and i % 2 == 0:
                print(".", end="", flush=True)
    else:
        print(f"{RED}❌ Timeout{END}")
        server.terminate()
        sys.exit(1)
    
    # Success
    print(f"\n{BOLD}{'='*60}{END}")
    print(f"{GREEN}✅ Server is running!{END}")
    print(f"{BOLD}{'='*60}{END}")
    print(f"\n🌐 Access the demo at: {CYAN}http://localhost:8000{END}")
    print(f"📊 API docs at: {CYAN}http://localhost:8000/docs{END}")
    print(f"\n{YELLOW}Press Ctrl+C to stop{END}\n")
    
    # Keep running
    try:
        # Stream server logs
        for line in server.stdout:
            print(line, end='')
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}🛑 Shutting down...{END}")
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except:
            server.kill()
        print(f"{GREEN}✅ Stopped{END}\n")

if __name__ == "__main__":
    main()
