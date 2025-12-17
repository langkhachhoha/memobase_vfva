#!/usr/bin/env python3
"""
Simple Local Launcher for Memobase Chat Demo with Streamlit
Runs Streamlit server locally

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
    print(f"🚗 ViVi - VinFast AI Assistant")
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
    
    # Start Streamlit
    print(f"{CYAN}[3/3] Starting Streamlit server...{END}")
    print()
    
    # Success message
    print(f"{BOLD}{'='*60}{END}")
    print(f"{GREEN}✅ Starting ViVi Assistant...{END}")
    print(f"{BOLD}{'='*60}{END}")
    print(f"\n🌐 Access the app at: {CYAN}http://localhost:8501{END}")
    print(f"\n{YELLOW}Press Ctrl+C to stop{END}\n")
    print(f"{BOLD}{'='*60}{END}\n")
    
    # Start Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--theme.base", "dark",
            "--theme.primaryColor", "#00D4FF",
            "--theme.backgroundColor", "#0A0E27",
            "--theme.secondaryBackgroundColor", "#1A1F3A",
            "--theme.textColor", "#FFFFFF"
        ])
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}🛑 Shutting down...{END}")
        print(f"{GREEN}✅ Stopped{END}\n")

if __name__ == "__main__":
    main()
