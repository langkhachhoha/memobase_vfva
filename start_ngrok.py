"""
Ngrok tunnel starter for Memobase Chat Demo
This script creates a public URL for the FastAPI application
"""

import os
import sys
import time
from pyngrok import ngrok, conf
import requests
from dotenv import load_dotenv
load_dotenv()

# Configuration
LOCAL_PORT = 8000
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")  # Optional: set in .env for custom domain

def start_ngrok_tunnel():
    """Start ngrok tunnel and display public URL"""
    
    print("\n" + "="*60)
    print("🌍 NGROK TUNNEL SETUP")
    print("="*60)
    
    # Set auth token if provided
    if NGROK_AUTH_TOKEN:
        print("🔑 Using ngrok auth token from environment")
        conf.get_default().auth_token = NGROK_AUTH_TOKEN
    else:
        print("ℹ️  No NGROK_AUTH_TOKEN found. Using free tier (limited sessions)")
    
    try:
        # Check if local server is running
        print(f"\n🔍 Checking local server at http://localhost:{LOCAL_PORT}...")
        try:
            response = requests.get(f"http://localhost:{LOCAL_PORT}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Local server is running")
            else:
                print("⚠️  Local server responded but health check failed")
        except requests.exceptions.RequestException:
            print("❌ Local server is not responding!")
            print(f"   Make sure FastAPI is running on port {LOCAL_PORT}")
            sys.exit(1)
        
        # Start ngrok tunnel
        print(f"\n🚀 Creating ngrok tunnel for port {LOCAL_PORT}...")
        public_url = ngrok.connect(LOCAL_PORT, bind_tls=True)
        
        print("\n" + "="*60)
        print("✅ NGROK TUNNEL ACTIVE")
        print("="*60)
        print(f"\n🌐 Public URL: {public_url}")
        print(f"📱 Local URL:  http://localhost:{LOCAL_PORT}")
        print("\n💡 Share the public URL with anyone to try the demo!")
        print("="*60)
        
        # Get tunnel info
        tunnels = ngrok.get_tunnels()
        if tunnels:
            print("\n📊 Active Tunnels:")
            for tunnel in tunnels:
                print(f"   - {tunnel.public_url} -> {tunnel.config['addr']}")
        
        print("\n⚠️  Press Ctrl+C to stop the tunnel and server")
        print("="*60 + "\n")
        
        # Keep the tunnel alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping ngrok tunnel...")
            ngrok.disconnect(public_url)
            print("✅ Tunnel closed")
            
    except Exception as e:
        print(f"\n❌ Error starting ngrok: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure ngrok is installed: pip install pyngrok")
        print("2. Check if port 8000 is already in use")
        print("3. Set NGROK_AUTH_TOKEN in .env for better limits")
        sys.exit(1)

if __name__ == "__main__":
    start_ngrok_tunnel()

