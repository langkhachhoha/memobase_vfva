"""
Interactive Chat with OpenAI + Memobase Memory
Read the docs: https://docs.memobase.io/practices/openai

Features:
- Interactive chat loop (type 'exit' to quit)
- Buffer size = 5 messages (short-term context)
- Auto-flush after every 5 conversation turns
- Long-term memory with Memobase
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src" / "client"))

from memobase import MemoBaseClient
from openai import OpenAI
from memobase.patch.openai import openai_memory
from time import sleep
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
STREAM = True
USER_NAME = "langkhachhoha"
BUFFER_SIZE = 5  
MODEL = "gpt-4o-mini"

# 1. Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv('llm_api_key'), 
    base_url="https://api.openai.com/v1/"
)

# 2. Initialize MemoBase client
mb_client = MemoBaseClient(
    project_url="http://localhost:8019",
    api_key="secret",
)

from memobase.utils import string_to_uuid
u = mb_client.get_or_create_user(string_to_uuid(USER_NAME)).profile(need_json=True)


# 3. Patch OpenAI client with memory capability
client = openai_memory(client, mb_client, max_context_size=1000)

# Short-term conversation history (buffer)
conversation_history = []
conversation_count = 0


def chat_interactive():
    """
    Interactive chat with buffer-based short-term memory.
    Keeps last BUFFER_SIZE messages as context.
    Auto-flushes to long-term memory every BUFFER_SIZE turns.
    """
    global conversation_history, conversation_count
    
    print("\n" + "="*60)
    print("🤖 INTERACTIVE CHAT WITH MEMORY")
    print("="*60)
    print(f"📝 Buffer Size: {BUFFER_SIZE} messages")
    print(f"👤 User: {USER_NAME}")
    print(f"💡 Commands: 'exit' to quit, '/memory' to view memory, '/flush' to save")
    print("="*60 + "\n")
    
    while True:
        # Get user input
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.lower() == 'exit':
            print("\n💾 Saving remaining conversations...")
            if conversation_history:
                sleep(0.1)
                client.flush(USER_NAME)
                print("✅ Memory saved!")
            print("👋 Goodbye!")
            break
        
        if user_input.lower() == '/memory':
            show_memory()
            continue
        
        if user_input.lower() == '/flush':
            manual_flush()
            continue
        
        # Add user message to buffer
        conversation_history.append({"role": "user", "content": user_input})
        
        # Keep only last BUFFER_SIZE messages (sliding window)
        if len(conversation_history) > BUFFER_SIZE * 2:  # *2 because user+assistant pairs
            conversation_history = conversation_history[-(BUFFER_SIZE * 2):]
        
        # Create chat completion with conversation history
        try:
            response = client.chat.completions.create(
                messages=conversation_history,
                model=MODEL,
                stream=STREAM,
                user_id=USER_NAME,
            )
            
            # Display and collect response
            print("AI: ", end="", flush=True)
            assistant_message = ""
            
            if STREAM:
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        print(content, end="", flush=True)
                        assistant_message += content
                print("\n")
            else:
                assistant_message = response.choices[0].message.content
                print(assistant_message + "\n")
            
            # Add assistant response to buffer
            conversation_history.append({"role": "assistant", "content": assistant_message})
            
            # Increment conversation count
            conversation_count += 1
            
            # Auto-flush after every BUFFER_SIZE turns
            if conversation_count % BUFFER_SIZE == 0:
                print(f"💾 [Auto-flush] {BUFFER_SIZE} turns completed. Saving to long-term memory...")
                sleep(0.1)
                client.flush(USER_NAME)
                print(f"✅ Memory saved! (Total turns: {conversation_count})\n")
        
        except Exception as e:
            print(f"❌ Error: {e}\n")


def manual_flush():
    """Manually flush memory buffer"""
    print("\n💾 Flushing memory...")
    sleep(0.1)
    client.flush(USER_NAME)
    print("✅ Memory flushed successfully!\n")


def show_memory():
    """Display current long-term memory"""
    print("\n" + "="*60)
    print("📚 LONG-TERM MEMORY (Memobase)")
    print("="*60)
    memory = client.get_memory_prompt(USER_NAME)
    if memory:
        print(memory)
    else:
        print("[No long-term memory stored yet]")
    print("="*60 + "\n")


# ============================================
# Main Entry Point
# ============================================

if __name__ == "__main__":
    chat_interactive()





