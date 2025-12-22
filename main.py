"""
Interactive Chat with OpenAI + Memobase Memory
Read the docs: https://docs.memobase.io/practices/openai

Features:
- Interactive chat loop (type 'exit' to quit)
- Buffer size = 5 messages (short-term context)
- Auto-flush at 1024 tokens (handled by MemoBase)
- Manual flush only on exit (to catch remaining data)
- Long-term memory with Memobase
"""

import sys
import signal
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
MODEL = "gpt-4o"

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
u = mb_client.get_or_create_user(string_to_uuid(USER_NAME))


# 3. Patch OpenAI client with memory capability
client = openai_memory(client, mb_client, max_context_size=1000)

# Short-term conversation history (buffer)
conversation_history = []
conversation_count = 0


def graceful_exit():
    """Flush and exit gracefully"""
    print("\n💾 Saving remaining conversations...")
    sleep(0.1)
    try:
        client.flush(USER_NAME)
        print("✅ Memory saved!")
    except Exception as e:
        print(f"⚠️  Warning: Failed to flush: {e}")
    print("👋 Goodbye!")


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    graceful_exit()
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def chat_interactive():
    """
    Interactive chat with buffer-based short-term memory.
    Keeps last BUFFER_SIZE messages as context.
    Relies on auto-flush (1024 tokens) for processing.
    Manual flush only on exit to catch remaining data.
    """
    global conversation_history, conversation_count
    
    print("\n" + "="*60)
    print("🤖 INTERACTIVE CHAT WITH MEMORY")
    print("="*60)
    print(f"📝 Buffer Size: {BUFFER_SIZE} messages")
    print(f"👤 User: {USER_NAME}")
    print(f"💡 Commands: 'exit' to quit, '/memory' to view memory")
    print(f"🔄 Auto-flush: Every 1024 tokens or 1 hour")
    print("="*60 + "\n")
    
    while True:
        # Get user input
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            graceful_exit()
            break
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.lower() == 'exit':
            graceful_exit()
            break
        
        if user_input.lower() == '/memory':
            show_memory()
            continue
        
        # Add user message to buffer
        conversation_history.append({"role": "user", "content": user_input})
        
        # Keep only last BUFFER_SIZE messages (sliding window)
        if len(conversation_history) > BUFFER_SIZE * 2:  # *2 because user+assistant pairs
            conversation_history = conversation_history[1:]
        
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
            conversation_count += 1
        
        except Exception as e:
            print(f"❌ Error: {e}\n")


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

def demo_search_event_profile():
    """Demo function for search_event_profile"""
    from rich import print as rprint
    from concurrent.futures import ThreadPoolExecutor
    
    def search_event_profile(query: str = None):
        chats = [{"role": "user", "content": query}]

        # Chạy song song
        with ThreadPoolExecutor(max_workers=2) as executor:
            search_future = executor.submit(u.search_event, query=query)
            profile_future = executor.submit(u.profile, chats=chats)
            
            search_result = search_future.result()
            profile_result = profile_future.result()

        # Xử lý profile: topic::sub_topic::content
        profile_string = "\n".join([
            f"{p.topic}::{p.sub_topic}::{p.content}" 
            for p in profile_result
        ])
        
        # Nối event + profile với separator
        combined_result = f"## Events\n{search_result}\n\n## Profile\n{profile_string}"
        
        return combined_result

    query = "My occupation is a software engineer"
    result = search_event_profile(query)
    rprint(result)


def demo_agent_interactive():
    """Interactive demo with LangChain Agent"""
    from memobase.patch.Agent import create_memobase_agent
    from memobase.patch.Agent_pro import create_memobase_agent as create_memobase_agent_pro
    
    print("\n" + "="*60)
    print("🤖 LANGCHAIN AGENT WITH MEMOBASE MEMORY")
    print("="*60)
    print(f"👤 User: {USER_NAME}")
    print(f"🧠 Model: {MODEL}")
    print(f"💡 Commands: 'exit' to quit, '/memory' to view profile, '/flush' to save")
    print(f"🔧 Tools: search_event_profile")
    print("="*60 + "\n")
    
    # Create agent
    agent = create_memobase_agent_pro(
        mb_client=mb_client,
        llm_api_key=os.getenv('llm_api_key'),
        llm_base_url="https://api.openai.com/v1/",
        model=MODEL,
        max_profile_tokens=1000,
        temperature=0.7,
    )
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n💾 Saving remaining conversations...")
            agent.flush(USER_NAME)
            print("✅ Memory saved!")
            print("👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.lower() == 'exit':
            print("\n💾 Saving remaining conversations...")
            agent.flush(USER_NAME)
            print("✅ Memory saved!")
            print("👋 Goodbye!")
            break
        
        if user_input.lower() == '/memory':
            print("\n" + "="*60)
            print("📚 CURRENT PROFILE")
            print("="*60)
            profile = agent.get_profile(USER_NAME)
            print(profile)
            print("="*60 + "\n")
            continue
        
        if user_input.lower() == '/flush':
            print("💾 Flushing buffer...")
            agent.flush(USER_NAME)
            agent.refresh_profile(USER_NAME)
            print("✅ Buffer flushed and profile refreshed!")
            continue
        
        # Get response from agent
        print("AI: ", end="", flush=True)
        
        if STREAM:
            # Stream response
            for chunk in agent.chat_stream(USER_NAME, user_input, verbose=False):
                print(chunk, end="", flush=True)
            print("\n")
        else:
            # Non-streaming response
            response = agent.chat(USER_NAME, user_input, verbose=False)
            print(response + "\n")


def demo_agent_single_query():
    """Demo single query with agent"""
    from memobase.patch.Agent import create_memobase_agent
    
    print("\n" + "="*60)
    print("🤖 SINGLE QUERY DEMO")
    print("="*60 + "\n")
    
    # Create agent
    agent = create_memobase_agent(
        mb_client=mb_client,
        llm_api_key=os.getenv('llm_api_key'),
        llm_base_url="https://api.openai.com/v1/",
        model=MODEL,
        max_profile_tokens=1000,
    )
    
    # Test queries
    queries = [
        "What do you know about me?",
        "What's my occupation?",
        "Tell me about my work experience",
    ]
    
    for query in queries:
        print(f"User: {query}")
        response = agent.chat(USER_NAME, query)
        print(f"AI: {response}\n")
    
    # Flush at the end
    print("💾 Flushing buffer...")
    agent.flush(USER_NAME)
    print("✅ Done!")


if __name__ == "__main__":
    from rich import print as rprint
    from memobase.utils import string_to_uuid
    demo_agent_interactive()