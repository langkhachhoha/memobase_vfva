"""Read the docs of how this patch works: https://docs.memobase.io/features/openai"""

from memobase import MemoBaseClient
from openai import OpenAI
from memobase.patch.openai import openai_memory
from time import sleep
import os
from dotenv import load_dotenv  # Thêm dòng này

load_dotenv()  # Load biến từ file .env



stream = True
user_name = "langkhachhoha"

# 1. Patch the OpenAI client to use MemoBase
client = OpenAI(api_key=os.getenv('llm_api_key'), 
                base_url="https://api.openai.com/v1/")
mb_client = MemoBaseClient(
    project_url="http://localhost:8019",
    api_key="secret",
)


client = openai_memory(client, mb_client)
# ------------------------------------------


def chat(message, close_session=True, use_users=True):
    print("Q: ", message)
    r = client.chat.completions.create(
        messages=[
            {"role": "user", "content": message},
        ],
        model="gpt-4o-mini",
        stream=stream,
        user_id=user_name if use_users else None,
    )
    # Below is just displaying response from OpenAI
    if stream:
        for i in r:
            if not i.choices[0].delta.content:
                continue
            print(i.choices[0].delta.content, end="", flush=True)
        print()
    else:
        print(r.choices[0].message.content)

    # 4. Once the chat session is closed, remember to flush to keep memory updated.
    if close_session:
        sleep(0.1)  # Wait for the last message to be processed
        client.flush(user_name)


def interactive_chat():
    """Chạy vòng lặp chat tương tác với AI. Nhấn Ctrl+C để thoát."""
    print("=" * 50)
    print("🤖 Chào mừng bạn đến với AI Chat!")
    print(f"👤 User: {user_name}")
    print("💡 Nhập tin nhắn và nhấn Enter để gửi")
    print("🚪 Nhấn Ctrl+C để thoát")
    print("=" * 50)
    print()
    
    try:
        while True:
            # Nhận input từ người dùng
            user_input = input("Bạn: ").strip()
            
            # Bỏ qua nếu input rỗng
            if not user_input:
                continue
            
            # Gửi tin nhắn và nhận phản hồi
            print("AI: ", end="", flush=True)
            r = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": user_input},
                ],
                model="gpt-4o-mini",
                stream=stream,
                user_id=user_name,
            )
            
            # Hiển thị phản hồi
            if stream:
                for chunk in r:
                    if chunk.choices[0].delta.content:
                        print(chunk.choices[0].delta.content, end="", flush=True)
                print()  # Xuống dòng sau khi hoàn thành
            else:
                print(r.choices[0].message.content)
            
            print()  # Thêm dòng trống giữa các cuộc hội thoại
            
    except KeyboardInterrupt:
        # Xử lý khi người dùng nhấn Ctrl+C
        print("\n")
        print("=" * 50)
        print("👋 Đang thoát và lưu memory...")
        
        # Flush memory trước khi thoát
        sleep(0.1)
        client.flush(user_name)
        
        # Hiển thị memory đã lưu
        print("\n📝 Memory đã lưu:")
        print("-" * 30)
        print(client.get_memory_prompt(user_name))
        print("=" * 50)
        print("✅ Tạm biệt!")


if __name__ == "__main__":
    interactive_chat()
