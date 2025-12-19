"""
ViVi - VinFast AI Assistant
Gemini-inspired Modern UI with Streamlit
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src" / "client"))

import streamlit as st
import os
from dotenv import load_dotenv
from memobase import MemoBaseClient
from memobase.utils import string_to_uuid
from memobase.patch.Agent import create_memobase_agent
import time
import json
from datetime import datetime

load_dotenv()

# Page config
st.set_page_config(
    page_title="ViVi - VinFast AI Assistant",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS - Gemini-inspired design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&display=swap');
    
    /* Global styles */
    * {
        font-family: 'Google Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main background */
    .main {
        background: linear-gradient(135deg, #0A0E27 0%, #1A1F3A 50%, #0A0E27 100%);
        padding: 0;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #1E1E1E !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
        padding-top: 1rem !important;
        min-width: 280px !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: #1E1E1E !important;
    }
    
    [data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Force sidebar to show */
    section[data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
    }
    
    /* New chat button */
    .new-chat-btn {
        background: transparent;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 24px;
        padding: 12px 24px;
        color: #E3E3E3;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        margin-bottom: 1rem;
        width: 100%;
        text-align: left;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .new-chat-btn:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(255, 255, 255, 0.3);
    }
    
    /* Chat history items - compact */
    .chat-history-item {
        background: transparent;
        border-radius: 8px;
        padding: 8px 12px;
        margin: 2px 0;
        color: #C4C4C4;
        font-size: 13px;
        cursor: pointer;
        transition: all 0.2s ease;
        border: 1px solid transparent;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 100%;
    }
    
    .chat-history-item:hover {
        background: rgba(255, 255, 255, 0.08);
        color: #E3E3E3;
    }
    
    .chat-history-item.active {
        background: rgba(138, 180, 248, 0.15);
        color: #8AB4F8;
        border-color: rgba(138, 180, 248, 0.3);
    }
    
    /* Compact buttons in sidebar - no border */
    [data-testid="stSidebar"] .stButton > button {
        padding: 8px 12px;
        font-size: 12px;
        border-radius: 8px;
        min-height: auto;
        background: transparent;
        border: none;
        box-shadow: none;
        color: #C4C4C4;
        text-align: left;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255, 255, 255, 0.08);
        color: #E3E3E3;
    }
    
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: rgba(138, 180, 248, 0.15);
        color: #8AB4F8;
        border: none;
    }
    
    [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
        background: rgba(138, 180, 248, 0.2);
    }
    
    [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        background: transparent;
        border: none;
    }
    
    /* Main container with max width */
    .main .block-container {
        max-width: 900px;
        padding-top: 3rem;
        padding-bottom: 8rem;
        margin: 0 auto;
    }
    
    /* Ensure sidebar is visible */
    [data-testid="stSidebar"][aria-expanded="true"] {
        display: block;
    }
    
    /* Welcome section */
    .welcome-container {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        justify-content: flex-start;
        min-height: 50vh;
        padding: 2rem 0;
    }
    
    .welcome-title {
        font-size: 56px;
        font-weight: 400;
        color: #E3E3E3;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #8AB4F8 0%, #A8C7FA 50%, #8AB4F8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradient 3s ease infinite;
        background-size: 200% auto;
    }
    
    @keyframes gradient {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .welcome-subtitle {
        font-size: 32px;
        color: #C4C4C4;
        font-weight: 400;
        margin-bottom: 4rem;
    }
    
    /* Suggestion cards - Platinum theme */
    .suggestion-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 16px;
        width: 100%;
        margin-bottom: 2rem;
    }
    
    .suggestion-card {
        background: linear-gradient(135deg, rgba(229, 228, 226, 0.05) 0%, rgba(255, 255, 255, 0.03) 100%);
        border: 1.5px solid rgba(229, 228, 226, 0.2);
        border-radius: 16px;
        padding: 20px;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: left;
        position: relative;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .suggestion-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(229, 228, 226, 0.1) 0%, transparent 100%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .suggestion-card:hover {
        background: linear-gradient(135deg, rgba(229, 228, 226, 0.08) 0%, rgba(255, 255, 255, 0.05) 100%);
        border-color: rgba(229, 228, 226, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(229, 228, 226, 0.15);
    }
    
    .suggestion-card:hover::before {
        opacity: 1;
    }
    
    .suggestion-text {
        color: #E5E4E2;
        font-size: 15px;
        line-height: 1.5;
        font-weight: 400;
    }
    
    /* Chat container with platinum border */
    .chat-container {
        border: 1.5px solid rgba(229, 228, 226, 0.2);
        border-radius: 24px;
        padding: 2rem;
        background: linear-gradient(135deg, rgba(229, 228, 226, 0.02) 0%, rgba(255, 255, 255, 0.01) 100%);
        box-shadow: 0 4px 24px rgba(229, 228, 226, 0.1);
        margin-bottom: 2rem;
    }
    
    /* Chat messages */
    .stChatMessage {
        background: transparent !important;
        padding: 1.5rem 0 !important;
        max-width: 100%;
    }
    
    .stChatMessage > div {
        max-width: 100%;
    }
    
    .stChatMessage [data-testid="chatAvatarIcon-user"] {
        background: rgba(138, 180, 248, 0.2);
        color: #8AB4F8;
    }
    
    .stChatMessage [data-testid="chatAvatarIcon-assistant"] {
        background: transparent;
        color: #8AB4F8;
    }
    
    /* Message content */
    .stChatMessage p {
        color: #E3E3E3;
        font-size: 16px;
        line-height: 1.6;
    }
    
    /* Chat input - Fixed at bottom */
    .stChatInputContainer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: linear-gradient(to top, #0A0E27 80%, transparent) !important;
        border-top: none !important;
        padding: 2rem 0 2rem 0 !important;
        z-index: 100;
    }
    
    .stChatInputContainer > div {
        max-width: 800px;
        margin: 0 auto;
        padding: 0 1rem;
    }
    
    .stChatInput > div {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 28px !important;
        backdrop-filter: blur(10px);
    }
    
    .stChatInput input {
        padding: 16px 24px !important;
        color: #E3E3E3 !important;
        font-size: 16px !important;
        background: transparent !important;
    }
    
    .stChatInput:focus-within > div {
        border-color: rgba(138, 180, 248, 0.5) !important;
        box-shadow: 0 0 0 2px rgba(138, 180, 248, 0.1) !important;
    }
    
    /* Buttons - Platinum style */
    .stButton > button {
        background: linear-gradient(135deg, rgba(229, 228, 226, 0.05) 0%, rgba(255, 255, 255, 0.03) 100%);
        border: 1.5px solid rgba(229, 228, 226, 0.2);
        border-radius: 16px;
        color: #E5E4E2;
        padding: 20px;
        font-weight: 400;
        font-size: 15px;
        transition: all 0.3s ease;
        text-align: left;
        height: auto;
        white-space: normal;
        line-height: 1.5;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(229, 228, 226, 0.08) 0%, rgba(255, 255, 255, 0.05) 100%);
        border-color: rgba(229, 228, 226, 0.4);
        color: #FFFFFF;
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(229, 228, 226, 0.15);
    }
    
    .stButton > button:active {
        transform: translateY(0) scale(0.98);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.3);
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: #E3E3E3;
        padding: 10px 16px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: rgba(138, 180, 248, 0.5);
        box-shadow: 0 0 0 2px rgba(138, 180, 248, 0.1);
    }
    
    /* Labels */
    .stTextInput > label, .stSelectbox > label {
        color: #C4C4C4 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        margin-bottom: 8px !important;
    }
    
    /* Divider */
    hr {
        border-color: rgba(255, 255, 255, 0.1);
        margin: 1rem 0;
    }
    
    /* Typing indicator */
    .typing-indicator {
        display: inline-flex;
        gap: 4px;
        padding: 8px 0;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #8AB4F8;
        animation: typing 1.4s infinite;
    }
    
    .typing-dot:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .typing-dot:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes typing {
        0%, 60%, 100% {
            transform: translateY(0);
            opacity: 0.7;
        }
        30% {
            transform: translateY(-10px);
            opacity: 1;
        }
    }
    
    /* Settings section */
    .settings-section {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 16px;
        margin: 16px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .settings-title {
        color: #8AB4F8;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
MODEL = "gpt-4o-mini"
HISTORY_FILE = Path(__file__).parent / "chat_history.json"

# Initialize clients
@st.cache_resource
def init_clients():
    """Initialize MemoBase client and agent"""
    mb_client = MemoBaseClient(
        project_url="http://localhost:8019",
        api_key="secret",
    )
    return mb_client

mb_client = init_clients()

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "user_id" not in st.session_state:
    st.session_state.user_id = "langkhachhoha"
if "assistant_name" not in st.session_state:
    st.session_state.assistant_name = "ViVi"
if "agent" not in st.session_state:
    st.session_state.agent = None

# Helper functions
def create_agent(user_id):
    """Create agent for user"""
    return create_memobase_agent(
        mb_client=mb_client,
        llm_api_key=os.getenv('llm_api_key'),
        llm_base_url="https://api.openai.com/v1/",
        model=MODEL,
        max_profile_tokens=1000,
        temperature=0.7,
    )

def load_chat_history():
    """Load chat history from file"""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_chat_history():
    """Save chat history to file"""
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.chat_sessions, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Error saving chat history: {e}")

def create_new_session():
    """Create a new chat session"""
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.current_session_id = session_id
    st.session_state.messages = []
    st.session_state.chat_sessions[session_id] = {
        "title": "Cuộc trò chuyện mới",
        "messages": [],
        "created_at": datetime.now().isoformat()
    }
    save_chat_history()

def load_session(session_id):
    """Load a chat session"""
    st.session_state.current_session_id = session_id
    st.session_state.messages = st.session_state.chat_sessions[session_id]["messages"]

def update_session_title(session_id, first_message):
    """Update session title based on first message"""
    if len(first_message) > 50:
        title = first_message[:50] + "..."
    else:
        title = first_message
    st.session_state.chat_sessions[session_id]["title"] = title
    save_chat_history()

def stream_agent_response(agent, prompt, user_id):
    """Stream response from agent"""
    for chunk in agent.chat_stream(user_id, prompt, verbose=False):
        yield chunk

# Load chat history
if not st.session_state.chat_sessions:
    st.session_state.chat_sessions = load_chat_history()

# Initialize agent if needed
if st.session_state.agent is None:
    st.session_state.agent = create_agent(st.session_state.user_id)

# Sidebar
with st.sidebar:
    # New chat button
    st.markdown("""
        <div style='padding: 1rem 0;'>
            <div style='font-size: 24px; font-weight: 500; color: #E3E3E3; margin-bottom: 1rem;'>
                🚗 ViVi Assistant
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("✨ Cuộc trò chuyện mới", use_container_width=True, key="new_chat"):
        create_new_session()
        st.rerun()
    
    st.markdown("---")
    
    # Chat history
    st.markdown("<div class='settings-title'>Lịch sử trò chuyện</div>", unsafe_allow_html=True)
    
    if st.session_state.chat_sessions:
        # Sort sessions by created_at (newest first)
        sorted_sessions = sorted(
            st.session_state.chat_sessions.items(),
            key=lambda x: x[1].get("created_at", ""),
            reverse=True
        )
        
        for session_id, session_data in sorted_sessions:
            title = session_data.get("title", "Cuộc trò chuyện mới")
            
            # Truncate title to 40 characters with ellipsis
            if len(title) > 40:
                display_title = title[:40] + "..."
            else:
                display_title = title
            
            is_active = session_id == st.session_state.current_session_id
            
            col1, col2 = st.columns([6, 1])
            with col1:
                if st.button(
                    f"{display_title}",
                    key=f"session_{session_id}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary"
                ):
                    load_session(session_id)
                    st.rerun()
            
            with col2:
                if st.button("🗑️", key=f"delete_{session_id}", help="Xóa"):
                    del st.session_state.chat_sessions[session_id]
                    if st.session_state.current_session_id == session_id:
                        st.session_state.current_session_id = None
                        st.session_state.messages = []
                    save_chat_history()
                    st.rerun()
    else:
        st.caption("Chưa có lịch sử trò chuyện")
    
    st.markdown("---")
    
    # Settings
    st.markdown("<div class='settings-title'>Cài đặt</div>", unsafe_allow_html=True)
    
    with st.expander("⚙️ Tùy chỉnh", expanded=False):
        new_user_id = st.text_input("User ID:", value=st.session_state.user_id)
        if new_user_id != st.session_state.user_id:
            st.session_state.user_id = new_user_id
            st.session_state.agent = create_agent(new_user_id)
            st.success("✅ Đã cập nhật User ID")
        
        new_assistant_name = st.text_input("Tên trợ lý:", value=st.session_state.assistant_name)
        if new_assistant_name != st.session_state.assistant_name:
            st.session_state.assistant_name = new_assistant_name
            st.success("✅ Đã cập nhật tên trợ lý")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Lưu bộ nhớ", use_container_width=True):
                st.session_state.agent.flush(st.session_state.user_id)
                st.session_state.agent.refresh_profile(st.session_state.user_id)
                st.success("✅ Đã lưu!")
        
        with col2:
            if st.button("📚 Xem hồ sơ", use_container_width=True):
                profile = st.session_state.agent.get_profile(st.session_state.user_id)
                st.text_area("Hồ sơ người dùng:", profile if profile else "[Chưa có dữ liệu]", height=200)

# Main content
if not st.session_state.messages:
    # Welcome screen with centered layout
    st.markdown("""
        <div class='welcome-container'>
            <div class='welcome-title'>💥 Xin chào Hiếu!</div>
            <div class='welcome-subtitle'>Chúng ta nên bắt đầu từ đâu nhỉ?</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Suggestion cards in grid
    suggestions = [
        {"text": "Hôm nay ăn gì được?"},
        {"text": "Hãy cho list nhạc để qua được mùa đông cô đơn này"},
        {"text": "Hãy cho tôi list những địa điểm đáng để đến trong năm 2026"},
        {"text": "Làm sao để thiết kế AI Agent hiệu quả hơn?"},
    ]
    
    # Create suggestion grid
    st.markdown('<div class="suggestion-grid">', unsafe_allow_html=True)
    cols = st.columns(2)
    for idx, suggestion in enumerate(suggestions):
        with cols[idx % 2]:
            if st.button(
                suggestion['text'],
                key=f"suggestion_{idx}",
                use_container_width=True
            ):
                # Create new session if needed
                if st.session_state.current_session_id is None:
                    create_new_session()
                
                # Add suggestion as user message
                st.session_state.messages.append({"role": "user", "content": suggestion['text']})
                
                # Update session title
                update_session_title(st.session_state.current_session_id, suggestion['text'])
                
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # Chat interface with platinum border
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    st.markdown('</div>', unsafe_allow_html=True)

# Chat input (always visible)
if prompt := st.chat_input(f"Hỏi {st.session_state.assistant_name}..."):
    # Create new session if needed
    if st.session_state.current_session_id is None:
        create_new_session()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Update session title if this is the first message
    if len(st.session_state.messages) == 1:
        update_session_title(st.session_state.current_session_id, prompt)
    
    # Save to session
    st.session_state.chat_sessions[st.session_state.current_session_id]["messages"] = st.session_state.messages
    save_chat_history()
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Show typing indicator
        response_placeholder.markdown("""
            <div class='typing-indicator'>
                <div class='typing-dot'></div>
                <div class='typing-dot'></div>
                <div class='typing-dot'></div>
            </div>
        """, unsafe_allow_html=True)
        
        # Stream response
        for chunk in stream_agent_response(st.session_state.agent, prompt, st.session_state.user_id):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)
    
    # Add assistant response to messages
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Save to session
    st.session_state.chat_sessions[st.session_state.current_session_id]["messages"] = st.session_state.messages
    save_chat_history()


