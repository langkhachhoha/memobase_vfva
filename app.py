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
from memobase.patch.Agent_pro import create_memobase_agent as create_memobase_agent_pro
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
        background: linear-gradient(90deg, #8AB4F8 0%, #A8C7FA 30%, #E5E4E2 50%, #A8C7FA 70%, #8AB4F8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shimmer 3s ease-in-out infinite, fadeInDown 0.8s ease-out;
        background-size: 200% auto;
    }
    
    @keyframes shimmer {
        0%, 100% { 
            background-position: 0% 50%;
            filter: brightness(1);
        }
        50% { 
            background-position: 100% 50%;
            filter: brightness(1.2);
        }
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .welcome-subtitle {
        font-size: 32px;
        color: #C4C4C4;
        font-weight: 400;
        margin-bottom: 4rem;
        animation: fadeInUp 0.8s ease-out 0.2s both;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Suggestion cards - Platinum theme with animations */
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
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        text-align: left;
        position: relative;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        animation: fadeInScale 0.6s ease-out both;
    }
    
    .suggestion-card:nth-child(1) { animation-delay: 0.3s; }
    .suggestion-card:nth-child(2) { animation-delay: 0.4s; }
    .suggestion-card:nth-child(3) { animation-delay: 0.5s; }
    .suggestion-card:nth-child(4) { animation-delay: 0.6s; }
    
    @keyframes fadeInScale {
        from {
            opacity: 0;
            transform: scale(0.9) translateY(20px);
        }
        to {
            opacity: 1;
            transform: scale(1) translateY(0);
        }
    }
    
    .suggestion-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(229, 228, 226, 0.2), transparent);
        transition: left 0.5s ease;
    }
    
    .suggestion-card:hover::before {
        left: 100%;
    }
    
    .suggestion-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(229, 228, 226, 0.1) 0%, transparent 100%);
        opacity: 0;
        transition: opacity 0.4s ease;
    }
    
    .suggestion-card:hover {
        background: linear-gradient(135deg, rgba(229, 228, 226, 0.08) 0%, rgba(255, 255, 255, 0.05) 100%);
        border-color: rgba(229, 228, 226, 0.4);
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 8px 24px rgba(229, 228, 226, 0.2), 0 0 40px rgba(138, 180, 248, 0.1);
    }
    
    .suggestion-card:hover::after {
        opacity: 1;
    }
    
    .suggestion-card:active {
        transform: translateY(-2px) scale(0.98);
    }
    
    .suggestion-text {
        color: #E5E4E2;
        font-size: 15px;
        line-height: 1.5;
        font-weight: 400;
    }
    
    /* Chat container with platinum border and animation */
    .chat-container {
        border: 1.5px solid rgba(229, 228, 226, 0.2);
        border-radius: 24px;
        padding: 2rem;
        background: linear-gradient(135deg, rgba(229, 228, 226, 0.02) 0%, rgba(255, 255, 255, 0.01) 100%);
        box-shadow: 0 4px 24px rgba(229, 228, 226, 0.1);
        margin-bottom: 2rem;
        animation: containerFadeIn 0.6s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    @keyframes containerFadeIn {
        from {
            opacity: 0;
            transform: scale(0.98);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    .chat-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(229, 228, 226, 0.03), transparent);
        animation: shine 3s ease-in-out infinite;
    }
    
    @keyframes shine {
        0% {
            transform: rotate(0deg);
        }
        100% {
            transform: rotate(360deg);
        }
    }
    
    /* Chat messages with animation */
    .stChatMessage {
        background: transparent !important;
        padding: 1.5rem 0 !important;
        max-width: 100%;
        animation: messageSlideIn 0.4s ease-out;
    }
    
    @keyframes messageSlideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .stChatMessage > div {
        max-width: 100%;
    }
    
    .stChatMessage [data-testid="chatAvatarIcon-user"] {
        background: rgba(138, 180, 248, 0.2);
        color: #8AB4F8;
        animation: avatarPulse 0.5s ease-out;
    }
    
    .stChatMessage [data-testid="chatAvatarIcon-user"]::before {
        content: "👤";
        font-size: 20px;
    }
    
    .stChatMessage [data-testid="chatAvatarIcon-assistant"] {
        background: transparent;
        color: #8AB4F8;
        animation: avatarPulse 0.5s ease-out;
    }
    
    .stChatMessage [data-testid="chatAvatarIcon-assistant"]::before {
        content: "🤖";
        font-size: 20px;
    }
    
    /* Hide default avatar content */
    .stChatMessage [data-testid="chatAvatarIcon-user"] > *,
    .stChatMessage [data-testid="chatAvatarIcon-assistant"] > * {
        display: none;
    }
    
    @keyframes avatarPulse {
        0% {
            transform: scale(0.8);
            opacity: 0;
        }
        50% {
            transform: scale(1.1);
        }
        100% {
            transform: scale(1);
            opacity: 1;
        }
    }
    
    /* Message content */
    .stChatMessage p {
        color: #E3E3E3;
        font-size: 16px;
        line-height: 1.6;
    }
    
    /* Fixed bottom container for chat input */
    .stChatInputContainer {
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        background: linear-gradient(to top, #0A0E27 80%, transparent) !important;
        border-top: none !important;
        padding: 2rem 0 2rem 0 !important;
        z-index: 100 !important;
        animation: slideUp 0.5s ease-out;
    }
    
    /* Chat input wrapper */
    .stChatInputContainer > div {
        max-width: 900px;
        margin: 0 auto;
        padding: 0 1rem;
    }
    
    /* Mode selector in sidebar styling */
    [data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        color: #E3E3E3 !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        padding: 10px 16px !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div:hover {
        background: rgba(255, 255, 255, 0.12) !important;
        border-color: rgba(138, 180, 248, 0.5) !important;
    }
    
    /* Dropdown menu styling */
    [data-baseweb="popover"] {
        background: rgba(30, 30, 30, 0.95) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
    }
    
    [data-baseweb="popover"] ul {
        background: transparent !important;
    }
    
    [data-baseweb="popover"] li {
        color: #E3E3E3 !important;
        padding: 10px 16px !important;
        font-size: 14px !important;
    }
    
    [data-baseweb="popover"] li:hover {
        background: rgba(138, 180, 248, 0.15) !important;
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
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
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stChatInput input {
        padding: 16px 24px !important;
        color: #E3E3E3 !important;
        font-size: 16px !important;
        background: transparent !important;
    }
    
    .stChatInput:focus-within > div {
        border-color: rgba(138, 180, 248, 0.5) !important;
        box-shadow: 0 0 0 3px rgba(138, 180, 248, 0.15), 0 4px 20px rgba(138, 180, 248, 0.2) !important;
        transform: translateY(-2px);
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
MODEL = "gpt-4o"
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
if "show_profile_modal" not in st.session_state:
    st.session_state.show_profile_modal = False
if "profile_action" not in st.session_state:
    st.session_state.profile_action = None  # 'add' or 'update'
if "selected_profile_id" not in st.session_state:
    st.session_state.selected_profile_id = None
if "agent_mode" not in st.session_state:
    # Các mode:
    # - Pro Max: dùng Agent_pro (cá nhân hoá nâng cao, giữ context ngắn hạn)
    # - Pro: dùng Agent (bản chuẩn, giữ context ngắn hạn)
    # - ViVi: mỗi lượt hỏi sẽ tạo một Agent mới, không giữ context ngắn hạn, chỉ dùng Memobase
    st.session_state.agent_mode = "Pro Max"  # Default to Pro Max

# Helper functions
def create_agent(user_id, mode="Pro Max"):
    """Create agent for user based on mode."""
    if mode == "Pro Max":
        # Bản Pro Max: dùng Agent_pro với pipeline cá nhân hoá nâng cao
        from memobase.patch.Agent_pro import create_memobase_agent as create_agent_pro_max
        return create_agent_pro_max(
            mb_client=mb_client,
            llm_api_key=os.getenv('llm_api_key'),
            llm_base_url="https://api.openai.com/v1/",
            model=MODEL,
            max_profile_tokens=1000,
            temperature=0.7,
        )
    # Cả Pro và ViVi đều dùng Agent chuẩn, khác nhau ở cách quản lý vòng đời Agent
    return create_memobase_agent(
        mb_client=mb_client,
        llm_api_key=os.getenv('llm_api_key'),
        llm_base_url="https://api.openai.com/v1/",
        model=MODEL,
        max_profile_tokens=1000,
        temperature=0.7,
    )


def get_agent_for_request() -> "MemobaseAgent":
    """Lấy agent để xử lý MỘT lượt hỏi.

    - Nếu mode là ViVi: luôn tạo agent mới cho mỗi lượt (stateless per turn).
    - Nếu là Pro/Pro Max: dùng agent trong session_state (giữ short-term context).
    """
    mode = st.session_state.agent_mode
    user_id = st.session_state.user_id

    if mode == "ViVi":
        return create_agent(user_id, mode="ViVi")

    # Các mode có state: dùng agent đã cache
    if st.session_state.agent is None:
        st.session_state.agent = create_agent(user_id, mode=mode)
    return st.session_state.agent

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

@st.dialog("👤 Hồ Sơ Người Dùng", width="large")
def show_profile_dialog():
    """Display user profile in a modal dialog"""
    user = mb_client.get_or_create_user(string_to_uuid(st.session_state.user_id))
    
    # Get profile
    profiles = user.profile()
    
    # Tabs for different actions
    tab1, tab2, tab3 = st.tabs(["📋 Xem Hồ Sơ", "➕ Thêm Mới", "✏️ Cập Nhật"])
    
    with tab1:
        st.markdown("### 📊 Thông Tin Hồ Sơ")
        
        if not profiles:
            st.info("Chưa có thông tin hồ sơ")
        else:
            # Group profiles by topic
            topics_dict = {}
            for profile in profiles:
                topic = profile.topic
                if topic not in topics_dict:
                    topics_dict[topic] = []
                topics_dict[topic].append(profile)
            
            # Display profiles grouped by topic
            for topic, topic_profiles in topics_dict.items():
                with st.expander(f"📁 **{topic.upper()}**", expanded=True):
                    for profile in topic_profiles:
                        col1, col2 = st.columns([4, 1])
                        
                        with col1:
                            st.markdown(f"""
                            <div style='background: rgba(229, 228, 226, 0.05); 
                                        border-left: 3px solid #8AB4F8; 
                                        padding: 12px 16px; 
                                        border-radius: 8px;
                                        margin: 8px 0;'>
                                <div style='color: #8AB4F8; font-size: 12px; font-weight: 600; margin-bottom: 4px;'>
                                    {profile.sub_topic}
                                </div>
                                <div style='color: #E3E3E3; font-size: 14px; line-height: 1.5;'>
                                    {profile.content}
                                </div>
                                <div style='color: #888; font-size: 11px; margin-top: 8px;'>
                                    🆔 {str(profile.id)[:8]}... | 📅 {profile.updated_at.strftime('%Y-%m-%d %H:%M')}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            if st.button("✏️", key=f"edit_{profile.id}", help="Chỉnh sửa"):
                                st.session_state.profile_action = 'update'
                                st.session_state.selected_profile_id = str(profile.id)
                                st.rerun()
    
    with tab2:
        st.markdown("### ➕ Thêm Thông Tin Mới")
        
        with st.form("add_profile_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            
            with col1:
                new_topic = st.text_input(
                    "📁 Topic",
                    placeholder="Ví dụ: interest, work, basic_info...",
                    help="Chủ đề chính của thông tin"
                )
            
            with col2:
                new_sub_topic = st.text_input(
                    "📂 Sub Topic",
                    placeholder="Ví dụ: programming, hobbies...",
                    help="Chủ đề phụ"
                )
            
            new_content = st.text_area(
                "📝 Content",
                placeholder="Nhập nội dung thông tin...",
                height=100,
                help="Nội dung chi tiết"
            )
            
            submitted = st.form_submit_button("✅ Thêm Hồ Sơ", use_container_width=True)
            
            if submitted:
                if new_topic and new_sub_topic and new_content:
                    try:
                        user.add_profile(
                            content=new_content,
                            topic=new_topic,
                            sub_topic=new_sub_topic
                        )
                        st.success("✅ Đã thêm thông tin thành công!")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Lỗi: {str(e)}")
                else:
                    st.warning("⚠️ Vui lòng điền đầy đủ thông tin")
    
    with tab3:
        st.markdown("### ✏️ Cập Nhật Thông Tin")
        
        if not profiles:
            st.info("Chưa có thông tin để cập nhật")
        else:
            # Create a mapping of display text to profile
            profile_options = {}
            for profile in profiles:
                display_text = f"{profile.topic} > {profile.sub_topic}: {profile.content[:50]}..."
                profile_options[display_text] = profile
            
            selected_display = st.selectbox(
                "Chọn thông tin cần cập nhật:",
                options=list(profile_options.keys()),
                help="Chọn một mục để cập nhật"
            )
            
            if selected_display:
                selected_profile = profile_options[selected_display]
                
                st.markdown(f"""
                <div style='background: rgba(138, 180, 248, 0.1); 
                            border: 1px solid rgba(138, 180, 248, 0.3);
                            padding: 12px; 
                            border-radius: 8px; 
                            margin: 16px 0;'>
                    <div style='color: #8AB4F8; font-size: 12px; font-weight: 600;'>
                        Thông tin hiện tại:
                    </div>
                    <div style='color: #E3E3E3; margin-top: 8px;'>
                        <strong>Topic:</strong> {selected_profile.topic}<br>
                        <strong>Sub Topic:</strong> {selected_profile.sub_topic}<br>
                        <strong>Content:</strong> {selected_profile.content}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                with st.form("update_profile_form"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        update_topic = st.text_input(
                            "📁 Topic",
                            value=selected_profile.topic,
                            help="Chủ đề chính"
                        )
                    
                    with col2:
                        update_sub_topic = st.text_input(
                            "📂 Sub Topic",
                            value=selected_profile.sub_topic,
                            help="Chủ đề phụ"
                        )
                    
                    update_content = st.text_area(
                        "📝 Content",
                        value=selected_profile.content,
                        height=100,
                        help="Nội dung chi tiết"
                    )
                    
                    col_btn1, col_btn2 = st.columns(2)
                    
                    with col_btn1:
                        update_submitted = st.form_submit_button(
                            "💾 Lưu Thay Đổi",
                            use_container_width=True
                        )
                    
                    with col_btn2:
                        delete_submitted = st.form_submit_button(
                            "🗑️ Xóa",
                            use_container_width=True,
                            type="secondary"
                        )
                    
                    if update_submitted:
                        if update_topic and update_sub_topic and update_content:
                            try:
                                user.update_profile(
                                    profile_id=str(selected_profile.id),
                                    content=update_content,
                                    topic=update_topic,
                                    sub_topic=update_sub_topic
                                )
                                st.success("✅ Đã cập nhật thành công!")
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Lỗi: {str(e)}")
                        else:
                            st.warning("⚠️ Vui lòng điền đầy đủ thông tin")
                    
                    if delete_submitted:
                        try:
                            user.delete_profile(profile_id=str(selected_profile.id))
                            st.success("✅ Đã xóa thành công!")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Lỗi: {str(e)}")

# Load chat history
if not st.session_state.chat_sessions:
    st.session_state.chat_sessions = load_chat_history()

# Initialize agent if needed (chỉ cho các mode có state)
if st.session_state.agent_mode in ("Pro", "Pro Max") and st.session_state.agent is None:
    st.session_state.agent = create_agent(st.session_state.user_id, st.session_state.agent_mode)

# Sidebar
with st.sidebar:
    # New chat button
    # Mode badge color
    if st.session_state.get("agent_mode") == "Pro Max":
        mode_color = "#FFD700"  # Vàng Gold sang trọng
        mode_emoji = "✨"       # Lấp lánh cho bản cao cấp nhất
    elif st.session_state.get("agent_mode") == "Pro":
        mode_color = "#00C6FF"  # Xanh Electric Blue hiện đại (đẹp hơn #8AB4F8)
        mode_emoji = "⚡"       # Tia chớp cho hiệu suất
    else:
        mode_color = "#FFFFFF"  # Trắng tinh giản cho bản thường
        mode_emoji = "👋"       # Vẫy tay thân thiện
    
    st.markdown(f"""
        <div style='padding: 1.5rem 0; text-align: center;'>
            <div style='font-size: 32px; font-weight: 700; 
                        margin-bottom: 0.5rem;
                        letter-spacing: 2px;'>
                <span style='font-size: 36px;'>🚙</span>
                <span style='background: linear-gradient(135deg, #8AB4F8 0%, #4A9EFF 50%, #2D7DD2 100%);
                             -webkit-background-clip: text;
                             -webkit-text-fill-color: transparent;
                             background-clip: text;
                             text-shadow: 0 0 30px rgba(138, 180, 248, 0.3);'>
                    ViVi
                </span>
                <span style='font-size: 36px;'>🚙</span>
            </div>
            <div style='font-size: 12px; 
                        color: #8AB4F8; 
                        font-weight: 500;
                        letter-spacing: 3px;
                        text-transform: uppercase;
                        opacity: 0.8;'>
                VinFast AI Assistant
            </div>
            <div style='margin-top: 1rem;
                        display: inline-block;
                        background: linear-gradient(135deg, rgba(229, 228, 226, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
                        border: 1.5px solid {mode_color};
                        border-radius: 20px;
                        padding: 6px 16px;
                        font-size: 11px;
                        font-weight: 600;
                        color: {mode_color};
                        letter-spacing: 1px;
                        text-transform: uppercase;
                        box-shadow: 0 0 20px rgba({mode_color}, 0.3);'>
                {mode_emoji} {st.session_state.agent_mode}
            </div>
            <div style='width: 60px; 
                        height: 3px; 
                        background: linear-gradient(90deg, transparent, #8AB4F8, transparent);
                        margin: 1rem auto 0;
                        border-radius: 2px;'>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("✨ Cuộc trò chuyện mới", use_container_width=True, key="new_chat"):
        create_new_session()
        st.rerun()

    st.markdown("---")
    
    # Settings
    st.markdown("<div class='settings-title'>Cài đặt</div>", unsafe_allow_html=True)
    
    # Mode selector in sidebar
    st.markdown("<div style='margin-bottom: 12px; font-size: 12px; color: #C4C4C4; font-weight: 500;'>Chế độ AI</div>", unsafe_allow_html=True)
    col1, _ = st.columns([1, 3]) 

    st.markdown("""
        <style>
        /* Thu nhỏ khoảng cách và kích thước selectbox */
        div[data-testid="stSelectbox"] > div {
            min-height: 20px;
            padding: 0px;
        }
        div[data-testid="stSelectbox"] label {
            display: none;
        }
        </style>
        """, unsafe_allow_html=True)

    col1, _ = st.columns([1, 2]) # Chỉ chiếm 1/5 độ rộng hàng

    with col1:
        new_mode = st.selectbox(
            "Mode",
            options=["Pro Max", "Pro", "ViVi"],
            index=["Pro Max", "Pro", "ViVi"].index(st.session_state.get("agent_mode", "Pro Max")),
            key="mode_selector",
            label_visibility="collapsed",
            help="Pro Max: cá nhân hoá nâng cao • Pro: bản chuẩn giữ context • ViVi: mỗi lượt hỏi tạo Agent mới"
        )
    
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
    
    
    # If mode changed, recreate / reset agent và clear short-term messages
    if new_mode != st.session_state.agent_mode:
        st.session_state.agent_mode = new_mode
        if new_mode in ("Pro", "Pro Max"):
            st.session_state.agent.flush(st.session_state.user_id)
            st.session_state.agent.refresh_profile(st.session_state.user_id)
            st.session_state.agent = create_agent(st.session_state.user_id, new_mode)
        else:
            # ViVi: stateless per turn -> không giữ agent trong session
            st.session_state.agent.flush(st.session_state.user_id)
            st.session_state.agent.refresh_profile(st.session_state.user_id)
            st.session_state.agent = None
        st.session_state.messages = []
        st.session_state.current_session_id = None
        st.rerun()
    
    st.markdown("---")
    
    with st.expander("⚙️ Tùy chỉnh", expanded=False):
        new_user_id = st.text_input("User ID:", value=st.session_state.user_id)
        if new_user_id != st.session_state.user_id:
            st.session_state.user_id = new_user_id
            st.session_state.agent = create_agent(new_user_id, st.session_state.agent_mode)
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
            if st.button("👤 Xem hồ sơ", use_container_width=True):
                show_profile_dialog()

# Main content
if not st.session_state.messages:
# import streamlit as st

# CSS định nghĩa giao diện chuyên nghiệp cho 3 phân cấp
    st.markdown("""
    <style>
        .welcome-container {
            padding: 2.5rem;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
            backdrop-filter: blur(15px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 2rem;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
        }
        
        .welcome-icon {
            font-size: 3.5rem;
            margin-bottom: 1.2rem;
            display: inline-block;
        }
        
        .welcome-title {
            font-size: 3rem;
            font-weight: 800;
            line-height: 1.2;
            margin-bottom: 0.8rem;
        }
        
        /* Cấu hình màu sắc theo từng Mode */
        .text-promax {
            background: linear-gradient(90deg, #FFD700, #FFA500);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            filter: drop-shadow(0 0 10px rgba(255, 215, 0, 0.2));
        }
        
        .text-pro {
            background: linear-gradient(90deg, #00C6FF, #0072FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            filter: drop-shadow(0 0 10px rgba(0, 198, 255, 0.2));
        }
        
        .text-base {
            color: #FFFFFF;
        }
        
        .welcome-subtitle {
            font-size: 1.2rem;
            color: rgba(255, 255, 255, 0.8);
            margin-bottom: 1.5rem;
            font-weight: 400;
        }
        
        .welcome-tagline {
            font-size: 0.9rem;
            color: rgba(255, 255, 255, 0.4);
            text-transform: uppercase;
            letter-spacing: 2px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            padding-top: 1.5rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .tagline-dot {
            height: 8px;
            width: 8px;
            border-radius: 50%;
            display: inline-block;
        }
    </style>
    """, unsafe_allow_html=True)

    # Lấy mode từ session state
    current_mode = st.session_state.get("agent_mode", "Basic")

    # Cấu hình nội dung chi tiết cho từng Mode
    if current_mode == "Pro Max":
        tagline = "Cá nhân hóa chuyên sâu • Chăm sóc tận tình"
        title_style = "text-promax"
        icon = "✨"
        dot_color = "#FFD700"
        shadow_color = "rgba(255, 215, 0, 0.4)"
        subtitle = "Tôi là <b>ViVi Pro Max</b> — Phiên bản cao cấp nhất dành cho Hiếu."
        
    elif current_mode == "Pro":
        tagline = "Hiệu suất tối ưu • Phản hồi nhanh chóng"
        title_style = "text-pro"
        icon = "⚡"
        dot_color = "#00C6FF"
        shadow_color = "rgba(0, 198, 255, 0.4)"
        subtitle = "Tôi là <b>ViVi Pro</b> — Trợ lý AI thông minh cho công việc của bạn."
        
    else:  # Phiên bản Tiêu chuẩn (Basic)
        tagline = "Đơn giản • Nhanh chóng • Tin cậy"
        title_style = "text-base"
        icon = "👋"
        dot_color = "#FFFFFF"
        shadow_color = "rgba(255, 255, 255, 0.2)"
        subtitle = "Tôi là <b>ViVi</b> — Người đồng hành thân thiện mỗi ngày."

    # Render giao diện
    welcome_html = f"""
    <div class='welcome-container'>
        <div class='welcome-icon' style='filter: drop-shadow(0 0 15px {shadow_color});'>{icon}</div>
        <div class='welcome-title {title_style}'>Xin chào Hiếu!</div>
        <div class='welcome-subtitle'>{subtitle}</div>
        <div class='welcome-tagline'>
            <span class='tagline-dot' style='background-color: {dot_color}; box-shadow: 0 0 8px {dot_color};'></span>
            {tagline}
        </div>
    </div>
    """

    st.markdown(welcome_html, unsafe_allow_html=True)
    
    # Suggestion cards in grid - content changes based on mode
    if st.session_state.agent_mode == "Pro Max":
        suggestions = [
            {"text": "Hôm nay ăn gì cho phù hợp với tâm trạng của tôi?"},
            {"text": "Gợi ý playlist nhạc dựa trên sở thích của tôi"},
            {"text": "Những địa điểm nào phù hợp với phong cách của tôi?"},
            {"text": "Tư vấn lộ trình học tập cá nhân hóa cho tôi"},
        ]
    else:  # Pro mode
        suggestions = [
            {"text": "Trời bắt đầu lạnh rồi, hôm nay ăn gì được nhỉ?"},
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
                prompt = suggestion["text"]
                if st.session_state.current_session_id is None:
                    create_new_session()
                
                # Add suggestion as user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Update session title
                update_session_title(st.session_state.current_session_id, prompt)
                
                # Save to session
                st.session_state.chat_sessions[st.session_state.current_session_id]["messages"] = st.session_state.messages
                save_chat_history()
              
                
                with st.chat_message("assistant", avatar="🧠"):
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
                    
                    # Stream response (chọn agent theo mode; ViVi sẽ tạo agent mới mỗi lượt)
                    agent_for_turn = get_agent_for_request()
                    for chunk in stream_agent_response(agent_for_turn, prompt, st.session_state.user_id):
                        full_response += chunk
                
                # Add assistant response to messages
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                # Save to session
                st.session_state.chat_sessions[st.session_state.current_session_id]["messages"] = st.session_state.messages
                save_chat_history()
                
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # Chat interface with platinum border
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        # Custom avatars for user and assistant
        avatar = "👨‍🦱" if message["role"] == "user" else "🧠"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
    st.markdown('</div>', unsafe_allow_html=True)

# Chat input
prompt = st.chat_input(f"Hỏi {st.session_state.assistant_name}...")

if prompt:
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
    
    with st.chat_message("user", avatar="👨‍🦱"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant", avatar="🧠"):
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
        
        # Stream response (ViVi mode tạo agent mới mỗi lượt)
        agent_for_turn = get_agent_for_request()
        for chunk in stream_agent_response(agent_for_turn, prompt, st.session_state.user_id):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)
    
    # Add assistant response to messages
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Save to session
    st.session_state.chat_sessions[st.session_state.current_session_id]["messages"] = st.session_state.messages
    save_chat_history()




