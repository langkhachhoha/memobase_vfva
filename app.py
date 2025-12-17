"""
Streamlit App for Memobase + VinFast Assistant
Modern UI with 5 different demo modes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src" / "client"))

import streamlit as st
import os
from dotenv import load_dotenv
from memobase import MemoBaseClient
from openai import OpenAI
from memobase.patch.openai import openai_memory
from memobase.utils import string_to_uuid
from memobase.patch.Agent import create_memobase_agent
from memobase.patch.config import (
    QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME,
    PAGEINDEX_API_KEY
)
from concurrent.futures import ThreadPoolExecutor
from rich import print as rprint
import time
import base64
from pathlib import Path

load_dotenv()

# Page config
st.set_page_config(
    page_title="ViVi - VinFast AI Assistant",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, beautiful UI
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #0066FF;
        --secondary-color: #00D4FF;
        --background-dark: #0A0E27;
        --background-light: #1A1F3A;
        --text-primary: #FFFFFF;
        --text-secondary: #B4B9D3;
        --accent-purple: #8B5CF6;
        --accent-green: #10B981;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A0E27 0%, #1A1F3A 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] h1 {
        color: var(--secondary-color);
        text-align: center;
        font-size: 2rem;
        margin-bottom: 1rem;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
    }
    
    /* Radio buttons */
    .stRadio > label {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    .stRadio > div {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1rem;
    }
    
    .stRadio > div > label {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.3rem 0;
        transition: all 0.3s ease;
        border: 1px solid transparent;
    }
    
    .stRadio > div > label:hover {
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid var(--secondary-color);
        transform: translateX(5px);
    }
    
    /* Main content area */
    .main {
        background: linear-gradient(135deg, #0A0E27 0%, #1A1F3A 100%);
    }
    
    /* Headers */
    h1 {
        color: var(--text-primary);
        font-weight: 700;
        background: linear-gradient(90deg, #0066FF 0%, #00D4FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h2, h3 {
        color: var(--text-primary);
    }
    
    /* Chat messages */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Input box */
    .stChatInputContainer {
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.02);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #0066FF 0%, #00D4FF 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 102, 255, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.5);
    }
    
    /* Info boxes */
    .stAlert {
        background: rgba(139, 92, 246, 0.1);
        border: 1px solid var(--accent-purple);
        border-radius: 10px;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: var(--secondary-color);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #0066FF 0%, #00D4FF 100%);
        border-radius: 10px;
    }
    
    /* Mode badges */
    .mode-badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    
    .mode-openai {
        background: linear-gradient(90deg, #10B981 0%, #059669 100%);
        color: white;
    }
    
    .mode-agent {
        background: linear-gradient(90deg, #8B5CF6 0%, #7C3AED 100%);
        color: white;
    }
    
    .mode-semantic {
        background: linear-gradient(90deg, #0066FF 0%, #00D4FF 100%);
        color: white;
    }
    
    .mode-reasoning {
        background: linear-gradient(90deg, #F59E0B 0%, #D97706 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
MODEL = "gpt-4o-mini"
BUFFER_SIZE = 5

# PDF Configuration
PDF_DIRECTORY = "/Users/apple/VSF/pageindex/document"
PDF_FILES = {
    "doc1.pdf": "VinFast VF8 User Manual",
    "doc2.pdf": "VinFast VF9 User Manual", 
    "doc3.pdf": "VinFast Safety Guide",
    "doc4.pdf": "VinFast Maintenance Guide"
}

# Initialize clients
@st.cache_resource
def init_clients():
    """Initialize MemoBase and OpenAI clients"""
    mb_client = MemoBaseClient(
        project_url="http://localhost:8019",
        api_key="secret",
    )
    
    openai_client = OpenAI(
        api_key=os.getenv('llm_api_key'), 
        base_url="https://api.openai.com/v1/"
    )
    
    # Patch OpenAI client with memory
    openai_client = openai_memory(openai_client, mb_client, max_context_size=1000)
    
    return mb_client, openai_client

mb_client, openai_client = init_clients()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = "1"
if "agent" not in st.session_state:
    st.session_state.agent = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "user_id" not in st.session_state:
    st.session_state.user_id = "langkhachhoha"
if "assistant_name" not in st.session_state:
    st.session_state.assistant_name = "ViVi Assistant"

def get_mode_info(mode):
    """Get mode information"""
    modes = {
        "1": {
            "name": "OpenAI Chat",
            "description": "Original OpenAI with memory",
            "icon": "💬",
            "badge": "mode-openai",
            "tools": "None"
        },
        "2": {
            "name": "Agent (No RAG)",
            "description": "LangChain Agent without RAG",
            "icon": "🤖",
            "badge": "mode-agent",
            "tools": "search_event_profile"
        },
        "3": {
            "name": "Semantic Search",
            "description": "Vector search with Qdrant",
            "icon": "🔍",
            "badge": "mode-semantic",
            "tools": "search_event_profile, semantic_search"
        },
        "4": {
            "name": "Logical Reasoning",
            "description": "Tree-based search with PageIndex",
            "icon": "🧠",
            "badge": "mode-reasoning",
            "tools": "search_event_profile, reasoning_search"
        },
        "5": {
            "name": "Profile Search",
            "description": "Demo search_event_profile",
            "icon": "👤",
            "badge": "mode-agent",
            "tools": "search_event_profile"
        },
        "6": {
            "name": "PDF Viewer",
            "description": "View VinFast documentation PDFs",
            "icon": "📄",
            "badge": "mode-agent",
            "tools": "None"
        }
    }
    return modes.get(mode, modes["1"])

def display_pdf(pdf_path):
    """Display PDF file in Streamlit"""
    try:
        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        
        pdf_display = f'''
        <iframe src="data:application/pdf;base64,{base64_pdf}" 
                width="100%" 
                height="800" 
                type="application/pdf"
                style="border: none; border-radius: 10px;">
        </iframe>
        '''
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Error loading PDF: {str(e)}")

def get_available_pdfs():
    """Get list of available PDF files"""
    pdf_dir = Path(PDF_DIRECTORY)
    if not pdf_dir.exists():
        return []
    
    available_pdfs = []
    for pdf_file, title in PDF_FILES.items():
        pdf_path = pdf_dir / pdf_file
        if pdf_path.exists():
            available_pdfs.append((pdf_file, title, str(pdf_path)))
    
    return available_pdfs

def create_agent_for_mode(mode, user_id):
    """Create agent based on mode"""
    if mode in ["2", "3", "4"]:
        agent_kwargs = {
            "mb_client": mb_client,
            "llm_api_key": os.getenv('llm_api_key'),
            "llm_base_url": "https://api.openai.com/v1/",
            "model": MODEL,
            "max_profile_tokens": 1000,
            "temperature": 0.7,
        }
        
        if mode == "3":  # Semantic
            agent_kwargs.update({
                "rag_mode": "semantic",
                "qdrant_url": QDRANT_URL,
                "qdrant_api_key": QDRANT_API_KEY,
                "qdrant_collection_name": COLLECTION_NAME,
            })
        elif mode == "4":  # Reasoning
            doc_ids = [
                "pi-cmj8en8v103mw0dqx5nz1p9a2",
                "pi-cmj8enxzm03no0dqx054d1y93",
                "pi-cmj8eoruv03on0dqxxok4cf88",
                "pi-cmj8eplpc03p90dqxwrv0qpqe"
            ]
            agent_kwargs.update({
                "rag_mode": "reasoning",
                "pageindex_api_key": PAGEINDEX_API_KEY,
                "pageindex_doc_ids": doc_ids,
            })
        
        return create_memobase_agent(**agent_kwargs)
    return None

def stream_openai_response(prompt, user_id):
    """Stream response from OpenAI"""
    # Add to conversation history
    st.session_state.conversation_history.append({"role": "user", "content": prompt})
    
    # Keep only last BUFFER_SIZE messages
    if len(st.session_state.conversation_history) > BUFFER_SIZE * 2:
        st.session_state.conversation_history = st.session_state.conversation_history[-BUFFER_SIZE*2:]
    
    # Get streaming response
    response = openai_client.chat.completions.create(
        messages=st.session_state.conversation_history,
        model=MODEL,
        stream=True,
        user_id=user_id,
    )
    
    full_response = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_response += content
            yield content
    
    # Add to history
    st.session_state.conversation_history.append({"role": "assistant", "content": full_response})

def stream_agent_response(agent, prompt, user_id):
    """Stream response from agent"""
    for chunk in agent.chat_stream(user_id, prompt, verbose=False):
        yield chunk

# Sidebar
with st.sidebar:
    st.markdown(f"# 🚗 {st.session_state.assistant_name}")
    st.markdown("### Choose Your Mode")
    
    mode = st.radio(
        "Select demo mode:",
        ["1", "2", "3", "4", "5", "6"],
        format_func=lambda x: f"{get_mode_info(x)['icon']} {get_mode_info(x)['name']}",
        key="mode_selector"
    )
    
    # Check if mode changed
    if mode != st.session_state.mode:
        st.session_state.mode = mode
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.session_state.agent = create_agent_for_mode(mode, st.session_state.user_id)
        st.session_state.assistant_name = "ViVi Assistant"  # Reset to default name
        st.rerun()
    
    st.markdown("---")
    
    # Mode info
    mode_info = get_mode_info(mode)
    st.markdown(f"<div class='mode-badge {mode_info['badge']}'>{st.session_state.assistant_name}</div>", unsafe_allow_html=True)
    st.caption(f"{mode_info['icon']} {mode_info['name']}")
    
    st.markdown("---")
    
    # User info
    st.markdown("### 👤 User Settings")
    
    # User ID input (editable)
    new_user_id = st.text_input("User ID:", value=st.session_state.user_id, key="user_id_input")
    
    # Check if user ID changed
    if new_user_id != st.session_state.user_id:
        st.session_state.user_id = new_user_id
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.session_state.agent = create_agent_for_mode(st.session_state.mode, new_user_id)
        st.success(f"✅ Switched to user: {new_user_id}")
    
    # Assistant name input (editable)
    new_assistant_name = st.text_input("Assistant Name:", value=st.session_state.assistant_name, key="assistant_name_input")
    
    # Check if assistant name changed
    if new_assistant_name != st.session_state.assistant_name:
        st.session_state.assistant_name = new_assistant_name
        st.success(f"✅ Assistant renamed to: {new_assistant_name}")
    
    st.markdown("---")
    
    # Actions
    st.markdown("### ⚙️ Actions")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.rerun()
    
    with col2:
        if st.button("💾 Flush", use_container_width=True):
            if st.session_state.agent:
                st.session_state.agent.flush(st.session_state.user_id)
                st.session_state.agent.refresh_profile(st.session_state.user_id)
            else:
                openai_client.flush(st.session_state.user_id)
            st.success("✅ Flushed!")
    
    if st.button("📚 View Profile", use_container_width=True):
        with st.spinner("Loading profile..."):
            if st.session_state.agent:
                profile = st.session_state.agent.get_profile(st.session_state.user_id)
            else:
                profile = openai_client.get_memory_prompt(st.session_state.user_id)
            
            with st.expander("📋 Current Profile", expanded=True):
                st.text(profile if profile else "[No profile available]")
    
    st.markdown("---")
    st.caption("🔋 Powered by MemoBase + VinFast")

# Main content
mode_info = get_mode_info(st.session_state.mode)

# Header
col1, col2, col3 = st.columns([2, 3, 2])
with col2:
    st.markdown(f"# {st.session_state.assistant_name}")
    # st.markdown(f"<p style='text-align: center; color: var(--text-secondary);'>{mode_info['description']}</p>", unsafe_allow_html=True)

st.markdown("---")

# Special handling for mode 5 and 6
if st.session_state.mode == "6":
    # PDF Viewer Mode
    st.markdown("### 📄 VinFast Documentation PDFs")
    
    # Get available PDFs
    available_pdfs = get_available_pdfs()
    
    if not available_pdfs:
        st.warning("⚠️ No PDF files found in the specified directory.")
        st.info(f"📁 Looking for PDFs in: `{PDF_DIRECTORY}`")
    else:
        # Create tabs for different PDFs
        tab_names = [title for _, title, _ in available_pdfs]
        tabs = st.tabs(tab_names)
        
        for i, (pdf_file, title, pdf_path) in enumerate(available_pdfs):
            with tabs[i]:
                st.markdown(f"#### 📖 {title}")
                st.caption(f"File: {pdf_file}")
                
                # Display PDF
                display_pdf(pdf_path)

elif st.session_state.mode == "5":
    # Profile Search Demo
    st.markdown("### 👤 Search Event Profile Demo")
    
    query = st.text_input("Enter search query:", "My occupation is a software engineer")
    
    if st.button("🔍 Search", type="primary"):
        with st.spinner("Searching..."):
            u = mb_client.get_or_create_user(string_to_uuid(st.session_state.user_id))
            
            chats = [{"role": "user", "content": query}]
            
            # Run in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                search_future = executor.submit(u.search_event, query=query)
                profile_future = executor.submit(u.profile, chats=chats)
                
                search_result = search_future.result()
                profile_result = profile_future.result()
            
            # Format profile
            profile_string = "\n".join([
                f"{p.topic}::{p.sub_topic}::{p.content}" 
                for p in profile_result
            ])
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                with st.expander("📅 Events", expanded=True):
                    st.text(search_result if search_result else "[No events found]")
            
            with col2:
                with st.expander("👤 Profile", expanded=True):
                    st.text(profile_string if profile_string else "[No profile data]")



else:
    # Chat modes (1, 2, 3, 4)
    
    # Initialize agent if needed
    if st.session_state.mode in ["2", "3", "4"] and st.session_state.agent is None:
        st.session_state.agent = create_agent_for_mode(st.session_state.mode, st.session_state.user_id)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Message ViVi..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            if st.session_state.mode == "1":
                # OpenAI streaming
                for chunk in stream_openai_response(prompt, st.session_state.user_id):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
            else:
                # Agent streaming
                for chunk in stream_agent_response(st.session_state.agent, prompt, st.session_state.user_id):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
        
        # Add assistant response to messages
        st.session_state.messages.append({"role": "assistant", "content": full_response})


