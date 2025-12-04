"""
FastAPI Demo Application for Memobase + OpenAI Memory
Provides a web interface for the interactive chat with memory functionality
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src" / "client"))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
from dotenv import load_dotenv
import asyncio
import json

from memobase import MemoBaseClient
from openai import OpenAI
from memobase.patch.openai import openai_memory
from memobase.utils import string_to_uuid

load_dotenv()

# Configuration
MODEL = "gpt-4o-mini"
BUFFER_SIZE = 5

# Initialize clients
openai_client = OpenAI(
    api_key=os.getenv('llm_api_key'), 
    base_url="https://api.openai.com/v1/"
)

mb_client = MemoBaseClient(
    project_url="http://localhost:8019",
    api_key="secret",
)

# Patch OpenAI client with memory capability
openai_client = openai_memory(openai_client, mb_client, max_context_size=1000)

# Store conversation histories per user
user_conversations: Dict[str, List[Dict]] = {}
user_conversation_counts: Dict[str, int] = {}

# Initialize FastAPI app
app = FastAPI(
    title="Memobase Chat Demo",
    description="Interactive chat with long-term memory using Memobase",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    user_id: str = "langkhachhoha"

class ChatResponse(BaseModel):
    response: str
    conversation_count: int
    auto_flushed: bool = False

class MemoryResponse(BaseModel):
    memory: str
    user_id: str

class UserProfile(BaseModel):
    user_id: str
    profile: dict

# HTML Frontend
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Memobase Chat Demo</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .main-layout {
            display: flex;
            gap: 20px;
            width: 100%;
            max-width: 1400px;
            height: 90vh;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            flex: 1;
            min-width: 600px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .profile-panel {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            width: 400px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 30px;
            border-radius: 20px 20px 0 0;
        }
        
        .header h1 {
            font-size: 24px;
            margin-bottom: 5px;
        }
        
        .header p {
            font-size: 14px;
            opacity: 0.9;
        }
        
        .user-info {
            background: rgba(255,255,255,0.2);
            padding: 10px 15px;
            border-radius: 10px;
            margin-top: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .user-info input {
            background: rgba(255,255,255,0.3);
            border: none;
            padding: 8px 12px;
            border-radius: 5px;
            color: white;
            font-size: 14px;
            width: 200px;
        }
        
        .user-info input::placeholder {
            color: rgba(255,255,255,0.7);
        }
        
        .user-info button {
            background: rgba(255,255,255,0.3);
            border: none;
            padding: 8px 15px;
            border-radius: 5px;
            color: white;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.3s;
        }
        
        .user-info button:hover {
            background: rgba(255,255,255,0.4);
        }
        
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 20px 30px;
            background: #f8f9fa;
        }
        
        .message {
            margin-bottom: 20px;
            display: flex;
            animation: slideIn 0.3s ease;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .message.user {
            justify-content: flex-end;
        }
        
        .message-content {
            max-width: 70%;
            padding: 12px 18px;
            border-radius: 18px;
            line-height: 1.5;
        }
        
        .message.user .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-bottom-right-radius: 4px;
        }
        
        .message.assistant .message-content {
            background: white;
            color: #333;
            border: 1px solid #e0e0e0;
            border-bottom-left-radius: 4px;
        }
        
        .system-message {
            text-align: center;
            color: #666;
            font-size: 13px;
            margin: 15px 0;
            padding: 8px;
            background: rgba(102, 126, 234, 0.1);
            border-radius: 8px;
        }
        
        .input-container {
            padding: 20px 30px;
            background: white;
            border-top: 1px solid #e0e0e0;
            display: flex;
            gap: 10px;
        }
        
        .input-container input {
            flex: 1;
            padding: 12px 18px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 15px;
            outline: none;
            transition: border-color 0.3s;
        }
        
        .input-container input:focus {
            border-color: #667eea;
        }
        
        .input-container button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 15px;
            font-weight: 600;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .input-container button:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .input-container button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        
        .loading {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #667eea;
            animation: pulse 1.5s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 0.3; }
            50% { opacity: 1; }
        }
        
        .action-buttons {
            display: flex;
            gap: 10px;
        }
        
        .action-buttons button {
            padding: 8px 15px;
            font-size: 13px;
        }
        
        /* Profile Panel Styles */
        .profile-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 20px 20px 0 0;
        }
        
        .profile-header h2 {
            font-size: 20px;
            margin-bottom: 5px;
        }
        
        .profile-header p {
            font-size: 13px;
            opacity: 0.9;
        }
        
        .profile-content {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
        }
        
        .profile-empty {
            text-align: center;
            padding: 40px 20px;
            color: #999;
        }
        
        .profile-empty-icon {
            font-size: 48px;
            margin-bottom: 10px;
        }
        
        .profile-section {
            background: white;
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        
        .profile-section-title {
            font-size: 14px;
            font-weight: 600;
            color: #667eea;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .profile-item {
            padding: 8px 0;
            border-bottom: 1px solid #f0f0f0;
        }
        
        .profile-item:last-child {
            border-bottom: none;
        }
        
        .profile-label {
            font-size: 12px;
            color: #999;
            margin-bottom: 4px;
        }
        
        .profile-value {
            font-size: 14px;
            color: #333;
            font-weight: 500;
        }
        
        .profile-badge {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            margin: 4px 4px 4px 0;
        }
        
        .profile-list {
            list-style: none;
            padding: 0;
        }
        
        .profile-list li {
            padding: 6px 0;
            font-size: 14px;
            color: #555;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .profile-list li:before {
            content: "•";
            color: #667eea;
            font-weight: bold;
            font-size: 18px;
        }
        
        .profile-refresh {
            text-align: center;
            padding: 10px;
            background: white;
            border-top: 1px solid #e0e0e0;
        }
        
        .profile-refresh button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 600;
            transition: transform 0.2s;
        }
        
        .profile-refresh button:hover {
            transform: translateY(-2px);
        }
        
        .profile-loading {
            text-align: center;
            padding: 20px;
            color: #999;
        }
        
        @media (max-width: 1200px) {
            .main-layout {
                flex-direction: column;
                height: auto;
            }
            
            .profile-panel {
                width: 100%;
                max-height: 400px;
            }
            
            .container {
                min-width: auto;
            }
        }
    </style>
</head>
<body>
    <div class="main-layout">
        <div class="container">
        <div class="header">
            <h1>🤖 Memobase Chat Demo</h1>
            <p>Interactive chat with long-term memory • Buffer Size: 5 messages</p>
            <div class="user-info">
                <input type="text" id="userId" placeholder="Enter your user ID" value="langkhachhoha">
                <div class="action-buttons">
                    <button onclick="viewMemory()">📚 View Memory</button>
                    <button onclick="flushMemory()">💾 Flush Memory</button>
                    <button onclick="refreshProfile()">👤 Refresh Profile</button>
                </div>
            </div>
        </div>
        
        <div class="chat-container" id="chatContainer">
            <div class="system-message">
                👋 Welcome! Start chatting to see memory in action. Memory auto-saves every 5 turns.
            </div>
        </div>
        
        <div class="input-container">
            <input 
                type="text" 
                id="messageInput" 
                placeholder="Type your message here..." 
                onkeypress="handleKeyPress(event)"
            >
            <button onclick="sendMessage()" id="sendButton">Send</button>
        </div>
    </div>
    
    <!-- Profile Panel -->
    <div class="profile-panel">
        <div class="profile-header">
            <h2>👤 User Profile</h2>
            <p>Real-time profile information</p>
        </div>
        
        <div class="profile-content" id="profileContent">
            <div class="profile-empty">
                <div class="profile-empty-icon">📋</div>
                <p>No profile data yet</p>
                <p style="font-size: 12px; margin-top: 8px;">Start chatting to build your profile</p>
            </div>
        </div>
        
        <div class="profile-refresh">
            <button onclick="refreshProfile()">🔄 Refresh Profile</button>
        </div>
    </div>
</div>
    
    <script>
        let isLoading = false;
        let profileRefreshInterval = null;
        
        // Start auto-refresh on load
        window.addEventListener('load', () => {
            refreshProfile();
            // Auto-refresh every 10 seconds
            profileRefreshInterval = setInterval(refreshProfile, 10000);
        });
        
        function addMessage(role, content, isSystem = false) {
            const chatContainer = document.getElementById('chatContainer');
            
            if (isSystem) {
                const systemDiv = document.createElement('div');
                systemDiv.className = 'system-message';
                systemDiv.textContent = content;
                chatContainer.appendChild(systemDiv);
            } else {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${role}`;
                
                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';
                contentDiv.textContent = content;
                
                messageDiv.appendChild(contentDiv);
                chatContainer.appendChild(messageDiv);
            }
            
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        async function sendMessage() {
            if (isLoading) return;
            
            const input = document.getElementById('messageInput');
            const userId = document.getElementById('userId').value.trim() || 'langkhachhoha';
            const message = input.value.trim();
            
            if (!message) return;
            
            // Add user message
            addMessage('user', message);
            input.value = '';
            
            // Disable input
            isLoading = true;
            document.getElementById('sendButton').disabled = true;
            document.getElementById('sendButton').textContent = 'Thinking...';
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        message: message,
                        user_id: userId
                    })
                });
                
                if (!response.ok) {
                    throw new Error('Failed to get response');
                }
                
                const data = await response.json();
                
                // Add assistant message
                addMessage('assistant', data.response);
                
                // Show auto-flush message if it happened
                if (data.auto_flushed) {
                    addMessage('', `💾 Memory auto-saved! (Total turns: ${data.conversation_count})`, true);
                }
                
            } catch (error) {
                console.error('Error:', error);
                addMessage('', '❌ Error: Failed to get response', true);
            } finally {
                isLoading = false;
                document.getElementById('sendButton').disabled = false;
                document.getElementById('sendButton').textContent = 'Send';
                input.focus();
            }
        }
        
        async function viewMemory() {
            const userId = document.getElementById('userId').value.trim() || 'langkhachhoha';
            
            try {
                const response = await fetch(`/memory/${userId}`);
                const data = await response.json();
                
                if (data.memory) {
                    addMessage('', `📚 Long-term Memory:\n${data.memory}`, true);
                } else {
                    addMessage('', '📚 No long-term memory stored yet', true);
                }
            } catch (error) {
                console.error('Error:', error);
                addMessage('', '❌ Error: Failed to retrieve memory', true);
            }
        }
        
        async function flushMemory() {
            const userId = document.getElementById('userId').value.trim() || 'langkhachhoha';
            
            try {
                const response = await fetch(`/flush/${userId}`, {
                    method: 'POST'
                });
                
                if (response.ok) {
                    addMessage('', '✅ Memory flushed successfully!', true);
                } else {
                    throw new Error('Failed to flush memory');
                }
            } catch (error) {
                console.error('Error:', error);
                addMessage('', '❌ Error: Failed to flush memory', true);
            }
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter' && !isLoading) {
                sendMessage();
            }
        }
        
        async function refreshProfile() {
            const userId = document.getElementById('userId').value.trim() || 'langkhachhoha';
            const profileContent = document.getElementById('profileContent');
            
            try {
                // Fetch both profile and memory
                const [profileRes, memoryRes] = await Promise.all([
                    fetch(`/profile/${userId}`),
                    fetch(`/memory/${userId}`)
                ]);
                
                if (!profileRes.ok && !memoryRes.ok) {
                    throw new Error('Failed to fetch profile');
                }
                
                let parsedProfile = {};
                
                // Try to get memory first (more structured)
                if (memoryRes.ok) {
                    const memoryData = await memoryRes.json();
                    if (memoryData.memory) {
                        parsedProfile = parseMemoryToProfile(memoryData.memory);
                    }
                }
                
                // Fallback to profile API if memory is empty
                if (Object.keys(parsedProfile).length === 0 && profileRes.ok) {
                    const profileData = await profileRes.json();
                    parsedProfile = profileData.profile || {};
                }
                
                displayProfile(parsedProfile);
                
            } catch (error) {
                console.error('Error fetching profile:', error);
                profileContent.innerHTML = `
                    <div class="profile-empty">
                        <div class="profile-empty-icon">❌</div>
                        <p>Failed to load profile</p>
                    </div>
                `;
            }
        }
        
        function parseMemoryToProfile(memoryText) {
            if (!memoryText) return {};
            
            const profile = {};
            
            // Extract User Current Profile section
            const profileMatch = memoryText.match(/## User Current Profile:([\s\S]*?)(?=##|---|\n\n$|$)/);
            if (!profileMatch) return {};
            
            const profileSection = profileMatch[1];
            
            // Parse each line with format: - category::field: content [mention date]
            const lines = profileSection.split('\n').filter(line => line.trim().startsWith('-'));
            
            lines.forEach(line => {
                // Match pattern: - category::field: content [mention date]
                const match = line.match(/- ([^:]+)::([^:]+):\s*(.+?)(?:\s*\[mention[^\]]*\])?$/);
                if (match) {
                    const [, category, field, content] = match;
                    const categoryKey = category.trim();
                    const fieldKey = field.trim();
                    const value = content.trim();
                    
                    // Initialize category if not exists
                    if (!profile[categoryKey]) {
                        profile[categoryKey] = {};
                    }
                    
                    // Add or append to field
                    if (profile[categoryKey][fieldKey]) {
                        // If field already exists, make it an array
                        if (!Array.isArray(profile[categoryKey][fieldKey])) {
                            profile[categoryKey][fieldKey] = [profile[categoryKey][fieldKey]];
                        }
                        profile[categoryKey][fieldKey].push(value);
                    } else {
                        profile[categoryKey][fieldKey] = value;
                    }
                }
            });
            
            return profile;
        }
        
        function displayProfile(profile) {
            const profileContent = document.getElementById('profileContent');
            
            // Check if profile is empty
            if (!profile || Object.keys(profile).length === 0) {
                profileContent.innerHTML = `
                    <div class="profile-empty">
                        <div class="profile-empty-icon">📋</div>
                        <p>No profile data yet</p>
                        <p style="font-size: 12px; margin-top: 8px;">Start chatting to build your profile</p>
                    </div>
                `;
                return;
            }
            
            let html = '';
            
            // Category icons mapping
            const categoryIcons = {
                'psychological': '🧠',
                'work': '💼',
                'interest': '⭐',
                'contact_info': '📧',
                'basic_info': '👤',
                'education': '🎓',
                'skill': '🛠️',
                'hobby': '🎨',
                'goal': '🎯',
                'experience': '📚'
            };
            
            // Render each profile section
            for (const [key, value] of Object.entries(profile)) {
                if (!value || (Array.isArray(value) && value.length === 0)) continue;
                
                const icon = categoryIcons[key.toLowerCase()] || '📌';
                html += `<div class="profile-section">`;
                html += `<div class="profile-section-title">${icon} ${formatKey(key)}</div>`;
                
                if (Array.isArray(value)) {
                    // Render as list
                    html += `<ul class="profile-list">`;
                    value.forEach(item => {
                        if (typeof item === 'object') {
                            html += `<li>${JSON.stringify(item)}</li>`;
                        } else {
                            const itemStr = String(item);
                            // Split by semicolon for multi-line content
                            if (itemStr.includes(';')) {
                                const parts = itemStr.split(';').map(p => p.trim()).filter(p => p);
                                parts.forEach(part => {
                                    html += `<li style="font-size: 13px;">${escapeHtml(part)}</li>`;
                                });
                            } else {
                                html += `<li>${escapeHtml(itemStr)}</li>`;
                            }
                        }
                    });
                    html += `</ul>`;
                } else if (typeof value === 'object') {
                    // Render nested object
                    for (const [subKey, subValue] of Object.entries(value)) {
                        html += `<div class="profile-item">`;
                        html += `<div class="profile-label">${formatKey(subKey)}</div>`;
                        
                        if (Array.isArray(subValue)) {
                            // Multiple values - render as list
                            html += `<ul class="profile-list" style="margin-top: 5px;">`;
                            subValue.forEach(item => {
                                const itemStr = String(item);
                                if (itemStr.includes(';')) {
                                    const parts = itemStr.split(';').map(p => p.trim()).filter(p => p);
                                    parts.forEach(part => {
                                        html += `<li style="font-size: 13px;">${escapeHtml(part)}</li>`;
                                    });
                                } else {
                                    html += `<li>${escapeHtml(itemStr)}</li>`;
                                }
                            });
                            html += `</ul>`;
                        } else {
                            // Single value
                            const valueStr = String(subValue);
                            if (valueStr.includes(';')) {
                                // Multi-line content
                                const parts = valueStr.split(';').map(p => p.trim()).filter(p => p);
                                html += `<ul class="profile-list" style="margin-top: 5px;">`;
                                parts.forEach(part => {
                                    html += `<li style="font-size: 13px;">${escapeHtml(part)}</li>`;
                                });
                                html += `</ul>`;
                            } else {
                                html += `<div class="profile-value">${escapeHtml(valueStr)}</div>`;
                            }
                        }
                        html += `</div>`;
                    }
                } else {
                    // Render simple value
                    const valueStr = String(value);
                    if (valueStr.includes(';')) {
                        const parts = valueStr.split(';').map(p => p.trim()).filter(p => p);
                        html += `<ul class="profile-list">`;
                        parts.forEach(part => {
                            html += `<li style="font-size: 13px;">${escapeHtml(part)}</li>`;
                        });
                        html += `</ul>`;
                    } else {
                        html += `<div class="profile-value">${escapeHtml(valueStr)}</div>`;
                    }
                }
                
                html += `</div>`;
            }
            
            if (html === '') {
                profileContent.innerHTML = `
                    <div class="profile-empty">
                        <div class="profile-empty-icon">📋</div>
                        <p>No profile data yet</p>
                    </div>
                `;
            } else {
                profileContent.innerHTML = html;
            }
        }
        
        function formatKey(key) {
            // Convert snake_case or camelCase to Title Case
            return key
                .replace(/_/g, ' ')
                .replace(/([A-Z])/g, ' $1')
                .split(' ')
                .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
                .join(' ')
                .trim();
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // Focus input on load
        document.getElementById('messageInput').focus();
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the chat interface"""
    return HTML_CONTENT

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Handle chat messages with memory"""
    user_id = message.user_id
    
    # Initialize user conversation history if not exists
    if user_id not in user_conversations:
        user_conversations[user_id] = []
        user_conversation_counts[user_id] = 0
        # Ensure user exists in Memobase
        try:
            mb_client.get_or_create_user(string_to_uuid(user_id))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize user: {str(e)}")
    
    # Add user message to conversation history
    user_conversations[user_id].append({"role": "user", "content": message.message})
    
    # Keep only last BUFFER_SIZE * 2 messages (sliding window)
    if len(user_conversations[user_id]) > BUFFER_SIZE * 2:
        user_conversations[user_id] = user_conversations[user_id][-(BUFFER_SIZE * 2):]
    
    try:
        # Get response from OpenAI with memory
        response = openai_client.chat.completions.create(
            messages=user_conversations[user_id],
            model=MODEL,
            stream=False,
            user_id=user_id,
        )
        
        assistant_message = response.choices[0].message.content
        
        # Add assistant response to conversation history
        user_conversations[user_id].append({"role": "assistant", "content": assistant_message})
        
        # Increment conversation count
        user_conversation_counts[user_id] += 1
        
        # Auto-flush after every BUFFER_SIZE turns
        auto_flushed = False
        if user_conversation_counts[user_id] % BUFFER_SIZE == 0:
            await asyncio.sleep(0.1)
            openai_client.flush(user_id)
            auto_flushed = True
        
        return ChatResponse(
            response=assistant_message,
            conversation_count=user_conversation_counts[user_id],
            auto_flushed=auto_flushed
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.get("/memory/{user_id}", response_model=MemoryResponse)
async def get_memory(user_id: str):
    """Get long-term memory for a user"""
    try:
        memory = openai_client.get_memory_prompt(user_id)
        return MemoryResponse(memory=memory or "", user_id=user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve memory: {str(e)}")

@app.post("/flush/{user_id}")
async def flush_memory(user_id: str):
    """Manually flush memory buffer for a user"""
    try:
        await asyncio.sleep(0.1)
        openai_client.flush(user_id)
        return {"status": "success", "message": "Memory flushed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to flush memory: {str(e)}")

@app.get("/profile/{user_id}", response_model=UserProfile)
async def get_user_profile(user_id: str):
    """Get user profile from Memobase"""
    try:
        user = mb_client.get_or_create_user(string_to_uuid(user_id))
        profile = user.profile(need_json=True)
        return UserProfile(user_id=user_id, profile=profile)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get profile: {str(e)}")

@app.delete("/conversation/{user_id}")
async def clear_conversation(user_id: str):
    """Clear conversation history for a user"""
    if user_id in user_conversations:
        user_conversations[user_id] = []
        user_conversation_counts[user_id] = 0
    return {"status": "success", "message": "Conversation cleared"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "memobase-chat-demo"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

