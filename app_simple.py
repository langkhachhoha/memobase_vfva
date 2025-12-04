"""
Simple FastAPI Chat Demo with Memobase Memory
Clean interface with parsed memory display
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src" / "client"))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import os
from dotenv import load_dotenv
import asyncio
import re

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

openai_client = openai_memory(openai_client, mb_client, max_context_size=1000)

# Store conversation histories
user_conversations: Dict[str, List[Dict]] = {}
user_conversation_counts: Dict[str, int] = {}

# Initialize FastAPI
app = FastAPI(title="Memobase Chat Demo")

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
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            width: 100%;
            max-width: 800px;
            height: 90vh;
            display: flex;
            flex-direction: column;
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
        
        .action-buttons {
            display: flex;
            gap: 10px;
        }
        
        .action-buttons button {
            background: rgba(255,255,255,0.3);
            border: none;
            padding: 8px 15px;
            border-radius: 5px;
            color: white;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.3s;
        }
        
        .action-buttons button:hover {
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
            white-space: pre-wrap;
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
        
        .memory-display {
            background: #f0f2f6;
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
            max-width: 100%;
        }
        
        .memory-section {
            margin-bottom: 15px;
        }
        
        .memory-title {
            font-weight: bold;
            color: #667eea;
            margin-bottom: 8px;
            font-size: 16px;
        }
        
        .memory-item {
            margin-left: 15px;
            margin-bottom: 8px;
        }
        
        .memory-label {
            font-weight: 600;
            color: #555;
            font-size: 14px;
        }
        
        .memory-value {
            color: #333;
            margin-left: 5px;
            font-size: 14px;
        }
        
        .memory-list {
            list-style: none;
            margin-left: 20px;
        }
        
        .memory-list li {
            padding: 4px 0;
            color: #555;
            font-size: 13px;
        }
        
        .memory-list li:before {
            content: "• ";
            color: #667eea;
            font-weight: bold;
            margin-right: 5px;
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
            transition: transform 0.2s;
        }
        
        .input-container button:hover:not(:disabled) {
            transform: translateY(-2px);
        }
        
        .input-container button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Memobase Chat Demo</h1>
            <p>Interactive chat with long-term memory</p>
            <div class="user-info">
                <input type="text" id="userId" placeholder="Enter your user ID" value="langkhachhoha">
                <div class="action-buttons">
                    <button onclick="viewMemory()">📚 View Memory</button>
                    <button onclick="flushMemory()">💾 Flush Memory</button>
                </div>
            </div>
        </div>
        
        <div class="chat-container" id="chatContainer">
            <div class="system-message">
                👋 Welcome! Start chatting to see memory in action.
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
    
    <script>
        let isLoading = false;
        
        function addMessage(role, content, isSystem = false) {
            const chatContainer = document.getElementById('chatContainer');
            
            if (isSystem) {
                const systemDiv = document.createElement('div');
                systemDiv.className = 'system-message';
                systemDiv.innerHTML = content;
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
            
            addMessage('user', message);
            input.value = '';
            
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
                addMessage('assistant', data.response);
                
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
        
        function parseMemoryToHTML(memoryText) {
            if (!memoryText) return '<p>No memory data yet</p>';
            
            // Extract User Current Profile section
            const profileMatch = memoryText.match(/## User Current Profile:([\s\S]*?)(?=##|---)/);
            if (!profileMatch) return '<p>No profile data found</p>';
            
            const profileSection = profileMatch[1];
            const lines = profileSection.split('\\n').filter(line => line.trim().startsWith('-'));
            
            // Parse into categories
            const profile = {};
            lines.forEach(line => {
                const match = line.match(/- ([^:]+)::([^:]+):\\s*(.+?)(?:\\s*\\[mention[^\\]]*\\])?$/);
                if (match) {
                    const [, category, field, content] = match;
                    const cat = category.trim();
                    const fld = field.trim();
                    const val = content.trim();
                    
                    if (!profile[cat]) profile[cat] = {};
                    if (!profile[cat][fld]) profile[cat][fld] = [];
                    profile[cat][fld].push(val);
                }
            });
            
            // Build HTML
            const icons = {
                'psychological': '🧠',
                'work': '💼',
                'interest': '⭐',
                'contact_info': '📧',
                'basic_info': '👤',
                'education': '🎓',
                'skill': '🛠️',
                'hobby': '🎨',
                'goal': '🎯'
            };
            
            let html = '<div class="memory-display">';
            
            for (const [category, fields] of Object.entries(profile)) {
                const icon = icons[category.toLowerCase()] || '📌';
                const catName = category.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase());
                
                html += `<div class="memory-section">`;
                html += `<div class="memory-title">${icon} ${catName}</div>`;
                
                for (const [field, values] of Object.entries(fields)) {
                    const fieldName = field.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase());
                    html += `<div class="memory-item">`;
                    html += `<span class="memory-label">${fieldName}:</span>`;
                    
                    if (values.length === 1) {
                        // Single value
                        const val = values[0];
                        if (val.includes(';')) {
                            // Multi-line
                            html += `<ul class="memory-list">`;
                            val.split(';').forEach(part => {
                                if (part.trim()) {
                                    html += `<li>${escapeHtml(part.trim())}</li>`;
                                }
                            });
                            html += `</ul>`;
                        } else {
                            html += `<span class="memory-value">${escapeHtml(val)}</span>`;
                        }
                    } else {
                        // Multiple values
                        html += `<ul class="memory-list">`;
                        values.forEach(val => {
                            html += `<li>${escapeHtml(val)}</li>`;
                        });
                        html += `</ul>`;
                    }
                    
                    html += `</div>`;
                }
                
                html += `</div>`;
            }
            
            html += '</div>';
            return html;
        }
        
        async function viewMemory() {
            const userId = document.getElementById('userId').value.trim() || 'langkhachhoha';
            
            try {
                const response = await fetch(`/memory/${userId}`);
                const data = await response.json();
                
                if (data.memory) {
                    const parsedHTML = parseMemoryToHTML(data.memory);
                    addMessage('', `<strong>📚 Your Memory Profile:</strong>${parsedHTML}`, true);
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
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        document.getElementById('messageInput').focus();
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_CONTENT

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    user_id = message.user_id
    
    if user_id not in user_conversations:
        user_conversations[user_id] = []
        user_conversation_counts[user_id] = 0
        try:
            mb_client.get_or_create_user(string_to_uuid(user_id))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize user: {str(e)}")
    
    user_conversations[user_id].append({"role": "user", "content": message.message})
    
    if len(user_conversations[user_id]) > BUFFER_SIZE * 2:
        user_conversations[user_id] = user_conversations[user_id][-(BUFFER_SIZE * 2):]
    
    try:
        response = openai_client.chat.completions.create(
            messages=user_conversations[user_id],
            model=MODEL,
            stream=False,
            user_id=user_id,
        )
        
        assistant_message = response.choices[0].message.content
        user_conversations[user_id].append({"role": "assistant", "content": assistant_message})
        user_conversation_counts[user_id] += 1
        
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

@app.get("/memory/{user_id}")
async def get_memory(user_id: str):
    try:
        memory = openai_client.get_memory_prompt(user_id)
        return {"memory": memory or "", "user_id": user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve memory: {str(e)}")

@app.post("/flush/{user_id}")
async def flush_memory(user_id: str):
    try:
        await asyncio.sleep(0.1)
        openai_client.flush(user_id)
        return {"status": "success", "message": "Memory flushed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to flush memory: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "memobase-chat-demo"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

