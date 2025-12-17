# 🚗 ViVi - VinFast AI Assistant (Streamlit UI)

Modern, beautiful Streamlit interface for VinFast AI Assistant with 6 demo modes.

## ✨ Features

- **Beautiful Modern UI**: Dark theme with gradient colors and smooth animations
- **6 Demo Modes**: Choose between different AI configurations
- **Real Streaming**: True streaming responses from LLM
- **Memory Management**: Long-term memory with MemoBase
- **Profile View**: View and manage user profiles
- **Clean UX**: No technical messages, just smooth responses

## 🎯 6 Demo Modes

### 1. 💬 OpenAI Chat
Original OpenAI with memory integration
- **Tools**: None
- **Best for**: Simple conversations with memory

### 2. 🤖 Agent (No RAG)
LangChain Agent without RAG
- **Tools**: search_event_profile
- **Best for**: Personalized conversations using user history

### 3. 🔍 Semantic Search
Vector search with Qdrant
- **Tools**: search_event_profile, semantic_search
- **Best for**: Finding VinFast documentation by meaning and keywords

### 4. 🧠 Logical Reasoning
Tree-based search with PageIndex
- **Tools**: search_event_profile, reasoning_search
- **Best for**: Complex questions requiring step-by-step reasoning

### 5. ⚡ Single Query
Test with predefined queries
- **Tools**: search_event_profile
- **Best for**: Testing agent functionality

### 6. 👤 Profile Search
Demo search_event_profile function
- **Tools**: search_event_profile
- **Best for**: Understanding how profile search works

## 🚀 Quick Start

### Prerequisites

1. **MemoBase Server** must be running:
   ```bash
   cd src/server
   docker-compose up -d
   ```

2. **Environment Variables** in `.env`:
   ```
   llm_api_key=your_openai_api_key_here
   ```

3. **Config File** at `src/client/memobase/patch/config.py`:
   - Set your Qdrant credentials (for mode 3)
   - Set your PageIndex API key (for mode 4)

### Installation

```bash
# Activate conda environment
conda activate memobase_vivi

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

### Run the App

```bash
python run_local.py
```

The app will automatically:
- ✅ Check .env file
- ✅ Check MemoBase server
- ✅ Start Streamlit on http://localhost:8501

## 🎨 UI Features

### Sidebar
- **Mode Selection**: Choose from 6 demo modes
- **User Info**: Current user ID
- **Actions**: Clear chat, flush buffer, view profile

### Main Chat Area
- **Streaming Responses**: Real-time streaming from LLM
- **Clean Messages**: No technical tool messages
- **Beautiful Design**: Modern dark theme with gradients

### Special Modes
- **Mode 5**: Click predefined queries to test
- **Mode 6**: Search profile with custom query

## 🔧 Configuration

### Change User ID
Edit in `app.py`:
```python
USER_NAME = "your_user_id_here"
```

### Change Model
Edit in `app.py`:
```python
MODEL = "gpt-4o-mini"  # or "gpt-4", "gpt-3.5-turbo", etc.
```

### Change Buffer Size
Edit in `app.py`:
```python
BUFFER_SIZE = 5  # number of message pairs to keep in memory
```

## 📝 Usage Tips

1. **Start Simple**: Begin with Mode 1 or 2 to understand basic functionality
2. **Try RAG Modes**: Use Mode 3 for keyword-based search, Mode 4 for reasoning
3. **Flush Regularly**: Click "Flush" button to save conversations to long-term memory
4. **View Profile**: Click "View Profile" to see what the AI knows about you
5. **Clear Chat**: Use "Clear Chat" to start a fresh conversation

## 🎯 Examples

### Mode 3 (Semantic Search)
```
User: "tính năng ACC là gì?"
ViVi: [Searches VinFast docs and responds with ACC feature details]
```

### Mode 4 (Logical Reasoning)
```
User: "Khi nào không nên sử dụng hệ thống ACC?"
ViVi: [Uses reasoning to find when NOT to use ACC]
```

## 🐛 Troubleshooting

### Port Already in Use
If port 8501 is busy:
```bash
streamlit run app.py --server.port 8502
```

### MemoBase Server Not Running
```bash
cd src/server
docker-compose up -d
```

### Missing Dependencies
```bash
pip install streamlit langchain langchain-openai langchain-core
```

## 🌟 Features Highlights

- ✅ **Real Streaming**: Not fake character-by-character, true LLM streaming
- ✅ **Clean Output**: No tool messages in chat
- ✅ **Beautiful UI**: Modern dark theme with gradients
- ✅ **6 Modes**: Different AI configurations for different use cases
- ✅ **Memory Management**: Long-term memory with MemoBase
- ✅ **Responsive**: Works on desktop and mobile

## 📞 Support

For issues or questions, check:
- MemoBase docs: https://docs.memobase.io
- Streamlit docs: https://docs.streamlit.io

---

**Made with ❤️ for VinFast**
