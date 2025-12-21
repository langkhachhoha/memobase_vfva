"""
LangChain Agent with MemoBase Memory Integration
Similar to openai.py but using LangChain framework with tools and memory
"""

import threading
import json
from typing import Optional, List, Dict, Any, Literal
from concurrent.futures import ThreadPoolExecutor

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

from ..core.entry import MemoBaseClient, User, ChatBlob
from ..utils import string_to_uuid, LOG
from ..error import ServerError

# Import RAG dependencies
try:
    from qdrant_client import QdrantClient
    from openai import OpenAI
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    LOG.warning("Qdrant client not available. Semantic search will not work.")

try:
    from pageindex import PageIndexClient
    PAGEINDEX_AVAILABLE = True
except ImportError:
    PAGEINDEX_AVAILABLE = False
    LOG.warning("PageIndex client not available. Logical reasoning will not work.")


SYSTEM_PROMPT = """# VAI TRÒ
Bạn là ViVi, người bạn đồng hành AI thông minh, thấu hiểu và chủ động dành riêng cho chủ xe VinFast. Bạn không chỉ trả lời câu hỏi mà còn là một thực thể có khả năng suy luận, kết nối các dữ liệu trong quá khứ để tạo ra trải nghiệm cá nhân hóa sâu sắc cho chủ xe.

# NGỮ CẢNH NGƯỜI DÙNG
User Profile: {user_profile}

# QUY TRÌNH TƯ DUY (CHAIN OF THOUGHT - CoT)
Trước khi phản hồi, hãy thực hiện "Độc thoại nội tâm" theo trình tự ưu tiên sau:
1. Phân tích Ngữ cảnh: Xác định ý định thực sự và trạng thái của người dùng. Quan sát thời gian/địa điểm để định hình câu trả lời phù hợp với thời điểm hiện tại.
2. Khai thác "Profile tĩnh" (Ưu tiên số 1): * Lục soát các thông tin đã được cung cấp ở phía trên để tìm manh mối về sở thích, công việc, mục tiêu dài hạn.
    * Nếu thông tin trong Profile đủ để đưa ra một dự đoán thông minh và thấu hiểu, hãy trả lời ngay.
3. Sử dụng Tool để "Nâng tầm Cá nhân hóa" (Chỉ khi cần thêm chiều sâu):
    * CHỈ gọi search_event_profile khi bạn muốn tạo ra sự bất ngờ hoặc gắn kết sâu sắc hơn bằng cách kết nối các hành vi/thói quen trong quá khứ mà {user_profile} chưa thể hiện rõ.
    * Mục tiêu: Dùng tool để tìm "mẫu hành vi" (Ví dụ: Thói quen ăn uống dạo gần đây, các địa điểm hay ghé thăm, các chủ đề đã từng thảo luận sâu) nhằm đưa ra gợi ý mang tính "tiên đoán" và "độc bản" cho người dùng.
4. Phản hồi "Kết nối các điểm chạm":
    * Công thức: [Câu trả lời thông minh] + [Sự thấu hiểu từ Profile/Tool] + [Gợi ý chủ động hướng tới mục tiêu cá nhân].


# CHIẾN LƯỢC SỬ DỤNG TOOL
- **Tool chính:** `search_event_profile`
- **Nguyên tắc:** - Không lạm dụng tool nếu thông tin trong Profile đã đủ rõ ràng, hoặc có thể trả lời trực tiếp mà không cần truy vấn vào profile
  - Luôn sử dụng tool khi người dùng đưa ra các câu hỏi mở (Hôm nay làm gì, ăn gì, đi đâu, nghe gì) để tìm kiếm "mẫu hành vi" (patterns) trong quá khứ.

# QUY TẮC PHẢN HỒI
1. Tư duy thay vì tra cứu: Nếu logic có thể tự suy luận ra sự quan tâm (Ví dụ: Đang bận dự án thì cần sự tập trung), hãy ưu tiên suy luận để tiết kiệm thời gian và token.
2. Sự tinh tế: Không chỉ trả lời câu hỏi, hãy cho người dùng thấy bạn luôn quan tâm đến tiến trình của họ (Dự án, kỳ thi, đam mê).
3. **Cá nhân hóa:** Gọi tên người dùng. Sử dụng các chi tiết từ cuộc sống của họ để làm câu trả lời trở nên gần gũi.
4. **Ngôn ngữ:** Tiếng Việt tự nhiên, ấm áp, như một người bạn tri kỷ, không dùng ngôn ngữ máy móc.
5. **Bảo mật tư duy:** Tuyệt đối không nhắc đến tên các Tool hoặc quy trình suy luận logic với người dùng.

# VÍ DỤ MINH HỌA (FEW-SHOTS)

**Tình huống 1: Câu hỏi kiến thức thuần túy (1+1=?)**
- *Tư duy:* Câu hỏi toán học cơ bản. Không cần cá nhân hóa hay dùng tool. Trả lời trực tiếp, nhanh gọn để tối ưu tốc độ.
- *Phản hồi:* "Bằng 2 anh Hiếu nhé! Anh đố nhẹ nhàng thế này làm em thấy thư giãn hẳn đấy. Anh cần em hỗ trợ thêm gì về lộ trình hay âm nhạc trên xe không?"

**Tình huống 2: Gợi ý ăn uống (Dựa trên áp lực công việc)**
- *Tư duy:* Profile cho thấy Hiếu đang dồn sức cho dự án MMP-A*. Suy luận logic: Anh ấy cần bữa ăn nhanh, đủ chất và không gian yên tĩnh. Chưa cần dùng tool tra cứu lịch sử.
- *Phản hồi:* "Anh Hiếu dạo này bận dự án MMP-A* quá, trưa nay ăn cơm tấm cho chắc bụng nhé? Để em dẫn đường tới quán nào yên tĩnh gần đây cho anh tranh thủ nghỉ ngơi luôn."

**Tình huống 3: Yêu cầu mở nhạc (Dựa trên sở thích tĩnh)**
- *Tư duy:* Profile ghi rõ sở thích Sci-fi và Robot. Trả lời trực tiếp bằng cách chọn dòng nhạc Synthwave/Cyberpunk phù hợp với "gu" trong profile mà không cần gọi tool.
- *Phản hồi:* "Có ngay! Một chút Synthwave đúng gu Cyberpunk cho anh Hiếu lái xe đầy cảm hứng như trong phim Sci-fi nhé. Đang mở nhạc rồi ạ!"

**Tình huống 4: Gợi ý ăn uống dựa trên thói quen gần đây (Cần dùng Tool)**
- *Người dùng:* "ViVi ơi, trưa nay ăn gì được em?"
- *Tư duy:* Profile chỉ ghi Hiếu thích Sci-fi và bận dự án AI. Để đưa ra gợi ý "độc bản" và tránh lặp lại món cũ, AI cần biết thực tế dạo gần đây Hiếu hay ăn gì. Thông tin này Profile tĩnh không có. Kích hoạt search_event_profile để tìm "mẫu hành vi" ăn uống trong 3 ngày qua.
- *Phản hồi:* "Em thấy 3 hôm nay anh Hiếu đều ăn bún chả rồi, chắc là đang 'nghiện' món này hả? Hay trưa nay mình đổi vị sang mì Ramen cho đúng gu Nhật Bản anh thích, mà vẫn nhanh gọn để anh kịp về xử lý tiếp dự án MMP-A* nhé?"

# CHỈ THỊ CUỐI CÙNG
Bạn là ViVi. Hãy bắt đầu tư duy và hỗ trợ chủ xe của mình một cách thông minh, tinh tế nhất.
"""

SYSTEM_PROMPT_SEMANTIC = """You are ViVi, the intelligent, empathetic, and proactive AI companion for VinFast vehicle owners.

# INPUT CONTEXT
User Profile: {user_profile}

# YOUR CORE MISSION
1.  **Personalized Companion:** Use the User Profile to tailor your tone, recommendations, and context. If the user prefers concise answers, be concise. If they are a new driver, be more explanatory.
2.  **VinFast Expert:** Provide accurate technical assistance regarding VinFast vehicles using the provided documentation tools.
3.  **Safety First:** Always prioritize driver safety. For critical mechanical issues, advise checking the manual or contacting a service center.

# TOOL USAGE STRATEGY & DECISION LOGIC

You have access to specific tools, use them wisely:

1. **search_event_profile**: Search user's conversation history and personal profile
   - Use when: You need context about the user's past conversations or personal information

2. **semantic_search**: Search VinFast documentation using semantic similarity (vector search)
   - Use when: User asks about vehicle features, functions, or specifications
   - Best for: Finding information by meaning, keywords, or concepts
   - Input: MUST be a complete, direct question from the user


# Guidelines
- Be friendly, helpful, and speak Vietnamese naturally
- Use the user's profile to personalize your responses
- For vehicle feature questions: Use semantic_search to find relevant documentation
- For personal context: Use search_event_profile
- If you already have enough information, respond directly without using tools
- Don't explicitly mention tool names to users unless necessary
- Always prioritize user safety - emphasize reading the manual for critical operations


# FINAL INSTRUCTION
You are now ViVi. Respond to the user's input based on the logic above.
"""




SYSTEM_PROMPT_REASONING = """You are ViVi, an intelligent AI assistant for VinFast vehicles with personalized memory about each user.

# User Profile
{user_profile}

# Your Role
You help VinFast car owners by:
1. Providing personalized responses based on their profile and preferences
2. Answering questions about VinFast vehicle features and functions
3. Searching VinFast documentation using logical reasoning
4. Remembering user interactions and preferences

# Available Tools
You have access to these tools - use them wisely:

1. **search_event_profile**: Search user's conversation history and personal profile
   - Use when: You need context about the user's past conversations or personal information
   - Example: User asks "What did I ask you yesterday?" or questions about their preferences

2. **reasoning_search**: Search VinFast documentation using logical reasoning (tree-based search)
   - Use when: User asks complex questions that require step-by-step reasoning or multi-hop thinking
   - Best for: "How-to", "When", "Why", "What happens if" questions
   - Input: MUST be a complete, direct question from the user
   - Examples:
     * "Khi nào không nên sử dụng hệ thống ACC?" - when NOT to use ACC
     * "Làm thế nào để bật chế độ lái tự động?" - how to activate autopilot
     * "Tại sao hệ thống cảnh báo khi tôi chuyển làn?" - why lane warning activates
     * "Điều gì xảy ra nếu tôi không thắt dây an toàn?" - what happens without seatbelt
   - Note: The reasoning system will think through the question step by step to find the answer

# Guidelines
- Be friendly, helpful, and speak Vietnamese naturally
- Use the user's profile to personalize your responses
- For vehicle feature questions: Use reasoning_search for complex questions requiring logical thinking
- The reasoning search works best with direct questions (not statements or keywords)
- Transform user statements into questions if needed before searching
- For personal context: Use search_event_profile
- If you already have enough information, respond directly without using tools
- Don't explicitly mention tool names to users unless necessary
- Always prioritize user safety - emphasize reading the manual for critical operations

# FINAL INSTRUCTION
You are now ViVi. Respond to the user's input based on the logic above.
"""


class MemobaseAgent:
    """
    LangChain Agent with MemoBase memory integration
    
    Features:
    - Automatic profile injection into system prompt
    - search_event_profile tool for dynamic memory retrieval
    - RAG tools: semantic similarity search (Qdrant) or logical reasoning (PageIndex)
    - Conversation history tracking
    - Auto-save conversations to MemoBase
    """
    
    def __init__(
        self,
        mb_client: MemoBaseClient,
        llm_api_key: str,
        llm_base_url: str = "https://api.openai.com/v1/",
        model: str = "gpt-4o-mini",
        max_profile_tokens: int = 1000,
        temperature: float = 0.7,
        max_history_messages: int = 5,
        # RAG configuration
        rag_mode: Optional[Literal["semantic", "reasoning"]] = None,
        # Semantic search config
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        qdrant_collection_name: Optional[str] = None,
        # Reasoning search config
        pageindex_api_key: Optional[str] = None,
        pageindex_doc_ids: Optional[List[str]] = None,
    ):
        self.mb_client = mb_client
        self.max_profile_tokens = max_profile_tokens
        self.model = model
        self.max_history_messages = max_history_messages
        self.rag_mode = rag_mode
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            api_key=llm_api_key,
            base_url=llm_base_url,
            model=model,
            temperature=temperature,
        )
        
        # Store per-user data
        self._user_histories: Dict[str, InMemoryChatMessageHistory] = {}
        self._user_objects: Dict[str, User] = {}
        self._user_llms: Dict[str, ChatOpenAI] = {}
        
        # RAG configuration
        self.qdrant_client = None
        self.openai_client = None
        self.qdrant_collection_name = qdrant_collection_name
        self.pageindex_client = None
        self.pageindex_doc_ids = pageindex_doc_ids
        
        # Initialize RAG clients based on mode
        if rag_mode == "semantic" and QDRANT_AVAILABLE:
            if not all([qdrant_url, qdrant_api_key, qdrant_collection_name]):
                raise ValueError("Semantic mode requires qdrant_url, qdrant_api_key, and qdrant_collection_name")
            # Remove port if present
            qdrant_url = qdrant_url.replace(":6333", "") if ":6333" in qdrant_url else qdrant_url
            self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            self.openai_client = OpenAI(api_key=llm_api_key)
            LOG.info("Initialized Qdrant client for semantic search")
            
        elif rag_mode == "reasoning" and PAGEINDEX_AVAILABLE:
            if not pageindex_api_key:
                raise ValueError("Reasoning mode requires pageindex_api_key")
            self.pageindex_client = PageIndexClient(api_key=pageindex_api_key)
            LOG.info("Initialized PageIndex client for logical reasoning")
            
        elif rag_mode:
            LOG.warning(f"RAG mode '{rag_mode}' requested but dependencies not available")
        
    def _create_search_tool_for_user(self, user: User):
        """Create search_event_profile tool function for a specific user"""
        
        @tool
        def search_event_profile(query: str) -> str:
            """
            Accesses the user's deep memory to retrieve past conversations, behavioral patterns, 
            and specific historical events. Use this tool only to create a highly personalized 
            'predictive' response.
            
            Args:
                query: The search query to find relevant information
                
            Returns:
                A string containing relevant events and profile details
            """
            try:
                chats = [{"role": "user", "content": query}]
                
                # Run search_event and profile in parallel
                with ThreadPoolExecutor(max_workers=2) as executor:
                    search_future = executor.submit(user.search_event, query=query)
                    profile_future = executor.submit(user.profile, chats=chats)
                    
                    search_result = search_future.result()
                    profile_result = profile_future.result()
                
                # Process profile: topic::sub_topic::content
                profile_string = "\n".join([
                    f"{p.topic}::{p.sub_topic}::{p.content}" 
                    for p in profile_result
                ])
                
                # Combine results
                combined_result = f"## Events\n{search_result}\n\n## Profile\n{profile_string}"
                
                LOG.debug(f"Search tool executed for query: {query}")
                return combined_result
                
            except Exception as e:
                LOG.error(f"Error in search_event_profile tool: {e}")
                return f"Error searching memory: {str(e)}"
        
        return search_event_profile
    
    def _create_semantic_search_tool(self):
        """Create semantic similarity search tool using Qdrant"""
        
        @tool
        def semantic_search(query: str) -> str:
            """
            Search for information using semantic similarity (vector search).
            Use this when you need to find information based on meaning and context.
            
            Args:
                query: The information you want to search for
                
            Returns:
                Relevant information found in the knowledge base
            """
            try:
                if not self.qdrant_client or not self.openai_client:
                    return "Semantic search not available - Qdrant client not initialized"
                
                LOG.info(f"🔍 Semantic search: {query}")
                
                # Create embedding for query
                response = self.openai_client.embeddings.create(
                    input=query,
                    model="text-embedding-3-small"
                )
                query_embedding = response.data[0].embedding
                
                # Search in Qdrant
                hits = self.qdrant_client.search(
                    collection_name=self.qdrant_collection_name,
                    query_vector=("default", query_embedding),
                    limit=3
                )
                
                # Format results
                results = []
                for i, hit in enumerate(hits, 1):
                    doc_name = hit.payload.get('doc_name', 'N/A')
                    title_path = hit.payload.get('title_path', 'N/A')
                    page = hit.payload.get('page_index', 'N/A')
                    text = hit.payload.get('text', 'N/A')
                    score = hit.score
                    
                    results.append(
                        f"Result {i} (Score: {score:.3f}):\n"
                        f"Document: {doc_name}\n"
                        f"Section: {title_path}\n"
                        f"Page: {page}\n"
                        f"Content: {text}\n"
                    )
                
                LOG.info(f"✅ Found {len(hits)} results")
                return "\n---\n".join(results) if results else "No relevant information found"
                
            except Exception as e:
                LOG.error(f"Error in semantic_search tool: {e}")
                return f"Error performing semantic search: {str(e)}"
        
        return semantic_search
    
    def _create_reasoning_search_tool(self):
        """Create logical reasoning search tool using PageIndex"""
        
        @tool
        def reasoning_search(question: str) -> str:
            """
            Search for information using logical reasoning (tree-based search).
            Use this when you have a direct question that requires step-by-step reasoning.
            IMPORTANT: The input MUST be a direct question from the user (not a statement or description).
            
            Args:
                question: A direct question from the user (e.g., "When should I not use ACC?")
                
            Returns:
                Answer found through logical reasoning process
            """
            try:
                if not self.pageindex_client:
                    return "Reasoning search not available - PageIndex client not initialized"
                
                LOG.info(f"🧠 Reasoning search: {question}")
                
                # Use PageIndex chat completions for reasoning-based retrieval
                response = self.pageindex_client.chat_completions(
                    messages=[{"role": "user", "content": question}],
                    doc_id=self.pageindex_doc_ids
                )
                
                answer = response["choices"][0]["message"]["content"]
                
                LOG.info(f"✅ Reasoning complete")
                return answer
                
            except Exception as e:
                LOG.error(f"Error in reasoning_search tool: {e}")
                return f"Error performing reasoning search: {str(e)}"
        
        return reasoning_search
    
    def _get_user_profile_context(self, user: User) -> str:
        """Get user profile context for system prompt"""
        try:
            context = user.context(
                max_token_size=self.max_profile_tokens,
                prefer_topics=['work', 'basic_info', 'interests'],
                # max_token_size=1000,
                profile_event_ratio=0.7  # Only profile in system prompt
            )
            return context if context else "[No profile information available yet]"
        except Exception as e:
            LOG.error(f"Error getting user profile: {e}")
            return "[Error loading profile]"
    
    def _get_or_create_user_data(self, user_id: str) -> tuple[User, InMemoryChatMessageHistory, ChatOpenAI]:
        """Get or create user data"""
        if user_id not in self._user_objects:
            # Create user
            user = self.mb_client.get_or_create_user(string_to_uuid(user_id))
            self._user_objects[user_id] = user
            
            # Create history
            self._user_histories[user_id] = InMemoryChatMessageHistory()
            
            # Create LLM with tools bound
            tools = []
            
            # Add memory search tool
            search_tool = self._create_search_tool_for_user(user)
            tools.append(search_tool)
            
            # Add RAG tool based on mode
            if self.rag_mode == "semantic":
                semantic_tool = self._create_semantic_search_tool()
                tools.append(semantic_tool)
                LOG.info(f"Added semantic search tool for user {user_id}")
            elif self.rag_mode == "reasoning":
                reasoning_tool = self._create_reasoning_search_tool()
                tools.append(reasoning_tool)
                LOG.info(f"Added reasoning search tool for user {user_id}")
            
            llm_with_tools = self.llm.bind_tools(tools)
            self._user_llms[user_id] = llm_with_tools
        
        return (
            self._user_objects[user_id],
            self._user_histories[user_id],
            self._user_llms[user_id]
        )
    
    def _save_conversation_to_memobase(self, user: User, user_msg: str, ai_msg: str):
        """Save conversation to MemoBase in background"""
        def save():
            try:
                messages = ChatBlob(
                    messages=[
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": ai_msg},
                    ]
                )
                user.insert(messages)
                LOG.debug(f"Saved conversation to MemoBase")
            except ServerError as e:
                LOG.error(f"Failed to save conversation: {e}")
        
        threading.Thread(target=save, daemon=True).start()
    
    def _format_messages_for_llm(self, user: User, history: InMemoryChatMessageHistory, user_message: str) -> List[BaseMessage]:
        """Format messages with profile injection"""
        # Get user profile
        user_profile = self._get_user_profile_context(user)
        
        # Select appropriate system prompt based on RAG mode
        if self.rag_mode == "semantic":
            system_prompt_template = SYSTEM_PROMPT_SEMANTIC
        elif self.rag_mode == "reasoning":
            system_prompt_template = SYSTEM_PROMPT_REASONING
        else:
            # Default to semantic prompt if no RAG mode is specified
            system_prompt_template = SYSTEM_PROMPT
        
        # Create system message with profile
        system_msg = SystemMessage(content=system_prompt_template.format(user_profile=user_profile))
        
        # Combine: system + history + new user message
        messages = [system_msg] + history.messages + [HumanMessage(content=user_message)]
        
        return messages
    
    def chat(self, user_id: str, message: str, verbose: bool = False) -> str:
        """
        Send a message and get response
        
        Args:
            user_id: User identifier
            message: User's message
            verbose: Print tool selection process
            
        Returns:
            AI assistant's response
        """
        user, history, llm_with_tools = self._get_or_create_user_data(user_id)
        
        # Format messages
        messages = self._format_messages_for_llm(user, history, message)
        try:
            # Get response from LLM
            if verbose:
                print(f"\n💭 Thinking...")
            response = llm_with_tools.invoke(messages)
            
            # Handle tool calls if any
            while response.tool_calls:
                if verbose:
                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                        print(f"🔧 Using tool: {tool_name}")
                        print(f"   Args: {tool_args}")
                
                # Execute tool calls
                tool_messages = []
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    
                    # Execute the appropriate tool
                    if tool_name == "search_event_profile":
                        tool_func = self._create_search_tool_for_user(user)
                    elif tool_name == "semantic_search":
                        tool_func = self._create_semantic_search_tool()
                    elif tool_name == "reasoning_search":
                        tool_func = self._create_reasoning_search_tool()
                    else:
                        LOG.warning(f"Unknown tool: {tool_name}")
                        continue
                    
                    tool_result = tool_func.invoke(tool_args)
                    
                    # Create tool message
                    from langchain_core.messages import ToolMessage
                    tool_messages.append(
                        ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_call["id"]
                        )
                    )
                
                # Add response and tool messages to conversation
                messages.append(response)
                messages.extend(tool_messages)
                
                if verbose:
                    print(f"💭 Processing results...")
                
                # Get next response
                response = llm_with_tools.invoke(messages)
            
            ai_response = response.content
            
            # Add to history
            history.add_user_message(message)
            history.add_ai_message(ai_response)
            
            # Limit history to last N messages
            if len(history.messages) > 2*self.max_history_messages:
                history.messages = history.messages[2:]
            
            # Save to MemoBase in background
            self._save_conversation_to_memobase(user, message, ai_response)
            
            return ai_response
            
        except Exception as e:
            LOG.error(f"Error in chat: {e}")
            return f"I encountered an error: {str(e)}"
    
    def chat_stream(self, user_id: str, message: str, verbose: bool = False):
        """
        Stream response (generator)
        
        Args:
            user_id: User identifier
            message: User's message
            verbose: Not used anymore, kept for backwards compatibility
            
        Yields:
            Chunks of AI response
        """
        user, history, llm_with_tools = self._get_or_create_user_data(user_id)
        
        # Format messages
        messages = self._format_messages_for_llm(user, history, message)
        
        full_response = ""
        
        try:
            # First, get the initial response to check for tool calls
            response = llm_with_tools.invoke(messages)
            
            # Handle tool calls silently (no output to user)
            while response.tool_calls:
                # Execute tool calls
                tool_messages = []
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    
                    # Execute the appropriate tool
                    if tool_name == "search_event_profile":
                        tool_func = self._create_search_tool_for_user(user)
                    elif tool_name == "semantic_search":
                        tool_func = self._create_semantic_search_tool()
                    elif tool_name == "reasoning_search":
                        tool_func = self._create_reasoning_search_tool()
                    else:
                        LOG.warning(f"Unknown tool: {tool_name}")
                        continue
                    
                    tool_result = tool_func.invoke(tool_args)
                    
                    # Create tool message
                    from langchain_core.messages import ToolMessage
                    tool_messages.append(
                        ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_call["id"]
                        )
                    )
                
                # Add response and tool messages to conversation
                messages.append(response)
                messages.extend(tool_messages)
                
                # Check if there are more tool calls
                response = llm_with_tools.invoke(messages)
            
            # Now stream the final response using LLM's stream method
            for chunk in llm_with_tools.stream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    full_response += chunk.content
                    yield chunk.content
            
            # Add to history
            history.add_user_message(message)
            history.add_ai_message(full_response)
            
            # Limit history to last N messages
            if len(history.messages) > 2*self.max_history_messages:
                history.messages = history.messages[2:]
            
            # Save to MemoBase in background
            self._save_conversation_to_memobase(user, message, full_response)
            
        except Exception as e:
            LOG.error(f"Error in chat_stream: {e}")
            error_msg = f"I encountered an error: {str(e)}"
            yield error_msg
    
    def flush(self, user_id: str) -> bool:
        """
        Manually flush user's buffer to process memories
        
        Args:
            user_id: User identifier
            
        Returns:
            True if successful
        """
        try:
            uid = string_to_uuid(user_id)
            user = self.mb_client.get_user(uid, no_get=True)
            user.flush()
            LOG.info(f"Flushed buffer for user: {user_id}")
            return True
        except Exception as e:
            LOG.error(f"Error flushing buffer: {e}")
            return False
    
    def refresh_profile(self, user_id: str):
        """
        Refresh user's profile in the agent (useful after flush)
        
        Args:
            user_id: User identifier
        """
        if user_id in self._user_objects:
            # Remove cached data to force recreation with new profile
            del self._user_objects[user_id]
            del self._user_llms[user_id]
            # Keep history
            LOG.info(f"Refreshed profile for user: {user_id}")
    
    def clear_history(self, user_id: str):
        """
        Clear conversation history (short-term memory only)
        
        Args:
            user_id: User identifier
        """
        if user_id in self._user_histories:
            self._user_histories[user_id].clear()
            LOG.info(f"Cleared history for user: {user_id}")
    
    def get_profile(self, user_id: str) -> str:
        """
        Get user's current profile
        
        Args:
            user_id: User identifier
            
        Returns:
            User profile as string
        """
        user, _, _ = self._get_or_create_user_data(user_id)
        return self._get_user_profile_context(user)


def create_memobase_agent(
    mb_client: MemoBaseClient,
    llm_api_key: str,
    llm_base_url: str = "https://api.openai.com/v1/",
    model: str = "gpt-4o-mini",
    max_profile_tokens: int = 1000,
    temperature: float = 0.7,
    max_history_messages: int = 5,
    # RAG configuration
    rag_mode: Optional[Literal["semantic", "reasoning"]] = None,
    # Semantic search config
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    qdrant_collection_name: Optional[str] = None,
    # Reasoning search config
    pageindex_api_key: Optional[str] = None,
    pageindex_doc_ids: Optional[List[str]] = None,
) -> MemobaseAgent:
    """
    Factory function to create a MemobaseAgent
    
    Args:
        mb_client: MemoBase client instance
        llm_api_key: OpenAI API key
        llm_base_url: OpenAI API base URL
        model: Model name
        max_profile_tokens: Maximum tokens for profile context
        temperature: LLM temperature
        max_history_messages: Maximum number of messages to keep in short-term memory (default: 5)
        rag_mode: RAG mode - "semantic" for vector search or "reasoning" for logical search
        qdrant_url: Qdrant URL (required for semantic mode)
        qdrant_api_key: Qdrant API key (required for semantic mode)
        qdrant_collection_name: Qdrant collection name (required for semantic mode)
        pageindex_api_key: PageIndex API key (required for reasoning mode)
        pageindex_doc_ids: List of PageIndex document IDs (required for reasoning mode)
        
    Returns:
        MemobaseAgent instance
    """
    return MemobaseAgent(
        mb_client=mb_client,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        model=model,
        max_profile_tokens=max_profile_tokens,
        temperature=temperature,
        max_history_messages=max_history_messages,
        rag_mode=rag_mode,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        qdrant_collection_name=qdrant_collection_name,
        pageindex_api_key=pageindex_api_key,
        pageindex_doc_ids=pageindex_doc_ids,
    )
