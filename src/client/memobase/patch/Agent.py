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
    * CHỈ gọi search_event_profile khi bạn muốn tạo ra sự bất ngờ hoặc gắn kết sâu sắc hơn bằng cách kết nối các hành vi/thói quen trong quá khứ mà các thông tin hiện tại chưa thể hiện rõ.
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



class MemobaseAgent:
    """
    LangChain Agent with MemoBase memory integration
    
    Features:
    - Automatic profile injection into system prompt
    - search_event_profile tool for dynamic memory retrieval
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
    ):
        self.mb_client = mb_client
        self.max_profile_tokens = max_profile_tokens
        self.model = model
        self.max_history_messages = max_history_messages
        
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
                    search_future = executor.submit(user.search_event_gist, query=query)
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
    
    def _get_user_profile_context(self, user: User) -> str:
        """Get user profile context for system prompt"""
        try:
            context = user.context(
                max_token_size=self.max_profile_tokens,
                prefer_topics=['basic_info', 'work', 'interests'],
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
        
        # Use default system prompt
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
    )
