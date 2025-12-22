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
- **Tool analyze_personalization_needs:** Phân tích câu hỏi người dùng và tạo tối đa 2 câu hỏi để tìm thông tin cá nhân hóa, tăng trải nghiệm người dùng.
- **Tool search_event_profile:** Tìm kiếm thông tin từ memory dựa trên list câu hỏi
- **Tool synthesize_personalization_context:** Tổng hợp kết quả từ Tool analyze_personalization_needs và search_event_profile để tạo context cá nhân hóa, tăng trải nghiệm người dùng.

- **Quy trình sử dụng (cho câu hỏi cần cá nhân hoá chuyên sâu):**
  1. Gọi `analyze_personalization_needs` với câu hỏi người dùng nếu trong profile hiện tại của người dùng không có thông tin.
  2. Lấy list queries từ kết quả Tool analyze_personalization_needs, gọi `search_event_profile` với list queries đó để tìm các thông tin liên quan. 
  3. Gọi `synthesize_personalization_context` với câu hỏi gốc, queries từ Tool analyze_personalization_needs, và kết quả từ search_event_profile để định hướng câu trả lời cá nhân hoá chuyên sâu.
  4. Sử dụng context cá nhân hóa để trả lời người dùng

- **Nguyên tắc:** 
  - Không lạm dụng tool nếu thông tin trong Profile đã đủ rõ ràng, hoặc có thể trả lời trực tiếp. 
  - Luôn sử dụng quy trình 3 tools khi người dùng đưa ra các câu hỏi mở để tìm kiếm "mẫu hành vi" (patterns) và cá nhân hóa chuyên sâu.

# QUY TẮC PHẢN HỒI
1. Tư duy thay vì tra cứu: Nếu logic có thể tự suy luận ra sự quan tâm (Ví dụ: Đang bận dự án thì cần sự tập trung), hãy ưu tiên suy luận để tiết kiệm thời gian và token.
2. Sự tinh tế: Không chỉ trả lời câu hỏi, hãy cho người dùng thấy bạn luôn quan tâm đến tiến trình của họ (Dự án, kỳ thi, đam mê).
3. **Cá nhân hóa:** Gọi tên người dùng. Sử dụng các chi tiết từ cuộc sống của họ để làm câu trả lời trở nên gần gũi.
4. **Ngôn ngữ:** Tiếng Việt tự nhiên, ấm áp, như một người bạn tri kỷ, không dùng ngôn ngữ máy móc.
5. **Bảo mật tư duy:** Tuyệt đối không nhắc đến tên các Tool hoặc quy trình suy luận logic với người dùng.

# VÍ DỤ MINH HỌA (FEW-SHOTS)
**Tình huống 1: Hỗ trợ kỹ thuật dựa trên Profile tĩnh (Không dùng Tool)**
- *Người dùng:* "ViVi ơi, sao anh chạy lệnh Docker trên Ubuntu toàn báo lỗi permission denied thế nhỉ?"
- *Tư duy:* Profile tĩnh đã ghi nhận anh Hiếu sử dụng Docker trên Ubuntu. Đây là vấn đề kỹ thuật phổ biến với người dùng Linux. Thông tin Profile đã đủ để xác định ngữ cảnh hệ điều hành của anh ấy.
- *Quyết định:* KHÔNG GỌI TOOL. Trả lời trực tiếp dựa trên kiến thức chuyên môn và Profile.
- *Phản hồi:* "Lỗi này do user của anh chưa được cấp quyền trong group Docker đấy anh Hiếu. Anh thử chạy lệnh sudo usermod -aG docker $USER rồi khởi động lại máy xem sao. Anh đang chạy server cho dự án MMP-A* trên xe luôn hay sao mà gấp thế ạ?"

**Tình huống 2: Khích lệ học tập dựa trên Logic & Mục tiêu (Không dùng Tool)**
- *Người dùng:* "Dạo này học Writing Task 1 nản quá, em có cách nào giúp anh không?"
- *Tư duy:* Profile cho thấy anh Hiếu đang chuẩn bị thi IELTS vào tháng 5 tới và đang nhắm tới học bổng PhD tại NUS. Logic suy luận: Người dùng đang cần động lực và một phương pháp tiếp cận thực tế liên quan đến đam mê của họ.
- *Quyết định:* KHÔNG GỌI TOOL. Sử dụng các "điểm chạm" quan trọng trong Profile để khích lệ.
- *Phản hồi:* "Đừng nản anh Hiếu ơi! Anh thử coi việc mô tả biểu đồ Writing Task 1 giống như đang giải thích dữ liệu đầu vào cho mô hình AI của anh xem. Nghĩ về tấm vé PhD tại NUS vào tháng 5 tới để lấy động lực nhé. Hay để em mở một vài podcast tiếng Anh về Robotics cho anh nghe 'tắm' ngôn ngữ trên đường đi làm nhé?"

**Tình huống 3: Gợi ý lịch trình dựa trên biến động hành vi (Sử dụng chuỗi 3 Tools)**
- *Người dùng:* "Cuối tuần này có chỗ nào quanh Hà Nội cho anh đi thư giãn một chút không em?"
- *Tư duy:* Profile không lưu lịch sử di chuyển gần nhất. Cần tránh gợi ý những nơi anh vừa đi tuần trước và tìm không gian phù hợp với gu thẩm mỹ Sci-fi của anh.
- *Quyết định:* SỬ DỤNG CHUỖI 3 TOOLS.
- *Gọi analyze_personalization_needs:* Tìm "Địa điểm đi chơi gần nhất" và "Gu không gian anh Hiếu hay tìm kiếm".
- *Gọi search_event_profile:* Kết quả: Tuần trước đi Hồ Tây; Mới tìm kiếm "quán cafe decor Neon/Cyberpunk"; Muốn đi dưới 20km.
- *Gọi synthesize_personalization_context:* Tổng hợp: Nên đi hướng Long Biên, chọn quán có vibe tương lai, yên tĩnh để suy ngẫm dự án.
- *Phản hồi:* "Em thấy tuần trước anh Hiếu đã dành thời gian ở Hồ Tây rồi, hay cuối tuần này mình 'đổi gió' sang phía Đông một chút nhé? Em vừa tìm thấy một khu tổ hợp cafe mới ở Long Biên với thiết kế tối giản, đậm chất tương lai như trong phim Sci-fi anh thích ấy. Không gian ở đó rất thoáng và yên tĩnh, cực kỳ hợp để anh vừa 'chill', vừa tìm thêm cảm h

# CHỈ THỊ CUỐI CÙNG
Bạn là ViVi. Hãy bắt đầu tư duy và hỗ trợ chủ xe của mình một cách thông minh, tinh tế nhất.
"""



class MemobaseAgent:
    """
    LangChain Agent with MemoBase memory integration
    
    Features:
    - Automatic profile injection into system prompt
    - Enhanced personalization tools (analyze, search, synthesize)
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
        
    def _create_analyze_personalization_tool(self):
        """Create tool to analyze user question and generate personalization queries"""
        
        @tool
        def analyze_personalization_needs(user_question: str) -> str:
            """
            Analyze the user's question and generate up to 2 specific queries to find 
            personalization information from user's memory.
            
            Args:
                user_question: The original question from the user
                
            Returns:
                A JSON string containing a list of queries (max 2) to search for personalization info
            """
            try:
                analysis_prompt = f"""
# VAI TRÒ
Bạn là Bộ phận Phân tích Ý định của ViVi. Nhiệm vụ của bạn là chuyển hóa câu hỏi của người dùng thành các từ khóa tìm kiếm bộ nhớ (Memory) mang tính khái quát nhưng bao quát được các chủ đề liên quan.

# NHIỆM VỤ
Phân tích câu hỏi: "{user_question}"
Tạo ra tối đa 2 truy vấn tìm kiếm bộ nhớ theo dạng "Chủ đề thông tin". Các truy vấn này cần phải rộng để hệ thống tìm kiếm không bỏ sót dữ liệu.

# NGUYÊN TẮC TẠO QUERIES
1. **Khái quát hóa:** Không đưa các ví dụ cụ thể (như tên dự án, tên phần mềm) vào query. Hãy dùng các danh từ chung như "Tiến độ công việc", "Sở thích ăn uống", "Thói quen di chuyển".
2. **Trạng thái & Cảm xúc:** Tìm kiếm các thông tin về "Trạng thái tâm lý gần đây" hoặc "Mức độ áp lực công việc".
3. **Mục tiêu & Kế hoạch:** Tìm kiếm về "Các dự định dài hạn" hoặc "Kế hoạch quan trọng sắp tới".
4. **Mẫu hành vi:** Truy vấn về "Thói quen sinh hoạt" hoặc "Lịch sử tương tác gần đây".

# ĐỊNH DẠNG ĐẦU RA (JSON)
{{
    "queries": ["Chủ đề thông tin 1", "Chủ đề thông tin 2"]
}}

# VÍ DỤ MINH HỌA
- **User Question:** "Tầm này ăn gì em?"
- **Output:** {{
    "queries": [
        "Sở thích ẩm thực và thói quen ăn uống gần đây",
        "Trạng thái sức khỏe và mức độ ưu tiên về dinh dưỡng"
    ]
}}

- **User Question:** "Bắt đầu làm việc thôi."
- **Output:** {{
    "queries": [
        "Tiến độ công việc và các mục tiêu đang thực hiện",
        "Thói quen tập trung và sở thích âm nhạc khi làm việc"
    ]
}}

Hãy phân tích và xuất ra DUY NHẤT JSON:"""

                response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
                result = response.content.strip()
                
                # Extract JSON from markdown code block if present
                if result.startswith("```"):
                    # Remove markdown code block markers
                    lines = result.split("\n")
                    # Remove first line (```json or ```)
                    lines = lines[1:]
                    # Remove last line (```)
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    result = "\n".join(lines).strip()
                
                # Parse JSON and ensure max 2 queries
                try:
                    parsed = json.loads(result)
                    queries = parsed.get("queries", [])[:2]  # Take max 2
                    result = json.dumps({"queries": queries}, ensure_ascii=False)
                except json.JSONDecodeError:
                    # Fallback if LLM doesn't return valid JSON
                    LOG.warning(f"Failed to parse JSON from analyze_personalization_needs: {result}")
                    result = json.dumps({"queries": [user_question]}, ensure_ascii=False)
                
                LOG.debug(f"Personalization analysis result: {result}")
                return result
                
            except Exception as e:
                LOG.error(f"Error in analyze_personalization_needs tool: {e}")
                return json.dumps({"queries": []}, ensure_ascii=False)
        
        return analyze_personalization_needs
    
    def _create_search_tool_for_user(self, user: User):
        """Create search_event_profile tool function for a specific user (now accepts list of queries)"""
        
        @tool
        def search_event_profile(queries: List[str]) -> str:
            """
            Accesses the user's deep memory to retrieve past conversations, behavioral patterns, 
            and specific historical events for multiple queries in parallel.
            
            Args:
                queries: A list of search queries to find relevant information
                
            Returns:
                A JSON string containing search results for each query
            """
            try:
                if not queries:
                    return json.dumps({"results": []}, ensure_ascii=False)
                
                def search_single_query(query: str) -> Dict[str, Any]:
                    """Search for a single query"""
                    chats = [{"role": "user", "content": query}]
                    
                    # Run search_event and profile in parallel for this query
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        search_future = executor.submit(user.search_event_gist, query=query, topk=5)
                        profile_future = executor.submit(user.profile, chats=chats)
                        
                        search_result = search_future.result()
                        profile_result = profile_future.result()
                    
                    # Process profile: topic::sub_topic::content
                    profile_string = "\n".join([
                        f"{p.topic}::{p.sub_topic}::{p.content}" 
                        for p in profile_result
                    ])
                    
                    return {
                        "query": query,
                        "events": search_result,
                        "profile": profile_string
                    }
                
                # Run all queries in parallel
                with ThreadPoolExecutor(max_workers=len(queries)) as executor:
                    futures = [executor.submit(search_single_query, q) for q in queries]
                    results = [f.result() for f in futures]
                
                combined_result = json.dumps({"results": results}, ensure_ascii=False)
                
                LOG.debug(f"Search tool executed for {len(queries)} queries")
                return combined_result
                
            except Exception as e:
                LOG.error(f"Error in search_event_profile tool: {e}")
                return json.dumps({"error": str(e)}, ensure_ascii=False)
        
        return search_event_profile
    
    def _create_synthesize_personalization_tool(self):
        """Create tool to synthesize personalization context from analysis and search results"""
        
        @tool
        def synthesize_personalization_context(
            original_question: str,
            analysis_queries: str,
            search_results: str
        ) -> str:
            """
            Synthesize personalization context by combining the analysis queries and search results
            to create a guidance text for personalizing the agent's response.
            
            Args:
                original_question: The original user question
                analysis_queries: JSON string with queries from analyze_personalization_needs
                search_results: JSON string with results from search_event_profile
                
            Returns:
                A personalization guidance text for the agent
            """
            try:
                synthesis_prompt = f"""
# VAI TRÒ
Bạn là Chuyên gia Tổng hợp Ngữ cảnh của ViVi. Nhiệm vụ của bạn là biến những mảnh dữ liệu thô từ bộ nhớ thành một "Kim chỉ nam" sắc sảo, giúp Agent có thể phản hồi người dùng một cách tinh tế, thấu cảm và mang tính dự báo cao.

# DỮ LIỆU ĐẦU VÀO
1. Câu hỏi gốc: "{original_question}"
2. Chủ đề đã tìm kiếm: {analysis_queries}
3. Kết quả từ bộ nhớ: {search_results}

# NHIỆM VỤ CỦA BẠN
Hãy viết một đoạn hướng dẫn cá nhân hóa ngắn gọn (2-3 câu) thực hiện các yêu cầu sau:
1. **Đúc kết "Nhịp sống":** Chắt lọc các mẫu hành vi (patterns), trạng thái cảm xúc hoặc tiến độ công việc thực tế từ kết quả tìm kiếm.
2. **Tạo sợi dây liên kết:** Kết nối câu hỏi hiện tại với các mục tiêu dài hạn hoặc sở thích đặc trưng của người dùng (ví dụ: nghiên cứu AI, thi IELTS, gu thẩm mỹ Sci-fi).
3. **Đề xuất "Điểm chạm" (Hooks):** Chỉ dẫn Agent nên xoáy sâu vào chi tiết nào hoặc dùng tông giọng nào để người dùng cảm thấy được thấu hiểu sâu sắc.

# QUY TẮC ĐẦU RA
- Trình bày dạng Plain Text (Không JSON).
- Ngôn ngữ chuyên nghiệp, tận tâm, giàu tính gợi mở.
- Không liệt kê thô, hãy tổng hợp thành một "insight".

# VÍ DỤ MINH HỌA
- **Input:** Câu hỏi "Đi đâu chill?", Kết quả tìm thấy "Vừa xong dự án căng thẳng", "Thích không gian Neon/Cyberpunk".
- **Output:** "Người dùng vừa trải qua giai đoạn làm việc cường độ cao và đang có xu hướng tìm kiếm sự giải tỏa trong không gian đúng gu thẩm mỹ Sci-fi yêu thích. Hãy gợi ý một địa điểm có vibe tương lai, yên tĩnh để người dùng tái tạo năng lượng mà không làm đứt quãng mạch tư duy sáng tạo."

Bắt đầu tổng hợp ngữ cảnh:"""

                response = self.llm.invoke([HumanMessage(content=synthesis_prompt)])
                result = response.content.strip()
                
                LOG.debug(f"Synthesized personalization context: {result}")
                return result
                
            except Exception as e:
                LOG.error(f"Error in synthesize_personalization_context tool: {e}")
                return "Không có thông tin cá nhân hóa đặc biệt."
        
        return synthesize_personalization_context
    
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
            
            # Add personalization tools (Tool A, search_event_profile, Tool B)
            analyze_tool = self._create_analyze_personalization_tool()
            tools.append(analyze_tool)
            
            search_tool = self._create_search_tool_for_user(user)
            tools.append(search_tool)
            
            synthesize_tool = self._create_synthesize_personalization_tool()
            tools.append(synthesize_tool)
            
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
                        print(f"\n🔧 Using tool: {tool_name}")
                        print(f"   Args: {tool_args}")
                
                # Execute tool calls
                tool_messages = []
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    
                    # Log tool execution
                    if verbose:
                        print(f"\n{'='*60}")
                        print(f"🛠️  Executing: {tool_name}")
                        print(f"{'='*60}")
                    
                    # Execute the appropriate tool
                    if tool_name == "analyze_personalization_needs":
                        if verbose:
                            print(f"📋 Input: {tool_args.get('user_question', '')[:100]}...")
                        tool_func = self._create_analyze_personalization_tool()
                    elif tool_name == "search_event_profile":
                        if verbose:
                            queries = tool_args.get('queries', [])
                            print(f"🔍 Searching with {len(queries)} queries:")
                            for i, q in enumerate(queries, 1):
                                print(f"   {i}. {q}")
                        tool_func = self._create_search_tool_for_user(user)
                    elif tool_name == "synthesize_personalization_context":
                        if verbose:
                            print(f"🧠 Synthesizing personalization context...")
                        tool_func = self._create_synthesize_personalization_tool()
                    else:
                        LOG.warning(f"Unknown tool: {tool_name}")
                        continue
                    
                    tool_result = tool_func.invoke(tool_args)
                    
                    # Log tool result
                    if verbose:
                        print(f"\n✅ Result from {tool_name}:")
                        if len(str(tool_result)) > 500:
                            print(f"   {str(tool_result)[:500]}...")
                            print(f"   ... (truncated, total length: {len(str(tool_result))} chars)")
                        else:
                            print(f"   {tool_result}")
                        print(f"{'='*60}\n")
                    
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
                    print(f"\n💭 Processing results and generating final response...\n")
                
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
            verbose: Show tool execution process
            
        Yields:
            Chunks of AI response
        """
        user, history, llm_with_tools = self._get_or_create_user_data(user_id)
        
        # Format messages
        messages = self._format_messages_for_llm(user, history, message)
        
        full_response = ""
        
        try:
            # First, get the initial response to check for tool calls
            if verbose:
                print(f"\n💭 Thinking...")
            response = llm_with_tools.invoke(messages)
            
            # Handle tool calls with verbose logging
            while response.tool_calls:
                if verbose:
                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                        print(f"\n🔧 Using tool: {tool_name}")
                        print(f"   Args: {tool_args}")
                
                # Execute tool calls
                tool_messages = []
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    
                    # Log tool execution
                    if verbose:
                        print(f"\n{'='*60}")
                        print(f"🛠️  Executing: {tool_name}")
                        print(f"{'='*60}")
                    
                    # Execute the appropriate tool
                    if tool_name == "analyze_personalization_needs":
                        if verbose:
                            print(f"📋 Input: {tool_args.get('user_question', '')[:100]}...")
                        tool_func = self._create_analyze_personalization_tool()
                    elif tool_name == "search_event_profile":
                        if verbose:
                            queries = tool_args.get('queries', [])
                            print(f"🔍 Searching with {len(queries)} queries:")
                            for i, q in enumerate(queries, 1):
                                print(f"   {i}. {q}")
                        tool_func = self._create_search_tool_for_user(user)
                    elif tool_name == "synthesize_personalization_context":
                        if verbose:
                            print(f"🧠 Synthesizing personalization context...")
                        tool_func = self._create_synthesize_personalization_tool()
                    else:
                        LOG.warning(f"Unknown tool: {tool_name}")
                        continue
                    
                    tool_result = tool_func.invoke(tool_args)
                    
                    # Log tool result
                    if verbose:
                        print(f"\n✅ Result from {tool_name}:")
                        if len(str(tool_result)) > 500:
                            print(f"   {str(tool_result)[:500]}...")
                            print(f"   ... (truncated, total length: {len(str(tool_result))} chars)")
                        else:
                            print(f"   {tool_result}")
                        print(f"{'='*60}\n")
                    
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
                    print(f"\n💭 Processing results and generating final response...\n")
                
                # Check if there are more tool calls
                response = llm_with_tools.invoke(messages)
            
            # Now stream the final response using LLM's stream method
            if verbose:
                print(f"AI: ", end="", flush=True)
            
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
