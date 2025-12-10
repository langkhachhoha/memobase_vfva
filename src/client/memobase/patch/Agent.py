"""
LangChain Agent with MemoBase Memory Integration
Similar to openai.py but using LangChain framework with tools and memory
"""

import threading
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

from ..core.entry import MemoBaseClient, User, ChatBlob
from ..utils import string_to_uuid, LOG
from ..error import ServerError


SYSTEM_PROMPT_TEMPLATE = """You are a helpful AI assistant with access to personalized memory about the user.

# User Profile
{user_profile}

# Instructions
- Use the user's profile information to provide personalized responses
- You have access to a search_event_profile tool that can search for additional context
- Only use the tool when you need information that is not in the current profile
- The tool searches for relevant historical events and additional profile details
- Be natural and conversational, don't explicitly mention using tools unless relevant
- If the current profile has enough information to answer, respond directly without using tools

Remember: The tool is for searching additional context when needed, not for every query."""


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
    ):
        self.mb_client = mb_client
        self.max_profile_tokens = max_profile_tokens
        self.model = model
        
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
            Search for relevant events and profile information based on a query.
            Use this when you need additional context about the user's history or profile.
            
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
    
    def _get_user_profile_context(self, user: User) -> str:
        """Get user profile context for system prompt"""
        try:
            context = user.context(
                max_token_size=self.max_profile_tokens,
                prefer_topics=['work', 'basic_info', 'interests'],
                profile_event_ratio=1.0  # Only profile in system prompt
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
            search_tool = self._create_search_tool_for_user(user)
            llm_with_tools = self.llm.bind_tools([search_tool])
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
        
        # Create system message with profile
        system_msg = SystemMessage(content=SYSTEM_PROMPT_TEMPLATE.format(user_profile=user_profile))
        
        # Combine: system + history + new user message
        messages = [system_msg] + history.messages + [HumanMessage(content=user_message)]
        
        return messages
    
    def chat(self, user_id: str, message: str) -> str:
        """
        Send a message and get response
        
        Args:
            user_id: User identifier
            message: User's message
            
        Returns:
            AI assistant's response
        """
        user, history, llm_with_tools = self._get_or_create_user_data(user_id)
        
        # Format messages
        messages = self._format_messages_for_llm(user, history, message)
        
        try:
            # Get response from LLM
            response = llm_with_tools.invoke(messages)
            
            # Handle tool calls if any
            while response.tool_calls:
                # Execute tool calls
                tool_messages = []
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    
                    # Execute the search tool
                    search_tool = self._create_search_tool_for_user(user)
                    tool_result = search_tool.invoke(tool_args)
                    
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
                
                # Get next response
                response = llm_with_tools.invoke(messages)
            
            ai_response = response.content
            
            # Add to history
            history.add_user_message(message)
            history.add_ai_message(ai_response)
            
            # Save to MemoBase in background
            self._save_conversation_to_memobase(user, message, ai_response)
            
            return ai_response
            
        except Exception as e:
            LOG.error(f"Error in chat: {e}")
            return f"I encountered an error: {str(e)}"
    
    def chat_stream(self, user_id: str, message: str):
        """
        Stream response (generator)
        
        Args:
            user_id: User identifier
            message: User's message
            
        Yields:
            Chunks of AI response
        """
        user, history, llm_with_tools = self._get_or_create_user_data(user_id)
        
        # Format messages
        messages = self._format_messages_for_llm(user, history, message)
        
        full_response = ""
        
        try:
            # Stream response
            for chunk in llm_with_tools.stream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    content = chunk.content
                    full_response += content
                    yield content
            
            # Add to history
            history.add_user_message(message)
            history.add_ai_message(full_response)
            
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
    )
