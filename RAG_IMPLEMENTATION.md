# RAG Implementation Summary

## Tổng quan

Đã implement 2 phương pháp RAG vào Agent:
1. **Semantic Search**: Sử dụng Qdrant vector database
2. **Logical Reasoning**: Sử dụng PageIndex tree-based reasoning

## Các file đã thay đổi

### 1. `src/client/memobase/patch/Agent.py`

#### Thêm dependencies:
- `qdrant_client` cho semantic search
- `pageindex` cho logical reasoning
- Import `OpenAI` client để tạo embeddings

#### Thay đổi class `MemobaseAgent`:

**Constructor (`__init__`)**:
- Thêm parameter `rag_mode`: `None`, `"semantic"`, hoặc `"reasoning"`
- Thêm các parameters cho Qdrant: `qdrant_url`, `qdrant_api_key`, `qdrant_collection_name`
- Thêm các parameters cho PageIndex: `pageindex_api_key`, `pageindex_doc_ids`
- Khởi tạo Qdrant client và PageIndex client dựa trên mode

**Thêm 2 methods mới**:

1. `_create_semantic_search_tool()`: 
   - Tool để tìm kiếm bằng semantic similarity
   - Input: `query` - thông tin cần tìm kiếm
   - Process:
     - Tạo embedding cho query bằng OpenAI API
     - Search trong Qdrant collection
     - Trả về top 3 kết quả với score, document, section, page, content

2. `_create_reasoning_search_tool()`:
   - Tool để tìm kiếm bằng logical reasoning
   - Input: `question` - **BẮT BUỘC phải là câu hỏi trực tiếp**
   - Process:
     - Gọi PageIndex API để thực hiện tree-based reasoning
     - Trả về câu trả lời đã được reasoning

**Cập nhật methods hiện có**:

- `_get_or_create_user_data()`: Bind RAG tools vào LLM dựa trên `rag_mode`
- `chat()`: Thêm parameter `verbose` để in quá trình suy nghĩ và lựa chọn tool
- `chat_stream()`: Thêm parameter `verbose` để in quá trình trong streaming mode

### 2. `src/client/memobase/patch/config.py`

Thêm config:
```python
COLLECTION_NAME = "vsf"  # VinFast collection
PAGEINDEX_API_KEY = "0584cae7327e46e88a2356e7d6343bce"
```

### 3. `main.py`

**Cập nhật `demo_agent_interactive()`**:
- Thêm parameter `rag_mode` để chọn phương pháp RAG
- Load config từ `config.py`
- Khởi tạo agent với RAG configuration phù hợp
- Thêm `verbose=True` khi gọi `chat()` hoặc `chat_stream()` để hiển thị quá trình

**Cập nhật menu chính**:
```
1. Interactive Chat (original OpenAI)
2. Interactive Agent (LangChain, no RAG)
3. Interactive Agent (Option 1: Semantic Search - Qdrant)
4. Interactive Agent (Option 2: Logical Reasoning - PageIndex)
5. Single Query Demo (Agent)
6. Search Event Profile Demo
```

### 4. `readme.md`

Thêm documentation về:
- Cách sử dụng RAG
- Cấu hình RAG
- Sự khác biệt giữa 2 phương pháp

## Cách sử dụng

### Option 1: Semantic Search (Qdrant)

**Khi nào dùng:**
- Tìm kiếm thông tin dựa trên ý nghĩa ngữ nghĩa
- Cần tốc độ nhanh
- Input có thể là câu hỏi hoặc mô tả

**Ví dụ:**
```python
agent = create_memobase_agent(
    mb_client=mb_client,
    llm_api_key=api_key,
    rag_mode="semantic",
    qdrant_url=QDRANT_URL,
    qdrant_api_key=QDRANT_API_KEY,
    qdrant_collection_name="vsf",
)

# Query có thể là câu hỏi hoặc mô tả
response = agent.chat(user_id, "tính năng ACC", verbose=True)
```

**Quá trình hoạt động:**
1. User gửi query
2. Agent nhận diện cần tìm kiếm thông tin
3. Agent gọi `semantic_search` tool
4. Tool tạo embedding cho query
5. Tool search trong Qdrant
6. Trả về top 3 kết quả relevance nhất
7. Agent tổng hợp và trả lời

### Option 2: Logical Reasoning (PageIndex)

**Khi nào dùng:**
- Câu hỏi phức tạp cần suy luận logic
- Cần hiểu cấu trúc tài liệu
- **Input BẮT BUỘC phải là câu hỏi trực tiếp**

**Ví dụ:**
```python
agent = create_memobase_agent(
    mb_client=mb_client,
    llm_api_key=api_key,
    rag_mode="reasoning",
    pageindex_api_key=PAGEINDEX_API_KEY,
    pageindex_doc_ids=["doc1", "doc2", "doc3"],
)

# Query BẮT BUỘC phải là câu hỏi trực tiếp
response = agent.chat(user_id, "Khi nào không nên sử dụng ACC?", verbose=True)
```

**Quá trình hoạt động:**
1. User gửi câu hỏi
2. Agent nhận diện cần reasoning
3. Agent gọi `reasoning_search` tool
4. Tool gửi câu hỏi đến PageIndex API
5. PageIndex thực hiện tree-based reasoning:
   - Duyệt cấu trúc document tree
   - Tìm các nodes liên quan
   - Reasoning qua các nodes
   - Tổng hợp câu trả lời
6. Trả về câu trả lời đã reasoning
7. Agent format và trả lời user

## Verbose Mode

Khi `verbose=True`, Agent sẽ in ra:
```
💭 Thinking...                    # Agent đang suy nghĩ
🔧 Using tool: semantic_search    # Agent chọn tool
   Args: {'query': '...'}         # Arguments của tool
💭 Processing results...          # Agent đang xử lý kết quả
```

## Lưu ý

1. **Không dùng đồng thời 2 tools**: Chỉ chọn 1 trong 2 mode khi khởi tạo agent
2. **Semantic search**: Input linh hoạt (câu hỏi hoặc description)
3. **Reasoning search**: Input BẮT BUỘC phải là câu hỏi trực tiếp
4. **Dependencies**: 
   - Semantic mode cần: `qdrant-client`, `openai`
   - Reasoning mode cần: `pageindex`
5. **Performance**:
   - Semantic search: Nhanh (~1-2s)
   - Reasoning search: Chậm hơn (~5-10s do có bước reasoning)

## Testing

Chạy agent:
```bash
python main.py
```

Chọn mode 3 hoặc 4 để test RAG.

**Test queries cho Semantic Search:**
- "tính năng ACC hoạt động như thế nào"
- "các biển báo giao thông"
- "hệ thống nhận dạng biển báo"

**Test queries cho Logical Reasoning:**
- "Khi nào không nên sử dụng ACC?"
- "Tính năng Chuyển làn tự động sẽ tắt khi nào?"
- "Làm thế nào để bật ACC?"
