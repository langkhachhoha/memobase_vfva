# Memobase

## Setup

### 1. Cài đặt dependencies

```bash
conda create -n memobase_vivi python=3.11
conda activate memobase_vivi
pip install -r requirements.txt
```

### 2. Tạo file .env trong src/server/

```bash
cd src/server
cp .env.example .env
```

### 3. Tạo file config.yaml trong src/server/api/

```bash
cd src/server/api
cp config.yaml.example config.yaml
```

Hoặc thủ công nếu dùng openai
```yaml
llm_api_key: XXX
llm_base_url: https://api.openai.com/v1/
best_llm_model: gpt-4o
```
Tuỳ chỉnh parameters trong config.yaml tuỳ context, tham khảo trong src/server/api/example_config

### 4. Khởi động server

```bash
cd src/server
docker-compose build && docker-compose up
```

### 5. Chạy demo local

```bash
python run_local.py
```

Access at: http://localhost:8000

### 6. Chạy Agent với RAG

Agent hỗ trợ 2 phương pháp RAG:
- **Option 1 (Semantic Search)**: Tìm kiếm dựa trên semantic similarity sử dụng Qdrant
- **Option 2 (Logical Reasoning)**: Tìm kiếm dựa trên logical reasoning sử dụng PageIndex

```bash
python main.py
```

Chọn mode:
- **Mode 1**: Interactive Chat (original OpenAI)
- **Mode 2**: Interactive Agent (không dùng RAG)
- **Mode 3**: Interactive Agent với Semantic Search (Qdrant)
- **Mode 4**: Interactive Agent với Logical Reasoning (PageIndex)
- **Mode 5**: Single Query Demo
- **Mode 6**: Search Event Profile Demo

#### Cấu hình RAG

Cấu hình RAG trong `src/client/memobase/patch/config.py`:

```python
# Semantic Search (Qdrant)
QDRANT_URL = "your_qdrant_url"
QDRANT_API_KEY = "your_qdrant_api_key"
COLLECTION_NAME = "vsf"

# Logical Reasoning (PageIndex)
PAGEINDEX_API_KEY = "your_pageindex_api_key"
```

#### Sự khác biệt giữa 2 phương pháp

**Semantic Search (Option 1)**:
- Input: Thông tin cần tìm kiếm (có thể là câu hỏi hoặc mô tả)
- Phương pháp: Vector similarity search
- Tốc độ: Nhanh
- Phù hợp: Tìm kiếm thông tin dựa trên ý nghĩa ngữ nghĩa

**Logical Reasoning (Option 2)**:
- Input: **BẮT BUỘC** phải là câu hỏi trực tiếp
- Phương pháp: Tree-based reasoning với LLM
- Tốc độ: Chậm hơn (do có bước reasoning)
- Phù hợp: Câu hỏi phức tạp cần suy luận logic



