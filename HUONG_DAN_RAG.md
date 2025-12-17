# Hướng dẫn sử dụng RAG trong Agent

## 🎯 Tổng quan nhanh

Agent hiện có **2 phương pháp RAG** để tìm kiếm thông tin:

| Phương pháp | Input | Tốc độ | Phù hợp cho |
|------------|-------|--------|-------------|
| **Option 1: Semantic Search** | Bất kỳ (câu hỏi/mô tả) | ⚡ Nhanh | Tìm kiếm theo ngữ nghĩa |
| **Option 2: Logical Reasoning** | ⚠️ Chỉ câu hỏi trực tiếp | 🐢 Chậm hơn | Câu hỏi phức tạp cần suy luận |

## 🚀 Cách chạy

```bash
python main.py
```

Chọn mode:
- **Mode 3**: Option 1 - Semantic Search (Qdrant)
- **Mode 4**: Option 2 - Logical Reasoning (PageIndex)

## 📝 Ví dụ sử dụng

### Option 1: Semantic Search

✅ **Input linh hoạt - có thể là:**
- "tính năng ACC"
- "Tìm thông tin về hệ thống ACC"
- "ACC hoạt động như thế nào?"
- "các biển báo giao thông"

```
You: tính năng ACC hoạt động như thế nào

💭 Thinking...
🔧 Using tool: semantic_search
   Args: {'query': 'tính năng ACC hoạt động như thế nào'}
💭 Processing results...
AI: Hệ thống ACC sử dụng radar và camera...
```

### Option 2: Logical Reasoning

⚠️ **Input BẮT BUỘC phải là câu hỏi trực tiếp:**
- ✅ "Khi nào không nên sử dụng ACC?"
- ✅ "Tính năng Chuyển làn tự động sẽ tắt khi nào?"
- ✅ "Làm thế nào để bật ACC?"
- ❌ "tính năng ACC" (không phải câu hỏi)
- ❌ "Tìm thông tin về ACC" (không hỏi trực tiếp)

```
You: Khi nào không nên sử dụng ACC?

💭 Thinking...
🔧 Using tool: reasoning_search
   Args: {'question': 'Khi nào không nên sử dụng ACC?'}
💭 Processing results...
AI: Không nên sử dụng ACC khi:
1. Khu vực nội đô...
2. Đường có nhiều khúc cua...
...
```

## 🔧 Cấu hình

File `src/client/memobase/patch/config.py`:

```python
# Semantic Search (Qdrant)
QDRANT_URL = "your_url"
QDRANT_API_KEY = "your_key"
COLLECTION_NAME = "vsf"

# Logical Reasoning (PageIndex)
PAGEINDEX_API_KEY = "your_key"
```

## 💡 Lựa chọn phương pháp phù hợp

### Dùng Semantic Search khi:
- ✅ Cần tìm kiếm nhanh
- ✅ Tìm theo ý nghĩa/từ khóa
- ✅ Input linh hoạt
- ✅ Tìm tài liệu liên quan

### Dùng Logical Reasoning khi:
- ✅ Câu hỏi phức tạp cần suy luận
- ✅ Cần hiểu cấu trúc tài liệu
- ✅ Câu hỏi "Khi nào?", "Làm thế nào?", "Tại sao?"
- ⚠️ Có thể chấp nhận tốc độ chậm hơn

## 🎨 Output format

### Verbose mode (mặc định đã bật)

Agent sẽ hiển thị:
1. 💭 **Thinking**: Agent đang suy nghĩ
2. 🔧 **Using tool**: Tool nào được chọn + arguments
3. 💭 **Processing results**: Agent đang xử lý kết quả
4. 🤖 **Final answer**: Câu trả lời cuối cùng

## ⚠️ Lưu ý quan trọng

1. **KHÔNG dùng 2 tools đồng thời**: Chỉ chọn 1 mode khi chạy
2. **Reasoning mode**: Query BẮT BUỘC phải là câu hỏi trực tiếp
3. **Semantic mode**: Query linh hoạt (câu hỏi hoặc mô tả)
4. **Streaming**: Cả 2 mode đều hỗ trợ streaming với verbose output

## 📊 So sánh chi tiết

```
┌─────────────────────┬────────────────────┬──────────────────────┐
│                     │ Semantic Search    │ Logical Reasoning    │
├─────────────────────┼────────────────────┼──────────────────────┤
│ Input requirement   │ Flexible           │ Must be question     │
│ Speed              │ Fast (1-2s)        │ Slower (5-10s)       │
│ Method             │ Vector similarity  │ Tree-based reasoning │
│ Best for           │ Keyword search     │ Complex reasoning    │
│ Database           │ Qdrant             │ PageIndex            │
└─────────────────────┴────────────────────┴──────────────────────┘
```

## 🧪 Test queries

### Semantic Search
```
- "tính năng ACC"
- "hệ thống nhận dạng biển báo"
- "các biển báo giao thông"
- "tốc độ tối đa"
```

### Logical Reasoning
```
- "Khi nào không nên sử dụng ACC?"
- "Tính năng Chuyển làn tự động sẽ tắt khi nào?"
- "Làm thế nào để bật ACC?"
- "ACC hoạt động như thế nào?"
```

## 🐛 Troubleshooting

**Lỗi: "Semantic search not available"**
- Kiểm tra đã cài `qdrant-client` chưa
- Kiểm tra config Qdrant trong `config.py`

**Lỗi: "Reasoning search not available"**
- Kiểm tra đã cài `pageindex` chưa
- Kiểm tra `PAGEINDEX_API_KEY` trong `config.py`

**Không tìm thấy kết quả**
- Semantic: Thử paraphrase query
- Reasoning: Đảm bảo input là câu hỏi trực tiếp

## 📞 Support

Nếu gặp vấn đề, kiểm tra:
1. Dependencies đã cài đủ chưa
2. Config trong `config.py` đã đúng chưa
3. API keys còn valid không
4. Query format có đúng không (đặc biệt với reasoning mode)
