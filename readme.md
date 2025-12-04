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

### 5. Chạy main.py

```bash
python main.py
```




