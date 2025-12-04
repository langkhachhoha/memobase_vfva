# Memobase Chat Demo

Simple local chat interface with memory and profile display.

## Quick Start

```bash
conda activate memobase_vivi
python run_local.py
```

Access at: http://localhost:8000

## Features

- ✅ Chat with memory
- ✅ Profile panel (auto-parsed from memory)
- ✅ Auto-flush every 5 turns
- ✅ Manual flush & refresh buttons

## Prerequisites

1. Memobase server on port 8019
2. `.env` file with `llm_api_key`
3. Conda environment `memobase_vivi`

## Files

- `app.py` - Main FastAPI application
- `run_local.py` - Simple launcher script

