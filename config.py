import os

# Configuration settings
# Use environment variables for Docker, with localhost as a fallback for local development

# Ollama settings for the main LLM (GPU) for inference
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# `os.getenv("LLM_MODEL", "gpt-oss:20b")` は以下の順序で動作します:
# 1. `LLM_MODEL` という名前の環境変数（.envファイルで設定可能）を探します。
# 2. 環境変数が見つかれば、その値を `LLM_MODEL` 変数に設定します。
# 3. 環境変数が見つからない場合、デフォルト値として "gpt-oss:20b" を使用します。
# したがって、.envファイルで `LLM_MODEL=別のモデル名` と設定すれば、そちらが優先されます。
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-oss:20b")
# LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")

# Ollama settings for the embedding model (CPU)
EMBEDDING_OLLAMA_URL = os.getenv("EMBEDDING_OLLAMA_URL", "http://localhost:11435")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embeddinggemma")

# --- Model Storage Configuration ---
# Use environment variables to allow users to specify external directories for models.
# The default values point to local directories within the project.
# In a Docker environment, these paths correspond to the paths INSIDE the container.
# Users should map their local Windows/Host paths to these container paths via docker-compose.yml volumes.

# Path to save and load LoRA adapters
LORA_MODEL_PATH = os.getenv("LORA_MODEL_PATH", "lora_models/")

# Path to store and load base models for fine-tuning
BASE_MODEL_PATH = os.getenv("BASE_MODEL_PATH", "base_models/")

