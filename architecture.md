
|ツール|機能範囲|対象モデル・量子化|Dtype (学習/推論)|ファイル形式|アーキテクチャ内での役割|  
|---|---|---|---|---|---|  
|Unsloth|学習 (QLoRA; CPT/SFT) 推論|Hugging Face互換、bitsandbytes (NF4/Int4)|bf16/fp16 (学習) NF4 (基盤)|Hugging Face形式|高速・省メモリなアダプタ作成（学習フェーズ）|
|TRL|学習 (QLoRA; CPT/SFT)"|Hugging Face互換、bitsandbytes (NF4/Int4)|bf16/fp16 (学習) NF4 (基盤)|Hugging Face形式|汎用的・柔軟なアダプタ作成（学習フェーズ）|
|vLLM|推論 (高スループット)|Hugging Face互換、FP8/INT8/INT4/FP16 推論|fp16/bf16 (通常) fp8/int8/int4 (量子化推論)|Hugging Face形式|プロダクションレベルの高速API推論（推論フェーズ）|
|llama.cpp|実行 (GGUF) 変換|GGUF形式 (Llama.cpp互換)|fp16 (通常) q4_0〜q8_0 (量子化推論)|GGUF形式|CPU/GPUハイブリッドなローカル推論実行（推論フェーズ）|
|Ollama|推論 (ローカル環境)|GGUF形式 (llama.cppランタイム)|q4_0〜q8_0 (量子化推論)|GGUF形式|最も手軽なローカルモデルの実行環境（応用/推論フェーズ）|
|LangChain|応用 (RAGエージェント)|LLMとのインターフェース、プロンプト管理、外部ツール連携|モデルのDtypeに依存しない|任意 (API/ローカルモデル)|LLM機能をアプリケーションに組み込むフレームワーク（応用フェーズ）|
