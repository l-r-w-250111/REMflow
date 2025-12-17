# REMflow: RAG & Fine-Tuning Framework with Multi-LLM Backend
## Active Retrieval by Day, Deep Consolidation by Night  
This project provides a comprehensive, containerized framework for building and experimenting with Large Language Models (LLMs). It offers an end-to-end workflow from data preparation and fine-tuning to model deployment and inference, all managed through a user-friendly Streamlit interface. The system is designed with a multi-container architecture to support multiple fine-tuning libraries (Unsloth, TRL) and inference engines (Ollama, vLLM), ensuring both stability and flexibility.

## Key Features

- **End-to-End LLM Workflow**: Manage the entire lifecycle of your model within a single application.
  - **Data Preparation**: Create datasets for Continued Pre-Training (CPT) from text files and Supervised Fine-Tuning (SFT) from chat history.
  - **Fine-Tuning**: Train LoRA adapters using powerful libraries like Unsloth and Hugging Face TRL.
  - **Model Conversion**: Merge LoRA adapters and convert models to deployment-ready formats like GGUF and AWQ.
  - **Inference**: Serve and interact with your custom models via a chat interface.

- **Multi-Container Architecture**: Each component runs in a dedicated Docker container for stability and isolation.
  - **`ui`**: A Streamlit container that serves the main user interface and orchestrates other services.
  - **`unsloth`**: A dedicated GPU container for running tasks that require the Unsloth library (e.g., training, model conversion).
  - **`vllm`**: A GPU container for high-throughput inference using the vLLM server.
  - **`ollama-gpu` / `ollama-cpu`**: GPU and CPU containers for serving GGUF models and handling embedding generation.

- **Flexible Inference & Training**:
  - Supports multiple inference backends (**Ollama**, **vLLM**, **Unsloth Server**) that can be switched on the fly.
  - Supports different fine-tuning libraries (**Unsloth**, **TRL**) for CPT and SFT.
  - Supports quantized inference with **AWQ** and **bitsandbytes** in vLLM.

- **Hybrid Learning Capabilities**:
  - **Long-term Memory (Fine-Tuning)**: Adapt models to specific domains or tasks using CPT and SFT.
  - **Short-term Memory (RAG)**: Dynamically provide context to the LLM from uploaded documents and conversation history.

- **Commercially Viable**: Built with free, commercially-usable, non-copyleft libraries under the Apache 2.0 License.

## System Requirements

- Docker and Docker Compose
- NVIDIA GPU with CUDA drivers installed on the host machine.

## Getting Started

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Configure Environment (Optional)

Copy the example environment file. The default settings are sufficient for most use cases.

```bash
cp .env.example .env
```

### 3. Launch the Application

Build and launch all services in the background. This may take some time on the first run as Docker images are built and downloaded.

```bash
sudo docker compose up --build -d
```
*Note: `sudo` may be required depending on your Docker installation.*

### 4. Access the Application

Once the containers are running, the Streamlit UI will be available at:
**`http://localhost:8501`**

The application will automatically download the default chat and embedding models required for the Ollama services on first startup. You can monitor the progress in the Streamline UI.

## Usage Guide

The application is organized into a main chat interface and a sidebar for configuration and workflow management.

### Sidebar Workflow

#### Step 1: Download Base Model
- Before you can train or run a model, you need to download it from Hugging Face.
- Go to the "Step 1: Download Base Model" expander.
- Enter a model identifier (e.g., `unsloth/mistral-7b-v0.1`) and click "Download Model".
- The model will be saved to the `./base_models` directory.

#### Step 2: Continued Pre-Training (CPT) (Optional)
- CPT adapts a model to a specific domain using unstructured text.
- Place your raw text files (`.txt`, `.pdf`, etc.) into subdirectories within `./data/cpt_data/`.
- In the UI, select the base model, choose the data directories, and give your new LoRA adapter a name.
- Click "Generate CPT Training Command". A command will be displayed.
- **Copy this command and run it in a new terminal on the host machine.** This is done to prevent the UI from freezing and to allow you to monitor the training logs.

#### Step 3: Supervised Fine-Tuning (SFT) (Optional)
- SFT teaches a model to follow instructions or a specific chat format.
- You can create a dataset from:
  - **Chat History**: Select past chat sessions from the UI.
  - **Data Files**: Place structured data files in the `./data/sft_data/` directory.
- Select a base model, (optionally) a CPT adapter to stack on, and a name for your new SFT LoRA.
- Click "Generate SFT Training Command" and run the generated command in a new terminal.

#### Step 4: Convert Model to GGUF/AWQ
- This step merges your trained LoRA adapters and converts the model to a format optimized for inference.
- Select the base model and any LoRA adapters you want to merge.
- **To GGUF**: Specify a quantization method (e.g., `q4_k_m`) and an output path (`./gguf_models/my_model.gguf`). Click "Convert to GGUF".
- **To AWQ**: Specify an output path (`./awq_models/my_model_awq`). Click "Convert to AWQ".

### Chat and Inference

1.  **Select Inference Engine**: In the sidebar, choose between `Ollama`, `Unsloth`, or `vLLM`.

2.  **Configure and Start Server**:
    - **Ollama**: Select a GGUF model that you have deployed. Models are deployed via the "Step 4" UI.
    - **vLLM**: Select a base model (from `./base_models`) or an AWQ model (from `./awq_models`). Choose a quantization method and click "Start Server".
    - **Unsloth**: Select a base model and an optional LoRA adapter. Click "Start Server".

3.  **Chat**: Use the main interface to chat with the currently active model. The agent will automatically use RAG or web search based on its assessment of your query.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. For information on third-party libraries, please see [thirdpartylicense.md](thirdpartylicense.md).
