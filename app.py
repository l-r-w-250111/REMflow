import streamlit as st
import os
import subprocess
import json
import time
import requests
from rag_handler import RAGHandler
from openai import OpenAI
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama as LlamaIndexOllama
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms.mock import MockLLM
from web_search import perform_web_search


# Set a dummy LLM at startup to allow RAGHandler to initialize without a real model.
# The actual LLM will be set later inside the Agent class.
if "llm_placeholder_set" not in st.session_state:
    Settings.llm = MockLLM()
    st.session_state.llm_placeholder_set = True


# --- Page Configuration ---
st.set_page_config(
    page_title="LLM Agent with RAG and LoRA",
    layout="wide"
)

# --- Environment Variables ---
OLLAMA_GPU_URL = os.getenv("OLLAMA_GPU_URL", "http://localhost:11434")
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000")
UNSLOTH_URL = os.getenv("UNSLOTH_URL", "http://localhost:8003")
COMPOSE_PROJECT_NAME = os.getenv("COMPOSE_PROJECT_NAME")

def get_docker_compose_command():
    """Constructs the base docker compose command with project name if available."""
    command = ["docker", "compose"]
    if COMPOSE_PROJECT_NAME:
        command.extend(["--project-name", COMPOSE_PROJECT_NAME])
    return command


# --- Helper Functions ---
def get_chat_history_path():
    return "chat_history.json"

def load_all_sessions():
    path = get_chat_history_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            if not content: return {}
            sessions = json.loads(content)
        # Migration for backward compatibility
        for session_id, data in sessions.items():
            if isinstance(data, list):
                now = int(time.time())
                sessions[session_id] = {"created_at": now, "last_updated": now, "messages": data}
        return sessions
    except (json.JSONDecodeError, TypeError):
        return {}

def save_all_sessions(sessions_data=None):
    path = get_chat_history_path()
    data_to_save = sessions_data if sessions_data is not None else st.session_state.sessions
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        f.write('\n')

def create_new_session():
    now = int(time.time())
    session_id = f"session_{now}"
    st.session_state.sessions[session_id] = {"created_at": now, "last_updated": now, "messages": []}
    return session_id

import pandas as pd

def get_ollama_models(ollama_url):
    """Fetches the list of available models from a specific Ollama service."""
    try:
        response = requests.get(f"{ollama_url}/api/tags")
        response.raise_for_status()
        return [model['name'] for model in response.json().get('models', [])]
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to Ollama service at {ollama_url}: {e}")
        return []



def ensure_ollama_model(model_name, ollama_url, service_name):
    """Checks if a model exists in an Ollama service and pulls it if it doesn't."""
    available_models = get_ollama_models(ollama_url)
    if not any(model_name in m for m in available_models):
        st.info(f"Model '{model_name}' not found in {service_name}. Pulling model...")
        try:
            payload = {"name": model_name, "stream": True}
            response = requests.post(f"{ollama_url}/api/pull", json=payload, stream=True)
            response.raise_for_status()

            progress_bar = st.progress(0, text="Starting download...")
            status_text = st.empty()
            total = 0
            completed = 0

            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "total" in data and "completed" in data:
                        total = data["total"]
                        completed = data["completed"]
                        progress = min(1.0, completed / total if total > 0 else 0)
                        status_text.text(f"Downloading '{model_name}': {data.get('status', '')}")
                        progress_bar.progress(progress)
                    else:
                        status_text.text(f"Status: {data.get('status', 'working...')}")

            progress_bar.empty()
            status_text.empty()
            st.success(f"Successfully pulled '{model_name}' to {service_name}.")
            st.rerun() # Rerun to refresh model lists
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to pull model '{model_name}' from {ollama_url}: {e}")
            st.stop()
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse response from Ollama service: {e}")
            st.stop()


# --- Initialization ---
def initialize_session_state():
    # --- General App State ---
    if "rag_handler" not in st.session_state:
        try:
            st.session_state.rag_handler = RAGHandler()
        except Exception as e:
            st.session_state.rag_handler = None
            st.warning(f"Failed to initialize RAG Handler. RAG features will be disabled. Error: {e}")
    if "sessions" not in st.session_state:
        st.session_state.sessions = load_all_sessions()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- Inference Engine Selection ---
    if "inference_engine" not in st.session_state:
        st.session_state.inference_engine = "Ollama"

    # --- Ollama Related State ---
    if "ollama_client" not in st.session_state:
        st.session_state.ollama_client = None
    if "selected_chat_model" not in st.session_state:
        st.session_state.selected_chat_model = None

    # --- Unsloth/HF Related State ---
    if "unsloth_base_model" not in st.session_state:
        st.session_state.unsloth_base_model = None
    if "unsloth_lora_model" not in st.session_state:
        st.session_state.unsloth_lora_model = "None"
    if "unsloth_server_running" not in st.session_state:
        st.session_state.unsloth_server_running = False


    # --- vLLM Related State ---
    if "vllm_base_model" not in st.session_state:
        st.session_state.vllm_base_model = None
    if "vllm_lora_model" not in st.session_state:
        st.session_state.vllm_lora_model = "None"
    if "vllm_server_running" not in st.session_state:
        st.session_state.vllm_server_running = False
    if "vllm_client" not in st.session_state:
        st.session_state.vllm_client = None
    if "vllm_quantization" not in st.session_state:
        st.session_state.vllm_quantization = "None"

    # --- Training State ---
    if "training_library" not in st.session_state:
        st.session_state.training_library = "Unsloth"


    # --- Session Management ---
    if "current_session_id" not in st.session_state:
        if st.session_state.sessions:
            latest_session_id = max(st.session_state.sessions.keys(), key=lambda s_id: st.session_state.sessions[s_id]['last_updated'])
            st.session_state.current_session_id = latest_session_id
        else:
            new_id = create_new_session()
            st.session_state.current_session_id = new_id
            save_all_sessions()

initialize_session_state()

# --- Ensure required Ollama models are available on startup ---
# This is placed here to run once when the app starts.
# if "ollama_models_checked" not in st.session_state:
#     CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "llama3:8b")
#     EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "embeddinggemma")
#     ensure_ollama_model(CHAT_MODEL_NAME, OLLAMA_GPU_URL, "Ollama GPU Service") # モデルが見つからない場合に自動ロード
#     ensure_ollama_model(EMBEDDING_MODEL_NAME, OLLAMA_CPU_URL, "Ollama CPU Service") # モデルが見つからない場合に自動ロード
#     st.session_state.ollama_models_checked = True


# --- Global LLM Configuration for LlamaIndex ---
# Set a default LLM for RAG at startup to avoid errors.
# This will be dynamically updated when the user makes a selection in the UI.
def check_unsloth_server_status():
    """Checks if the Unsloth server is running and responsive."""
    try:
        response = requests.get(f"{UNSLOTH_URL}/health")
        if response.status_code == 200:
            st.session_state.unsloth_server_running = True
            return True
    except requests.exceptions.RequestException:
        st.session_state.unsloth_server_running = False
    return False

def check_vllm_server_status():
    """Checks if the vLLM server is running and responsive."""
    try:
        response = requests.get(f"{VLLM_URL}/health")
        if response.status_code == 200:
            st.session_state.vllm_server_running = True
            return True
    except requests.exceptions.RequestException:
        st.session_state.vllm_server_running = False
    return False


# --- Sidebar ---
with st.sidebar:
    st.header("Chat Configuration")

    # Let user choose the inference engine
    st.session_state.inference_engine = st.radio(
        "Select Inference Engine:",
        ["Ollama", "Unsloth", "vLLM"],
        key="inference_engine_selector",
        help="Choose your preferred inference backend."
    )

    # --- Conditional UI based on engine selection ---
    if st.session_state.inference_engine == "Ollama":
        st.subheader("Ollama Model")
        available_ollama_models = get_ollama_models(OLLAMA_GPU_URL)
        if available_ollama_models:
            try:
                current_index = available_ollama_models.index(st.session_state.selected_chat_model) if st.session_state.selected_chat_model in available_ollama_models else 0
            except ValueError:
                current_index = 0

            selected_model = st.selectbox(
                "Select a running model:",
                options=available_ollama_models,
                index=current_index
            )
            if selected_model:
                # Create an OpenAI-compatible client for the selected Ollama model.
                # The '/v1' endpoint is crucial for compatibility.
                st.session_state.ollama_client = OpenAI(
                    api_key="ollama",
                    base_url=f"{OLLAMA_GPU_URL}/v1"
                )
                st.session_state.selected_chat_model = selected_model
                # Dynamically set the RAG LLM to the selected Ollama model
                Settings.llm = LlamaIndexOllama(
                    model=selected_model,
                    base_url=OLLAMA_GPU_URL,
                    request_timeout=120.0
                )
        else:
            st.warning("No Ollama models found or service is unavailable.")

    elif st.session_state.inference_engine == "Unsloth":
        st.subheader("Unsloth Server Control")
        Settings.llm = None # RAG not used in this path
        base_model_dir = "./base_models"
        lora_model_dir = "./lora_models"

        unsloth_base_model_options = [d for d in os.listdir(base_model_dir) if os.path.isdir(os.path.join(base_model_dir, d))]
        st.session_state.unsloth_base_model = st.selectbox("Select Base Model:", options=unsloth_base_model_options or [""], key="unsloth_base_selector")

        available_loras = ["None"] + [d for d in os.listdir(lora_model_dir) if os.path.isdir(os.path.join(lora_model_dir, d))]
        st.session_state.unsloth_lora_model = st.selectbox("Select LoRA Adapter (Optional):", options=available_loras, key="unsloth_lora_selector")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Server", key="unsloth_start", disabled=st.session_state.unsloth_server_running):
                if st.session_state.unsloth_base_model:
                    with st.spinner("Starting Unsloth server..."):
                        base_model_path = os.path.join("./base_models", st.session_state.unsloth_base_model)
                        
                        # Set permissions for model directories using docker exec
                        try:
                            st.write(f"Setting permissions for {base_model_path}...")
                            model_path_in_container = os.path.join("/app/base_models", st.session_state.unsloth_base_model)
                            perm_cmd = get_docker_compose_command() + ["exec", "-u", "root", "unsloth", "chmod", "-R", "777", model_path_in_container]
                            subprocess.run(perm_cmd, check=True, capture_output=True, text=True)

                            if st.session_state.unsloth_lora_model != "None":
                                lora_path = os.path.join("./lora_models", st.session_state.unsloth_lora_model)
                                lora_path_in_container = os.path.join("/app/lora_models", st.session_state.unsloth_lora_model)
                                st.write(f"Setting permissions for {lora_path}...")
                                perm_cmd_lora = get_docker_compose_command() + ["exec", "-u", "root", "unsloth", "chmod", "-R", "777", lora_path_in_container]
                                subprocess.run(perm_cmd_lora, check=True, capture_output=True, text=True)
                        except subprocess.CalledProcessError as e:
                            st.error(f"Failed to set model permissions via Docker exec: {e.stderr}")
                            st.stop()

                        # The uvicorn command no longer takes model paths as arguments
                        server_cmd = "uvicorn unsloth_server:app --host 0.0.0.0 --port 8003"
                        
                        log_file_path_host = "unsloth.log"
                        log_file_path_container = "/app/unsloth.log"

                        # Clear previous log file
                        if os.path.exists(log_file_path_host):
                            os.remove(log_file_path_host)

                        shell_command = f"{server_cmd} > {log_file_path_container} 2>&1"
                        
                        command = get_docker_compose_command()
                        
                        # Prepare environment variables to be passed to the exec command
                        env_vars = [
                            "-e", f"BASE_MODEL_PATH={model_path_in_container}"
                        ]
                        if st.session_state.unsloth_lora_model != "None":
                            lora_path_in_container = os.path.join("/app/lora_models", st.session_state.unsloth_lora_model)
                            env_vars.extend(["-e", f"LORA_MODEL_PATH={lora_path_in_container}"])

                        # Construct the final docker compose command with environment variables
                        command.extend(["exec"])
                        command.extend(env_vars) # Add environment variables
                        command.extend(["-d", "unsloth", "sh", "-c", shell_command])
                        
                        subprocess.Popen(command)
                        st.info("Unsloth server is starting. Polling for readiness...")
                        
                        start_time = time.time()
                        timeout = 120 # 2 minutes
                        server_ready = False
                        
                        with st.spinner("Waiting for Unsloth server to become available..."):
                            while time.time() - start_time < timeout:
                                if check_unsloth_server_status():
                                    server_ready = True
                                    break
                                time.sleep(5)
                        
                        if server_ready:
                            st.success("Unsloth server started successfully!")
                        else:
                            st.error("Unsloth server failed to start. See logs below.")
                            # Try to read the log file and display it
                            time.sleep(2) # Give a moment for the log file to be written
                            if os.path.exists(log_file_path_host):
                                with open(log_file_path_host, "r") as f:
                                    log_contents = f.read()
                                st.code(log_contents, language="log")
                            else:
                                st.warning(f"Could not find log file at: {log_file_path_host}")
                else:
                    st.error("Please select a base model for the Unsloth server.")

        with col2:
            if st.button("Stop Server", key="unsloth_stop", disabled=not st.session_state.unsloth_server_running):
                with st.spinner("Stopping Unsloth server..."):
                    kill_cmd = get_docker_compose_command()
                    kill_cmd.extend(["exec", "unsloth", "pkill", "-f", "uvicorn unsloth_server:app"])
                    subprocess.run(kill_cmd)
                    st.session_state.unsloth_server_running = False
                    st.success("Unsloth server stopped.")

        if st.button("Check Server Status", key="unsloth_status"):
            if check_unsloth_server_status():
                st.success("Unsloth server is running.")
            else:
                st.error("Unsloth server is not responding.")

    elif st.session_state.inference_engine == "vLLM":
        st.subheader("vLLM Server Control")
        base_model_dir = "./base_models"
        awq_model_dir = "./awq_models"
        lora_model_dir = "./lora_models"
        os.makedirs(awq_model_dir, exist_ok=True) # Ensure AWQ directory exists

        st.session_state.vllm_quantization = st.radio(
            "Select Quantization Method:",
            ["None", "bitsandbytes", "AWQ"],
            key="vllm_quant_selector"
        )

        model_dir_to_use = awq_model_dir if st.session_state.vllm_quantization == "AWQ" else base_model_dir
        model_options = [d for d in os.listdir(model_dir_to_use) if os.path.isdir(os.path.join(model_dir_to_use, d))]

        if model_options:
            st.session_state.vllm_base_model = st.selectbox(
                f"Select Model from '{model_dir_to_use}':",
                options=model_options,
                key="vllm_base_selector"
            )
        else:
            st.warning(f"No models found in '{model_dir_to_use}'.")
            st.session_state.vllm_base_model = None

        if st.session_state.vllm_quantization != "AWQ":
            available_loras = ["None"] + [d for d in os.listdir(lora_model_dir) if os.path.isdir(os.path.join(lora_model_dir, d))]
            st.session_state.vllm_lora_model = st.selectbox("Select LoRA Adapter (Optional):", options=available_loras, key="vllm_lora_selector")
        else:
            st.session_state.vllm_lora_model = "None"

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Server", disabled=st.session_state.vllm_server_running):
                if st.session_state.vllm_base_model:
                    with st.spinner("Starting vLLM server..."):
                        # Determine model path based on quantization
                        model_dir_host = awq_model_dir if st.session_state.vllm_quantization == "AWQ" else base_model_dir
                        model_dir_container = "/app/awq_models" if st.session_state.vllm_quantization == "AWQ" else "/app/base_models"
                        
                        model_path_host = os.path.join(model_dir_host, st.session_state.vllm_base_model)
                        model_path_in_container = os.path.join(model_dir_container, st.session_state.vllm_base_model)

                        # Set permissions for model directories using docker exec
                        try:
                            st.write(f"Setting permissions for {model_path_host}...")
                            perm_cmd = get_docker_compose_command() + ["exec", "-u", "root", "vllm", "chmod", "-R", "777", model_path_in_container]
                            subprocess.run(perm_cmd, check=True, capture_output=True, text=True)

                            if st.session_state.vllm_lora_model != "None":
                                lora_path = os.path.join("./lora_models", st.session_state.vllm_lora_model)
                                lora_path_in_container = os.path.join("/app/lora_models", st.session_state.vllm_lora_model)
                                st.write(f"Setting permissions for {lora_path}...")
                                perm_cmd_lora = get_docker_compose_command() + ["exec", "-u", "root", "vllm", "chmod", "-R", "777", lora_path_in_container]
                                subprocess.run(perm_cmd_lora, check=True, capture_output=True, text=True)
                        except subprocess.CalledProcessError as e:
                            st.error(f"Failed to set model permissions via Docker exec: {e.stderr}")
                            st.stop()

                        # Build the shell command for vLLM server
                        vllm_server_cmd_parts = [
                            "python3", "-m", "vllm.entrypoints.openai.api_server",
                            "--host", "0.0.0.0", "--port", "8000",
                            "--dtype", "float16",
                            "--model", model_path_in_container,
                            "--uvicorn-log-level", "debug"
                        ]

                        # Add quantization parameter if selected
                        if st.session_state.vllm_quantization != "None":
                            vllm_server_cmd_parts.extend(["--quantization", st.session_state.vllm_quantization])

                        # Add LoRA parameter if selected (and not AWQ)
                        if st.session_state.vllm_lora_model != "None":
                            lora_path = os.path.join("/app/lora_models", st.session_state.vllm_lora_model)
                            vllm_server_cmd_parts.extend(["--enable-lora", "--lora-modules", lora_path])

                        # Redirect output to a log file inside the container.
                        # The log will be available at `./vllm.log` on the host.
                        shell_command = " ".join(vllm_server_cmd_parts) + " > /app/vllm.log 2>&1"

                        command = get_docker_compose_command()
                        # Use -d to detach, and wrap the command in `sh -c` to handle redirection.
                        command.extend(["exec", "-d", "vllm", "sh", "-c", shell_command])

                        subprocess.Popen(command)

                        # Polling loop to check for server readiness
                        st.info("The vLLM server is starting. This can take several minutes, especially on the first run, as it needs to compile CUDA graphs.")
                        start_time = time.time()
                        timeout = 600  # Increased timeout to 10 minutes (600 seconds)
                        server_ready = False
                        progress_bar = st.progress(0)
                        
                        while time.time() - start_time < timeout:
                            elapsed_time = time.time() - start_time
                            progress_percentage = int((elapsed_time / timeout) * 100)
                            progress_bar.progress(progress_percentage, text=f"Waiting for server... ({int(elapsed_time)}s / {timeout}s)")

                            if check_vllm_server_status():
                                server_ready = True
                                break
                            time.sleep(5)
                        
                        progress_bar.empty()

                        if server_ready:
                            st.session_state.vllm_client = OpenAI(api_key="vllm", base_url=f"{VLLM_URL}/v1")
                            st.success("vLLM server started successfully!")
                        else:
                            st.error(f"vLLM server failed to start within the {timeout} second timeout. Check the container logs for 'vllm' service for more details.")
                else:
                    st.error("Please select a base model first.")

        with col2:
            if st.button("Stop Server", disabled=not st.session_state.vllm_server_running):
                with st.spinner("Stopping vLLM server..."):
                    # Find and kill the server process inside the container
                    kill_cmd = get_docker_compose_command()
                    kill_cmd.extend(["exec", "vllm", "pkill", "-f", "vllm.entrypoints.openai.api_server"])
                    result = subprocess.run(kill_cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        st.session_state.vllm_server_running = False
                        st.session_state.vllm_client = None
                        st.success("vLLM server stopped.")
                    else:
                        st.warning("Could not stop server cleanly, it might already be stopped.")
                        st.session_state.vllm_server_running = False

        if st.button("Check Server Status"):
            if check_vllm_server_status():
                 st.success("vLLM server is running.")
            else:
                 st.error("vLLM server is not responding.")

        # Configure LlamaIndex to use the vLLM OpenAI-compatible endpoint for RAG
        # This needs to be done here to ensure the RAG engine uses the correct model.
        if st.session_state.vllm_server_running and st.session_state.vllm_base_model:
            model_dir_container = "/app/awq_models" if st.session_state.vllm_quantization == "AWQ" else "/app/base_models"
            model_path_in_container = os.path.join(model_dir_container, st.session_state.vllm_base_model)
            Settings.llm = OpenAILike(
                model=model_path_in_container,
                api_base=f"{VLLM_URL}/v1",
                api_key="vllm",
                is_chat_model=True,
                temperature=0,
                max_tokens=1024,
            )


    st.header("Chat Sessions")
    if st.button("New Chat"):
        new_id = create_new_session()
        st.session_state.current_session_id = new_id
        st.session_state.messages = []
        save_all_sessions()
        st.rerun()

    st.subheader("History")
    sorted_session_ids = sorted(st.session_state.sessions.keys(), key=lambda s_id: st.session_state.sessions[s_id]['last_updated'], reverse=True)
    for session_id in sorted_session_ids:
        session_info = st.session_state.sessions[session_id]
        last_updated_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(session_info['last_updated']))
        if st.button(last_updated_str, key=session_id):
            st.session_state.current_session_id = session_id
            st.session_state.messages = session_info['messages']
            st.rerun()

    st.header("RAG Management")
    uploaded_file = st.file_uploader("Upload a document to RAG", type=['txt', 'pdf', 'md'])
    if uploaded_file is not None:
        data_dir = "./data"
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner('Adding document to RAG...'):
            st.session_state.rag_handler.add_document(file_path)
            st.success(f"Document '{uploaded_file.name}' added successfully!")

    st.header("Training Workflow")

    base_model_dir = "./base_models"
    lora_model_dir = "./lora_models"
    cpt_data_dir = "./data/cpt_data"
    sft_data_dir = "./data/sft_data"
    datasets_dir = "./datasets"

    os.makedirs(base_model_dir, exist_ok=True)
    os.makedirs(lora_model_dir, exist_ok=True)
    os.makedirs(cpt_data_dir, exist_ok=True)
    os.makedirs(sft_data_dir, exist_ok=True)
    os.makedirs(datasets_dir, exist_ok=True)

    with st.expander("Step 1: Download Base Model", expanded=True):
        new_model_name = st.text_input("Download Model from Hugging Face", placeholder="e.g., unsloth/mistral-7b-v0.1")
        if st.button("Download Model"):
            if new_model_name:
                safe_model_name = new_model_name.replace("/", "_")
                save_path = os.path.join(base_model_dir, safe_model_name)
                if os.path.exists(save_path):
                    st.warning(f"Model directory '{safe_model_name}' already exists.")
                else:
                    with st.spinner(f"Downloading {new_model_name}... This may take a while."):
                        command = get_docker_compose_command()
                        command.extend([
                            "exec", "unsloth", "python", "download_model.py",
                            "--model_name", new_model_name, "--save_path", save_path
                        ])

                        # Use subprocess.run to make it a blocking call and capture output
                        result = subprocess.run(command, capture_output=True, text=True, check=False)

                    if result.returncode == 0:
                        st.success(f"Successfully downloaded {new_model_name}!")
                        st.text("Download log:")
                        st.code(result.stdout)
                        st.rerun() # Rerun to refresh the model lists
                    else:
                        st.error(f"Failed to download {new_model_name}.")
                        st.text("Error log:")
                        st.code(result.stderr)
            else:
                st.error("Please enter a model name.")

    with st.expander("Step 2: Continued Pre-Training (CPT)"):
        st.write("This step fine-tunes the base model on raw text data to adapt it to a specific domain.")

        cpt_training_library = st.radio("Select CPT Library", ["Unsloth", "TRL"], key="cpt_library")

        # CPT Data Source Selection
        available_cpt_dirs = [d for d in os.listdir(cpt_data_dir) if os.path.isdir(os.path.join(cpt_data_dir, d))] if os.path.isdir(cpt_data_dir) else []
        selected_cpt_dirs = st.multiselect(
            "Select CPT Data Directories:",
            options=available_cpt_dirs,
            help="Select one or more directories containing your CPT data files."
        )

        cpt_base_model_options = [d for d in os.listdir(base_model_dir) if os.path.isdir(os.path.join(base_model_dir, d))]
        cpt_base_model = st.selectbox("Select Base Model for CPT:", options=cpt_base_model_options, key="cpt_base_model")
        cpt_lora_name = st.text_input("New CPT LoRA Name", placeholder="e.g., my_domain_cpt_v1", key="cpt_lora_name")

        if st.button("Generate CPT Training Command"):
            if not cpt_base_model or not cpt_lora_name or not selected_cpt_dirs:
                st.error("Please select a base model, provide a LoRA name, and select at least one CPT data directory.")
            else:
                with st.spinner("Preparing CPT dataset..."):
                    cpt_source_paths_in_container = []
                    for dir_name in selected_cpt_dirs:
                        host_path = os.path.join(cpt_data_dir, dir_name)
                        if os.path.isdir(host_path):
                            st.write(f"Setting permissions for {host_path}...")
                            try:
                                path_in_container = os.path.join("/app", host_path)
                                # The service for chmod should be 'ui' as it's where the script runs
                                perm_cmd = get_docker_compose_command() + ["exec", "-u", "root", "ui", "chmod", "-R", "777", path_in_container]
                                subprocess.run(perm_cmd, check=True, capture_output=True, text=True)
                                cpt_source_paths_in_container.append(path_in_container)
                            except subprocess.CalledProcessError as e:
                                st.error(f"Failed to set data permissions for {host_path} via Docker exec: {e.stderr}")
                                st.stop()

                    dataset_path = os.path.join("/app", datasets_dir, f"{cpt_lora_name}_dataset.json")

                    st.write("Setting permissions for datasets directory...")
                    try:
                        # The service for chmod should be 'ui' as it's where the script runs
                        perm_cmd = get_docker_compose_command() + ["exec", "-u", "root", "ui", "chmod", "-R", "777", os.path.join("/app", datasets_dir)]
                        subprocess.run(perm_cmd, check=True, capture_output=True, text=True)
                    except subprocess.CalledProcessError as e:
                        st.error(f"Failed to set dataset directory permissions via Docker exec: {e.stderr}")
                        st.stop()

                    create_cmd = get_docker_compose_command()
                    create_cmd.extend([
                        "exec", "ui", "python", "create_dataset.py",
                        "--output_path", dataset_path,
                        "--format_type", "cpt",
                        "--source_dir", *cpt_source_paths_in_container
                    ])
                    st.write("Creating CPT dataset...")
                    result = subprocess.run(create_cmd, capture_output=True, text=True, check=False)
                    if result.returncode != 0:
                        st.error(f"CPT dataset creation failed:\n{result.stderr}")
                        st.stop()
                    st.success(f"CPT dataset created at {dataset_path.replace('/app/', './')}")

                model_path = os.path.normpath(os.path.join("/app", base_model_dir, cpt_base_model))
                normalized_dataset_path = os.path.normpath(dataset_path)

                train_script = "train_trl.py" if cpt_training_library == "TRL" else "train_lora.py"
                target_service = "vllm" if train_script == "train_trl.py" else "unsloth"
                train_cmd = get_docker_compose_command()
                train_cmd.extend([
                    "exec", target_service, "python", train_script,
                    "--model_path", model_path, "--dataset_path", normalized_dataset_path,
                    "--lora_name", cpt_lora_name, "--training_type", "cpt"
                ])

                st.info("""
                **Training command generated.**
                
                Copy the command below and run it in a new terminal to start the training.
                
                **Why run it manually?**
                - **Prevent UI Freezing:** Training is a time-consuming process. Running it directly within the UI could cause the application to become unresponsive.
                - **Monitor Logs:** Running it in a terminal allows you to track training progress and check for errors in real time.

                **トレーニングコマンドが生成されました。**

                以下のコマンドをコピーし、新しいターミナルで実行してトレーニングを開始してください。

                **なぜ手動で実行するのか？**
                - **UIのフリーズ防止:** トレーニングは時間がかかるため、UI内で直接実行するとアプリが応答しなくなる可能性があります。
                - **ログの確認:** ターミナルで直接実行することで、学習の進捗状況やエラーをリアルタイムで確認できます。
                """)
                st.code(" ".join(train_cmd), language="bash")

    with st.expander("Step 3: Supervised Fine-Tuning (SFT)"):
        st.write("This step fine-tunes the model (potentially with a CPT adapter) on structured instruction-response data.")

        sft_training_library = st.radio("Select SFT Library", ["Unsloth", "TRL"], key="sft_library")

        sft_dataset_format = st.selectbox("SFT Dataset Format", ["alpaca", "harmony"], key="sft_format")
        sft_lora_name = st.text_input("New SFT LoRA Name", placeholder="e.g., my_sft_adapter_v1", key="sft_lora_name")
        dataset_output_path = os.path.join(datasets_dir, f"{sft_lora_name}_dataset.json")

        # --- SFT Data Source Selection ---
        st.subheader("Select SFT Data Sources")

        # Option 1: Select from Chat History
        all_sessions = st.session_state.get("sessions", {})
        session_options = {
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(s['last_updated']))} ({s_id})": s_id
            for s_id, s in sorted(all_sessions.items(), key=lambda item: item[1]['last_updated'], reverse=True)
        }
        selected_session_labels = st.multiselect(
            "Select chat sessions to include in the dataset:",
            options=session_options.keys(),
        )
        selected_session_ids = [session_options[label] for label in selected_session_labels]

        # Option 2: Provide a directory path for additional data
        sft_data_path = st.text_input("SFT Data Directory Path (Optional)", value=sft_data_dir, key="sft_data_path", help="Path to a directory containing additional SFT data files (e.g., in Alpaca format).")

        if st.button("Create SFT Dataset"):
            if not selected_session_ids and not os.path.isdir(sft_data_path):
                st.error("Please select at least one chat session or provide a valid data directory.")
                st.stop()

            with st.spinner("Creating SFT dataset..."):
                temp_chat_history_path = None
                if selected_session_ids:
                    selected_sessions_data = {s_id: all_sessions[s_id] for s_id in selected_session_ids}

                    # Create a temporary directory and file for the selected history
                    temp_dir = "./temp"
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_chat_history_path = os.path.join(temp_dir, f"{sft_lora_name}_selected_chats.json")

                    with open(temp_chat_history_path, "w", encoding="utf-8") as f:
                        json.dump(selected_sessions_data, f, indent=2, ensure_ascii=False)
                        f.write('\n')

                create_cmd = get_docker_compose_command()
                create_cmd.extend([
                    "exec", "ui", "python", "create_dataset.py",
                    "--output_path", os.path.join("/app", dataset_output_path),
                    "--format_type", sft_dataset_format,
                ])

                if temp_chat_history_path:
                    create_cmd.extend(["--chat_history_path", os.path.join("/app", temp_chat_history_path)])
                
                sft_input_path = st.session_state.sft_data_path
                if sft_input_path and os.path.isdir(sft_input_path):
                    st.write(f"Setting permissions for {sft_input_path}...")
                    try:
                        path_in_container = os.path.join("/app", sft_input_path)
                        perm_cmd = get_docker_compose_command() + ["exec", "-u", "root", "unsloth", "chmod", "-R", "777", path_in_container]
                        subprocess.run(perm_cmd, check=True, capture_output=True, text=True)
                    except subprocess.CalledProcessError as e:
                        st.error(f"Failed to set data permissions via Docker exec: {e.stderr}")
                        st.stop()
                        
                    all_files = []
                    for root, _, files in os.walk(sft_input_path):
                        for file in files:
                            all_files.append(os.path.join(root, file))
                    
                    if all_files:
                        file_paths_args = [os.path.join("/app", f) for f in all_files]
                        create_cmd.extend(["--file_paths", *file_paths_args])

                result = subprocess.run(create_cmd, capture_output=True, text=True, check=False)
                if result.returncode != 0:
                    st.error(f"SFT dataset creation failed:\n{result.stderr}")
                    st.stop()
                st.success(f"SFT dataset created at {dataset_output_path}")

        sft_base_model_options = [d for d in os.listdir(base_model_dir) if os.path.isdir(os.path.join(base_model_dir, d))]
        sft_base_model = st.selectbox("Select Base Model for SFT:", options=sft_base_model_options, key="sft_base_model")

        available_cpt_adapters = ["None"] + [d for d in os.listdir(lora_model_dir) if os.path.isdir(os.path.join(lora_model_dir, d))]
        sft_stacked_lora = st.selectbox("Stack on top of CPT LoRA (Optional):", options=available_cpt_adapters, key="sft_stack",
                                        help="Note: Stacking is only supported with Unsloth.")

        if st.button("Generate SFT Training Command"):
            if not sft_base_model or not sft_lora_name:
                st.error("Please select a base model and provide a LoRA name for SFT.")
            elif not os.path.exists(dataset_output_path):
                st.error(f"SFT dataset not found at {dataset_output_path}. Please create it first.")
            else:
                model_path = os.path.normpath(os.path.join("/app", base_model_dir, sft_base_model))
                normalized_dataset_path = os.path.normpath(os.path.join("/app", dataset_output_path))
                train_script = "train_trl.py" if sft_training_library == "TRL" else "train_lora.py"
                target_service = "vllm" if train_script == "train_trl.py" else "unsloth"

                train_cmd = get_docker_compose_command()
                train_cmd.extend([
                    "exec", target_service, "python", train_script,
                    "--model_path", model_path,
                    "--dataset_path", normalized_dataset_path,
                    "--lora_name", sft_lora_name,
                    "--training_type", "sft"
                ])

                if sft_stacked_lora != "None":
                    if sft_training_library == "Unsloth":
                        cpt_adapter_path = os.path.join("/app", lora_model_dir, sft_stacked_lora)
                        train_cmd.extend(["--cpt_adapter_path", cpt_adapter_path])
                    else:
                        st.warning("LoRA stacking is only enabled for Unsloth. The CPT adapter will be ignored for TRL training.")

                st.info("""                        
                **Training command generated.**
                
                Copy the command below and run it in a new terminal to start the training.
                
                **Why run it manually?**
                - **Prevent UI Freezing:** Training is a time-consuming process. Running it directly within the UI could cause the application to become unresponsive.
                - **Monitor Logs:** Running it in a terminal allows you to track training progress and check for errors in real time.

                **トレーニングコマンドが生成されました。**

                以下のコマンドをコピーし、新しいターミナルで実行してトレーニングを開始してください。

                **なぜ手動で実行するのか？**
                - **UIのフリーズ防止:** トレーニングは時間がかかるため、UI内で直接実行するとアプリが応答しなくなる可能性があります。
                - **ログの確認:** ターミナルで直接実行することで、学習の進捗状況やエラーをリアルタイムで確認できます。
                """)
                st.code(" ".join(train_cmd), language="bash")


    with st.expander("Step 4: Convert Model to GGUF/AWQ"):
        st.subheader("Convert Model to GGUF")

        # Base model selection for deployment
        available_base_models_deploy = [d for d in os.listdir(base_model_dir) if os.path.isdir(os.path.join(base_model_dir, d))]
        selected_base_model_deploy = st.selectbox("Select Base Model to Convert:", options=available_base_models_deploy, key="deploy_base_model")

        # LoRA adapter selection (multiselect)
        available_loras = [d for d in os.listdir(lora_model_dir) if os.path.isdir(os.path.join(lora_model_dir, d))]
        selected_loras = st.multiselect("Select LoRA Adapter(s) to Merge (Optional):", options=available_loras)

        quantization_method = st.text_input("Quantization Method", "q4_k_m", placeholder="e.g., q4_k_m, q8_0, f16")
        output_path = st.text_input("GGUF Output Path", placeholder="e.g., ./gguf_models/my_model_q4km.gguf")
        is_4bit_model = st.checkbox("Base model is 4-bit quantized")

        if st.button("Convert to GGUF"):
            if not all([selected_base_model_deploy, quantization_method, output_path]):
                st.error("Please select a base model, quantization method, and output path.")
            else:
                model_path = os.path.join(base_model_dir, selected_base_model_deploy)
                command = get_docker_compose_command()
                command.extend([
                    "exec", "unsloth", "python", "/app/convert_to_gguf.py",
                    "--model_path", model_path,
                    "--quantization_method", quantization_method,
                    "--output_path", output_path
                ])

                # Add lora if selected
                if selected_loras:
                    command.append("--lora_names")
                    command.extend(selected_loras)

                if is_4bit_model:
                    command.append("--is_4bit")

                with st.spinner(f"Converting '{selected_base_model_deploy}' to GGUF..."):
                    st.write("Running command:")
                    st.code(" ".join(command), language="bash")

                    result = subprocess.run(command, capture_output=True, text=True)

                    if result.returncode == 0:
                        st.success("GGUF conversion complete. You can now deploy it below.")
                        st.code(result.stdout)
                    else:
                        st.error("GGUF conversion failed.")
                        st.code(result.stderr)
                        st.stop()

        st.subheader("Convert Model to AWQ")
        awq_output_path = st.text_input("AWQ Output Path", placeholder="e.g., ./awq_models/my_model_awq")
        if st.button("Convert to AWQ"):
            if not all([selected_base_model_deploy, awq_output_path]):
                st.error("Please select a base model and provide an output path for AWQ.")
            else:
                model_path = os.path.join(base_model_dir, selected_base_model_deploy)
                command = get_docker_compose_command()
                command.extend([
                    "exec", "unsloth", "python", "/app/quantize_to_awq.py",
                    "--model_path", model_path,
                    "--output_path", awq_output_path
                ])

                # Add lora if selected
                if selected_loras:
                    command.append("--lora_names")
                    command.extend(selected_loras)
                
                if is_4bit_model:
                    command.append("--is_4bit")

                with st.spinner(f"Quantizing '{selected_base_model_deploy}' to AWQ..."):
                    st.write("Running command:")
                    st.code(" ".join(command), language="bash")
                    result = subprocess.run(command, capture_output=True, text=True)
                    if result.returncode == 0:
                        st.success("AWQ quantization complete.")
                        st.code(result.stdout)
                    else:
                        st.error("AWQ quantization failed.")
                        st.code(result.stderr)
                        st.stop()


        st.subheader("Deploy GGUF to Ollama")
        gguf_dir = "./gguf_models"
        os.makedirs(gguf_dir, exist_ok=True)
        available_ggufs = [f for f in os.listdir(gguf_dir) if f.endswith(".gguf")]
        selected_gguf = st.selectbox("Select GGUF file to deploy:", options=available_ggufs)
        ollama_model_name = st.text_input("Enter a name for the new Ollama model:", placeholder="e.g., my-custom-model")

        if st.button("Deploy to Ollama"):
            if not all([selected_gguf, ollama_model_name]):
                st.error("Please select a GGUF file and enter a name for the Ollama model.")
            else:
                with st.spinner(f"Deploying '{selected_gguf}' to Ollama as '{ollama_model_name}'..."):
                    gguf_path_in_container = os.path.join("/app/gguf_models", selected_gguf)
                    modelfile_content = f"FROM {gguf_path_in_container}"
                    create_payload = { "name": f"{ollama_model_name}:latest", "modelfile": modelfile_content }

                    try:
                        response = requests.post(f"{OLLAMA_GPU_URL}/api/create", json=create_payload, stream=True)
                        response.raise_for_status()

                        st.write("Ollama deployment response:")
                        response_placeholder = st.empty()
                        full_response = ""
                        for line in response.iter_lines():
                            if line:
                                decoded_line = line.decode('utf-8')
                                try:
                                    json_line = json.loads(decoded_line)
                                    if 'status' in json_line:
                                        full_response += json_line['status'] + "\n"
                                        response_placeholder.text(full_response)
                                except json.JSONDecodeError:
                                    full_response += decoded_line + "\n"
                                    response_placeholder.text(full_response)
                        st.success(f"Model '{ollama_model_name}' deployment process finished.")
                        st.rerun()
                    except requests.exceptions.RequestException as e:
                        st.error(f"Failed to deploy to Ollama: {e}")


# --- Main Chat Interface ---
st.title("LLM Agent Interface")

if st.session_state.current_session_id:
    st.session_state.messages = st.session_state.sessions[st.session_state.current_session_id]['messages']

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is your question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        with st.spinner("Thinking..."):
            client = None
            model_name = None
            context = ""

            # Determine which client and model to use
            if st.session_state.inference_engine == "Ollama":
                if not st.session_state.get("ollama_client"):
                    st.error("Ollama client is not initialized. Please select a model.")
                    st.stop()
                client = st.session_state.ollama_client
                model_name = st.session_state.selected_chat_model

            elif st.session_state.inference_engine == "vLLM":
                if not st.session_state.get("vllm_server_running"):
                    st.error("vLLM server is not running. Please start it.")
                    st.stop()
                client = st.session_state.vllm_client
                model_name = os.path.join("/app/base_models", st.session_state.vllm_base_model)

            elif st.session_state.inference_engine == "Unsloth":
                 if not st.session_state.get("unsloth_server_running"):
                    st.error("Unsloth server is not running. Please start it from the sidebar.")
                    st.stop()

            # --- Tool/Function Routing ---
            if prompt.lower().startswith("search:"):
                query = prompt.split("search:", 1)[1].strip()
                st.write(f"Performing explicit web search for: '{query}'")
                context = perform_web_search(query)
            else:
                router_prompt = f"""
                Given the user's query, decide if you should use a web search for up-to-date information
                or a local document search (RAG) for contextual information.

                User Query: "{prompt}"

                Respond with "WEB SEARCH" or "RAG".
                """
                decision = "RAG" # Default to RAG

                if st.session_state.inference_engine in ["Ollama", "vLLM"]:
                    try:
                        router_response = client.chat.completions.create(
                            messages=[{"role": "user", "content": router_prompt}],
                            model=model_name,
                            temperature=0,
                            max_tokens=10
                        )
                        decision = router_response.choices[0].message.content.strip().upper()
                    except Exception as e:
                        st.error(f"Error in routing: {e}")

                elif st.session_state.inference_engine == "Unsloth":
                    try:
                        payload = {"prompt": router_prompt, "history": []}
                        response = requests.post(f"{UNSLOTH_URL}/generate", json=payload)
                        response.raise_for_status()
                        decision = response.json().get("response", "RAG").strip().upper()
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error in routing: {e}")

                st.write(f"LLM decided to use: {decision}")

                if "WEB SEARCH" in decision:
                    context = perform_web_search(prompt)
                else:  # Default to RAG
                    st.write("Performing RAG retrieval...")
                    retriever = st.session_state.rag_handler.get_retriever()
                    retrieved_nodes = retriever.retrieve(prompt)
                    context = "\n".join([node.get_content() for node in retrieved_nodes])


            # --- Generating the Final Response ---
            enriched_prompt = f"""You are a helpful assistant. 
Answer the user's question using ONLY the "Context" information provided below. 
If the answer cannot be found within the context, respond with "I don't know." 
Do not use your own knowledge under any circumstances. 
If the context consists of web search results, cite the source URLs in your answer.
あなたは親切なアシスタントです。以下の「コンテキスト」情報だけを使って、ユーザーの質問に答えてください。
コンテキストから答えが見つからない場合は、「分かりません」と答えてください。
絶対にあなた自身の知識を使わないでください。
コンテキストがWeb検索結果である場合は、回答に情報源のURLを引用してください。

# コンテキスト
{context}

# ユーザーの質問
{prompt}
"""

            if st.session_state.inference_engine == "Unsloth":
                 try:
                    payload = { "prompt": enriched_prompt, "history": st.session_state.messages[:-1] }
                    response = requests.post(f"{UNSLOTH_URL}/generate", json=payload)
                    response.raise_for_status()
                    full_response = response.json().get("response", "")
                 except requests.exceptions.RequestException as e:
                    full_response = f"An error occurred during Unsloth inference: {str(e)}"

            elif client and model_name:
                try:
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": enriched_prompt},
                        ],
                        model=model_name,
                        temperature=0.1,
                    )
                    full_response = chat_completion.choices[0].message.content
                except Exception as e:
                    full_response = f"An error occurred during inference: {str(e)}"

            message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

    current_session = st.session_state.sessions[st.session_state.current_session_id]
    current_session['messages'] = st.session_state.messages
    current_session['last_updated'] = int(time.time())
    save_all_sessions()

    with st.spinner("Updating short-term memory (RAG)..."):
        if st.session_state.rag_handler:
            conversation_text = f"User: {prompt}\nAssistant: {full_response}"
            st.session_state.rag_handler.add_text_to_rag(conversation_text)
            st.toast("Short-term memory updated.")
        else:
            st.toast("RAG handler not available. Skipping short-term memory update.")
