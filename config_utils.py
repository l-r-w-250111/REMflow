import re
import os
import json
import config

def update_selected_chat_model(new_path):
    """
    Updates the SELECTED_CHAT_MODEL_PATH in the config.py file.

    Args:
        new_path (str): The new path to the selected chat model.
    """
    config_path = os.path.join(os.path.dirname(__file__), 'config.py')

    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Use repr() to ensure the path string is correctly quoted
    replacement_value = repr(new_path)

    # Use regex to replace the value of SELECTED_CHAT_MODEL_PATH
    new_content, count = re.subn(
        r"^(SELECTED_CHAT_MODEL_PATH\s*=\s*).*$",
        f"SELECTED_CHAT_MODEL_PATH = {replacement_value}",
        content,
        flags=re.MULTILINE
    )

    if count > 0:
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
    else:
        # If the variable wasn't found, append it.
        with open(config_path, 'a', encoding='utf-8') as f:
            f.write(f"\nSELECTED_CHAT_MODEL_PATH = {replacement_value}\n")

def get_base_model_from_lora(lora_path):
    """
    Reads the adapter_config.json from a LoRA directory to find the base model name.

    Args:
        lora_path (str): The path to the LoRA adapter directory.

    Returns:
        str: The name/path of the base model.

    Raises:
        FileNotFoundError: If the adapter config cannot be found.
        ValueError: If the base model path is not specified in the config.
    """
    if not lora_path or not os.path.isdir(lora_path):
        raise FileNotFoundError(f"LoRA path '{lora_path}' is not a valid directory.")

    config_file = os.path.join(lora_path, 'adapter_config.json')

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Adapter config not found at: {config_file}")

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            adapter_config = json.load(f)

        base_model = adapter_config.get('base_model_name_or_path')
        if base_model:
            return base_model
        else:
            raise ValueError(f"'base_model_name_or_path' not found in {config_file}")

    except (json.JSONDecodeError, IOError) as e:
        raise IOError(f"Error reading or parsing LoRA config file at {config_file}: {e}")
