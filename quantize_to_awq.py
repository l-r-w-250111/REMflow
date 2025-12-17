import argparse
import os
import shutil
import torch
from unsloth import FastLanguageModel
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter and quantize the model to AWQ format.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the base model directory.")
    parser.add_argument("--lora_names", type=str, nargs='*', help="Optional: List of LoRA adapter names under lora_models/.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the quantized AWQ model directory.")
    parser.add_argument("--is_4bit", action="store_true", help="Flag if the base model is 4-bit, used for loading.")
    args = parser.parse_args()

    temp_fp16_path = "./temp_fp16_model_for_awq"

    try:
        # --- 1. Load Model and Merge LoRA using Unsloth ---
        print("Loading base model and applying LoRA adapter (if specified)...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_path,
            load_in_4bit=args.is_4bit,
            dtype=torch.bfloat16 if not args.is_4bit else None,
            # trust_remote_code is needed for some models like Phi-3
            trust_remote_code=True,
        )

        if args.lora_names:
            print(f"Applying {len(args.lora_names)} LoRA adapter(s)...")
            for lora_name in args.lora_names:
                lora_path = os.path.join("./lora_models", lora_name)
                if not os.path.isdir(lora_path):
                    raise FileNotFoundError(f"LoRA adapter directory not found: {lora_path}")
                print(f"Loading adapter: {lora_name}")
                model.load_adapter(lora_path)

            print("All LoRA adapters loaded. Merging...")
            # Unsloth's merge_and_unload will merge all loaded adapters.
            model = model.merge_and_unload()
            print("LoRA adapters successfully merged.")

        # --- 2. Save Merged Model to FP16 Hugging Face Format ---
        print(f"Saving merged model to temporary FP16 directory: {temp_fp16_path}")
        if os.path.exists(temp_fp16_path):
            shutil.rmtree(temp_fp16_path)

        model.save_pretrained(temp_fp16_path, save_dtype=torch.float16)
        tokenizer.save_pretrained(temp_fp16_path)
        print("Merged model saved in FP16 format successfully.")

        # --- 3. Quantize the Merged Model to AWQ ---
        print(f"Starting AWQ quantization for model at: {temp_fp16_path}")

        # Define quantization configuration for AWQ
        quant_config = {"w_bit": 4, "q_group_size": 128, "zero_point": True, "version": "GEMM"}

        # Load the FP16 model for quantization
        awq_model = AutoAWQForCausalLM.from_pretrained(temp_fp16_path, safetensors=True)
        awq_tokenizer = AutoTokenizer.from_pretrained(temp_fp16_path, trust_remote_code=True)

        # Quantize the model
        print("Quantizing model...")
        awq_model.quantize(awq_tokenizer, quant_config=quant_config)
        print("Model quantization complete.")

        # --- 4. Save the Quantized AWQ Model ---
        print(f"Saving quantized AWQ model to: {args.output_path}")
        # The save_quantized method saves the model files and the tokenizer
        awq_model.save_quantized(args.output_path)
        awq_tokenizer.save_pretrained(args.output_path)

        print(f"Successfully created and saved AWQ model at: {args.output_path}")

    finally:
        # --- 5. Cleanup ---
        print("Cleaning up temporary directory...")
        if os.path.exists(temp_fp16_path):
            shutil.rmtree(temp_fp16_path)
            print(f"Removed temporary directory: {temp_fp16_path}")

if __name__ == "__main__":
    main()
