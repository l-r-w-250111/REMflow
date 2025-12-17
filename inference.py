# Ensure unsloth is imported before torch, transformers, and peft
from unsloth import FastLanguageModel
import torch
import argparse
import config

def run_inference(lora_model_path, is_cpt_model):
    """
    Loads a fine-tuned LoRA model and runs interactive inference,
    handling both SFT and CPT models correctly.
    """
    print("Loading LoRA model for inference...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=lora_model_path,
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    model.eval()

    print("\nInteractive inference started. Type 'exit' to quit.")
    if is_cpt_model:
        print("CPT model loaded. The model will complete the prompt you enter.")
        print("Example: 'The capital of France is'")
    else:
        print("SFT model loaded. You can chat with the model.")
        print("Example: 'What is the capital of France?'")
    
    conversation_history = []

    while True:
        try:
            user_input = input(">> User: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            if is_cpt_model:
                # For CPT, we don't use a chat template, just the raw prompt.
                prompt = user_input
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            else:
                # For SFT, manage conversation history and use the chat template.
                conversation_history.append({"role": "user", "content": user_input})
                prompt = tokenizer.apply_chat_template(conversation_history, tokenize=False, add_generation_prompt=True)
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

            # Generate the response
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=256,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id # Suppress warning
                )
            
            # More robustly decode only the generated part of the response
            generated_ids = outputs[0][input_ids.shape[1]:]
            response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            print(f">> Assistant: {response_text}")

            if not is_cpt_model:
                # Add assistant's response to history for SFT models
                conversation_history.append({"role": "assistant", "content": response_text})

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            # Reset history on error to prevent cascading failures
            conversation_history = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run interactive inference with a trained LoRA model.")
    parser.add_argument("--lora_model_path", type=str, default=config.LORA_MODEL_PATH,
                        help="Path to the directory containing the trained LoRA adapter.")
    parser.add_argument("--is_cpt_model", action="store_true",
                        help="Set this flag if the model is a CPT model to avoid chat templates.")
    args = parser.parse_args()

    run_inference(args.lora_model_path, args.is_cpt_model)
