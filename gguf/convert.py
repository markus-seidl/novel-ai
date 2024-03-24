from unsloth import FastLanguageModel
import torch

max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False  # Use 4bit quantization to reduce memory usage. Can be False.
model = "~/Downloads/tinyllama-240323_0947/"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model,  # "unsloth/tinyllama" for 16bit loading
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model.save_pretrained_gguf("tinyllama", tokenizer, quantization_method = "q4_k_m")

if __name__ == "__main__":
    pass
