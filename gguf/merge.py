from transformers import AutoModelForCausalLM, GemmaForCausalLM, AutoTokenizer
from peft import PeftModel

max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False  # Use 4bit quantization to reduce memory usage. Can be False.
MODEL_ID = "unsloth/gemma-7b-it"

model: GemmaForCausalLM = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model2 = PeftModel.from_pretrained(model, "./checkpoint-400/")

model2.merge_and_unload(progressbar=True, safe_merge=True)
# model.load_adapter("./checkpoint-400/")
# model.merge_and_unload()

model.save_pretrained("./merged/")
tokenizer.save_pretrained("./merged/")

if __name__ == "__main__":
    pass
