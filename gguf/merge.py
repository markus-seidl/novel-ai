from transformers import AutoModelForCausalLM, GemmaForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_model(model_src: str, model_adapter: str, model_out: str):
    max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
    dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = False  # Use 4bit quantization to reduce memory usage. Can be False.

    model: GemmaForCausalLM = AutoModelForCausalLM.from_pretrained(model_src)
    tokenizer = AutoTokenizer.from_pretrained(model_src)

    model2 = PeftModel.from_pretrained(model, model_adapter)

    model2.merge_and_unload(progressbar=True, safe_merge=True)

    model.save_pretrained(model_out)
    tokenizer.save_pretrained(model_out)


if __name__ == "__main__":
    pass
