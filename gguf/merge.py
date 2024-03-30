from transformers import AutoModelForCausalLM, GemmaForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_model(model_src: str, model_adapter: str, model_out: str):
    model: GemmaForCausalLM = AutoModelForCausalLM.from_pretrained(model_src)
    tokenizer = AutoTokenizer.from_pretrained(model_src)

    model2 = PeftModel.from_pretrained(model, model_adapter)

    model2.merge_and_unload(progressbar=True, safe_merge=True)

    model.save_pretrained(model_out)
    tokenizer.save_pretrained(model_out)


if __name__ == "__main__":
    pass
