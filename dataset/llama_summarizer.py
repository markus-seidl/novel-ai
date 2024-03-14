import os

from gpt4all import GPT4All
from nltk import sent_tokenize
import urllib.request

# MODEL_NAME = "llama2-13b-tiefighterlr.Q4_0.gguf"
# MODEL_NAME = "MistralRP-Noromaid-NSFW-7B-Q4_0.gguf"
# Very bad: MODEL_NAME = "uncensored-jordan-33b.Q5_K_M.gguf"
# MODEL_NAME = "Wizard-Vicuna-30B-Uncensored.Q5_K_S.gguf"

MODEL_DL = "https://huggingface.co/TheBloke/Wizard-Vicuna-30B-Uncensored-GGUF/resolve/main/Wizard-Vicuna-30B-Uncensored.Q5_K_S.gguf"
MODEL_NAME = "Wizard-Vicuna-30B-Uncensored.Q5_K_S.gguf"

# MODEL_DL = "https://huggingface.co/TheBloke/TinyLlama-1.1B-intermediate-step-1431k-3T-GGUF/resolve/main/tinyllama-1.1b-intermediate-step-1431k-3t.Q5_K_S.gguf"
# MODEL_NAME = "tinyllama-1.1b-intermediate-step-1431k-3t.Q5_K_S.gguf"

MODEL_PATH = os.path.dirname(os.path.abspath(__file__)) + "/../temp/"
MODEL = None

TEMPLATE = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n### USER: {0}\n### ASSISTANT:\n"


def download_model():
    print(f"Downloading summarizer model: {MODEL_NAME}")
    urllib.request.urlretrieve(MODEL_DL, MODEL_PATH + MODEL_NAME)


def ensure_model():
    global MODEL
    if MODEL is None:
        print("Loading model...")
        MODEL = GPT4All(model_name=MODEL_NAME,
                        model_path=MODEL_PATH)
        print("...done.")


def summarize_text(text, length):
    ensure_model()
    with MODEL.chat_session(TEMPLATE):
        # response = MODEL.generate(f"Summarize the following text into {length} sentences: {text}")
        response = MODEL.generate(
            f"Extract the three key points from the following text and summarize it into {length} sentences: {text}"
        )
    return response
