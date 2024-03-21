import os
import time

from huggingface_hub import hf_hub_download
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from nltk import sent_tokenize

# import torch
# import transformers
# import textwrap
# from llama_cpp import Llama
# from transformers import AutoTokenizer
# from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

PERFORMANCE = {
    "processed_chars": 0,
    "total_time": 0  # seconds
}


def performance_info() -> str:
    chars_per_second = PERFORMANCE["total_time"] / max(PERFORMANCE["processed_chars"], 1)
    return f"{chars_per_second} s/char"


# MODEL_ID = "TheBloke/Wizard-Vicuna-7B-Uncensored-HF"
MODEL_ID = "TheBloke/Wizard-Vicuna-7B-Uncensored-GGUF"
MODEL_FILENAME = "Wizard-Vicuna-7B-Uncensored.Q2_K.gguf"

MODEL_ID = "TheBloke/Wizard-Vicuna-13B-Uncensored-GGUF"
MODEL_FILENAME = "Wizard-Vicuna-13B-Uncensored.Q5_K_S.gguf"

# MODEL_NAME = "https://huggingface.co/TheBloke/Wizard-Vicuna-7B-Uncensored-GGML/resolve/main/Wizard-Vicuna-7B-Uncensored.ggmlv3.q2_K.bin"

MODEL_PATH = os.path.dirname(os.path.abspath(__file__)) + "/../../temp/"
MODEL: LLMChain = None
MODEL_SHORT: LLMChain = None
LLM: LlamaCpp = None
TEXT_SPLITTER: RecursiveCharacterTextSplitter = None

MAX_CTX_SIZE = 2048

TEMPLATE = """
              Write a summary of the following text delimited by triple backticks.
              Return your response by focusing on plot points and write in present tense.
              ```{text}```
              SUMMARY:
           """

TEMPLATE_SHORT = """
              Write a very short summary of the following text delimited by triple backticks.
              Return your response by focusing on plot points and write in present tense.
              ```{text}```
              SUMMARY:
           """


def ensure_model():
    global MODEL, LLM, TEXT_SPLITTER, MODEL_SHORT
    if MODEL is not None:
        return

    # llm = Llama.from_pretrained(
    #     repo_id=MODEL_ID,
    #     filename=MODEL_FILENAME,
    #     verbose=False
    # )

    print("Loading or downloading model...")
    model_path = hf_hub_download(repo_id=MODEL_ID, filename=MODEL_FILENAME, cache_dir=MODEL_PATH)

    n_gpu_layers = -1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
    n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.

    kwargs = {
        "model_path": model_path,
        "temperature": 0.75,
        "max_tokens": MAX_CTX_SIZE,
        "n_ctx": MAX_CTX_SIZE,
        "n_gpu_layers": n_gpu_layers,
        "n_batch": n_batch,
        # "f16_kv": True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        "verbose": "true" == os.getenv("LLAMACPP_VERBOSE") or False,
    }
    LLM = LlamaCpp(**kwargs)
    prompt = PromptTemplate(template=TEMPLATE, input_variables=["text"])
    MODEL = LLMChain(prompt=prompt, llm=LLM)

    prompt_short = PromptTemplate(template=TEMPLATE_SHORT, input_variables=["text"])
    MODEL_SHORT = LLMChain(prompt=prompt_short, llm=LLM)
    TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=MAX_CTX_SIZE, chunk_overlap=100, length_function=len)


def split_into_chunks(text):
    ensure_model()
    return TEXT_SPLITTER.split_text(text)


def summarize_text(chunk):
    ensure_model()
    PERFORMANCE['processed_chars'] += len(chunk)

    start_time = time.time()
    summary = MODEL.invoke(chunk)['text']

    if len(summary) > len(chunk) * 0.5:
        summary = MODEL_SHORT.invoke(chunk)['text']

    summary_sentences = sent_tokenize(summary)

    PERFORMANCE['total_time'] += (time.time() - start_time)

    # clean output a bit
    return " ".join(summary_sentences)


def summarize_text_v2(text, length):
    ensure_model()
    doc = Document(page_content=text)
    chain = load_summarize_chain(LLM, chain_type="map_reduce")

    return chain.run([doc])


if __name__ == "__main__":
    import sys

    sys.path.insert(1, '../')
    import mycrypt

    text = mycrypt.load_file_txt(
        "....txt.zst.enc"
    )

    chunks = split_into_chunks(text)
    result = summarize_text(chunks[0])
    print(result)
    print(len(result), " vs ", len(chunks[0]))
