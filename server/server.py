from llama_cpp import Llama

prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Continue the novel given at the input, write in third person and use direct speech. The synopsis is as follows: {}

### Input:
{}

### Response:
{}"""

llm = Llama(
    model_path="/Users/augunrik/temp/huggingface/llama.cpp/ggml-model-Q4_K_M.gguf",
    # n_gpu_layers=-1, # Uncomment to use GPU acceleration
    # seed=1337, # Uncomment to set a specific seed
    # n_ctx=2048, # Uncomment to increase the context window
)

synopsis = """Miranda, a vicious, very beautiful witch, which transforms Robert, an alpha male who treads women badly, into her pussy. Robert is disgusted and blown out of his mind, about the transformation. Miranda is delighted, about Roberts disgust. She conjours a Golem with the largest, hardest dick Robert has ever seen. The Golem starts fucking the Robert pussy. Robert feels the Golems dick as it would be in his mouth, which is a whole new experience for him. Then the Golem cums."""

prompt = prompt_template.format(synopsis, "Richard looks in shock at the Golems' cock. It was so much larger than his.",
                                "")

print(llm.create_completion(
    prompt,
    temperature=0,
    repeat_penalty=1.0,
    max_tokens=32,  # Generate up to 32 tokens, set to None to generate up to the end of the context window
    stop=["Q:", "\n"],  # Stop generating just before the model would generate a new question
    echo=True  # Echo the prompt back in the output
))

if __name__ == "__main__":
    pass
