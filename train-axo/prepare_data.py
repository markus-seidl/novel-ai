#!/usr/bin/python3

import sys
from datasets import Dataset

sys.path.insert(1, '../dataset/')

import loader_llm_21

if __name__ == '__main__':
    local_temp = "./temp/dl/"

    my_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        Continue the novel given at the input, write in third person and use direct speech. The synopsis is as follows: {}

        ### Input:
        """


    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = my_prompt.format(instruction, input)
            texts.append(text)
        return {"text": texts, "prompt": texts, }

    ds: Dataset = loader_llm_21.load_novel_dataset(local_temp, formatting_prompts_func, split=False)
    ds.to_json("./dataset/data.jsonl")
    # ds.save_to_disk("./dataset-save-to-disk")
