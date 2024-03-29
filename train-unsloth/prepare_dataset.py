from datasets.arrow_writer import SchemaInferenceError
from datasets.exceptions import DatasetGenerationError
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from trl.trainer import ConstantLengthDataset

import sys

sys.path.insert(1, '../dataset/')
import loader_llm_21


def _prepare_packed_dataloader(
        tokenizer,
        dataset,
        dataset_text_field,
        max_seq_length,
        num_of_sequences,
        chars_per_token,
        formatting_func=None,
        append_concat_token=True,
        add_special_tokens=True,
):
    if dataset_text_field is not None or formatting_func is not None:
        if tokenizer is None:
            raise ValueError("You need to pass a tokenizer when using `dataset_text_field` with `SFTTrainer`.")

        constant_length_iterator = ConstantLengthDataset(
            tokenizer,
            dataset,
            dataset_text_field=dataset_text_field,
            formatting_func=formatting_func,
            seq_length=max_seq_length,
            infinite=False,
            num_of_sequences=num_of_sequences,
            chars_per_token=chars_per_token,
            eos_token_id=tokenizer.eos_token_id,
            append_concat_token=append_concat_token,
            add_special_tokens=add_special_tokens,
        )

        def data_generator(constant_length_iterator):
            yield from constant_length_iterator

        try:
            packed_dataset = Dataset.from_generator(
                data_generator, gen_kwargs={"constant_length_iterator": constant_length_iterator}
            )
        except (DatasetGenerationError, SchemaInferenceError) as exc:
            raise ValueError(
                "Error occurred while packing the dataset. "
                "Make sure that your dataset has enough samples to at least yield one packed sequence."
            ) from exc
        return packed_dataset
    else:
        raise ValueError(
            "You need to pass a `dataset_text_field` or `formatting_func` argument to the SFTTrainer if you want to use the `ConstantLengthDataset`."
        )


def prepare_first_step_dataset(local_temp: str, output_dir: str, EOS_TOKEN):
    my_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
        ### Instruction:
        Continue the novel given at the input, write in third person and use direct speech. The synopsis is as follows: {}

        ### Input:
        {}
        """

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = my_prompt.format(instruction, input, "") + EOS_TOKEN
            texts.append(text)
        return {"text": texts }  # , "prompt": texts,

    ds: Dataset = loader_llm_21.load_novel_dataset(local_temp, formatting_prompts_func, split=False)
    # ds.to_json(json_output_file)  # "./dataset/data.jsonl"
    ds.save_to_disk(output_dir)


def generate_tokenized_dataset(
        model: str, output_path_train: str, output_path_test: str,
        test_size=0.01, shuffle=True, seed=0xAFFE,
):
    tokenizer = AutoTokenizer.from_pretrained(model)

    prepare_first_step_dataset("./raw_data/", "./untokenized_data/", tokenizer.eos_token)

    raw_dataset = Dataset.load_from_disk("./untokenized_data")

    print("Loaded raw data set for tokenizing: ", len(raw_dataset))

    dataset = raw_dataset.train_test_split(test_size=test_size, shuffle=shuffle, seed=seed)

    print("Tokenizing...")
    tokenized_train = _prepare_packed_dataloader(
        tokenizer,
        dataset['train'],
        "text",
        2048,
        1024,  # default
        3.6,  # default
        None,
    )
    tokenized_test = _prepare_packed_dataloader(
        tokenizer,
        dataset['test'],
        "text",
        2048,
        1024,  # default
        3.6,  # default
        None,
    )

    print("Tokenized: ", len(tokenized_train), len(tokenized_test))

    tokenized_train.save_to_disk(output_path_train)
    tokenized_test.save_to_disk(output_path_test)


if __name__ == "__main__":
    # MODEL = "unsloth/gemma-7b-it"
    MODEL = "unsloth/tinyllama"

    generate_tokenized_dataset(
        MODEL,
        "./tokenized_train",
        "./tokenized_test"
    )
