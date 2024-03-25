from datasets.arrow_writer import SchemaInferenceError
from datasets.exceptions import DatasetGenerationError
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from trl.trainer import ConstantLengthDataset


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


if __name__ == "__main__":
    MODEL = "unsloth/gemma-7b-bnb-4bit"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    dataset = Dataset.load_from_disk("../train-axo/dataset-save-to-disk/")

    print("Loaded", len(dataset))

    print("Tokenizing...")
    tokenized = _prepare_packed_dataloader(
        tokenizer,
        dataset,
        "text",
        2048,
        1024,  # default
        3.6,  # default
        None,
    )

    print("Tokenized", len(tokenized))

    print("Columns", tokenized.column_names)

    tokenized.save_to_disk("./tokenized")

