import trl


class CustomSFTTrainer(trl.SFTTrainer):
    def _prepare_dataset(
            self,
            dataset,
            tokenizer,
            packing,
            dataset_text_field,
            max_seq_length,
            formatting_func,
            num_of_sequences,
            chars_per_token,
            remove_unused_columns=True,
            append_concat_token=True,
            add_special_tokens=True,
    ):
        # NOP here, this allows to pass in a pre-tokenized dataset and skip the mangling on creation
        return dataset

