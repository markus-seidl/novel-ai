import json
import os
from webdav3.client import Client
from tqdm.auto import tqdm
import mycrypt
from entities import Book, SummaryBook, SummaryChunk, SummaryChapter
import multiprocessing
from datasets import Dataset


def download_input_data(webdav_folder: str, local_folder: str, max_files) -> [str]:
    webdav_client = Client({
        'webdav_hostname': os.environ.get('WEBDAV_HOSTNAME'),
        'webdav_login': os.environ.get('WEBDAV_USER'),
        'webdav_password': os.environ.get('WEBDAV_PASS'),
        'disable_check': True
    })

    downloaded = 0
    remote_files = webdav_client.list(webdav_folder)
    pbar = tqdm(remote_files)
    for file in pbar:
        local_file = local_folder + '/' + file
        pbar.set_description(f"Downloading {file}")

        if not os.path.exists(local_file):
            webdav_client.download_file(webdav_folder + '/' + file, local_file)

        if 0 < max_files < downloaded:
            break

        downloaded += 1


def transform_book(
        book: SummaryBook, cnt_previous_sentences=3, min_prev_length=50, cnt_next_sentences=2, min_next_length=200,
        window_step_size=3
) -> [{str, str}]:
    ret = []
    for chapter in book.chapters:
        for chunk in chapter.chunks:

            sentences = chunk.sentences
            if len(" ".join(sentences)) < min_next_length:
                continue

            # create empty prompt with summary
            # ret.append({
            #     "instruction": chunk.summary,
            #     "input": "",
            #     "output": " ".join(sentences[0:cnt_next_sentences]),
            # })

            for idx in range(cnt_previous_sentences, len(sentences), window_step_size):
                # Fill until min_prev_length is reached
                previous_sentences_text = ""
                prev_idx = idx - cnt_previous_sentences
                while len(previous_sentences_text) < min_prev_length and prev_idx >= 0:
                    previous_sentences_text = " ".join(sentences[prev_idx:idx])
                    prev_idx -= 1

                # Fill until min_next_length is reached
                next_sentences_text = ""
                next_idx = idx + cnt_next_sentences
                while len(next_sentences_text) < min_next_length and next_idx < len(sentences):
                    next_sentences_text = " ".join(sentences[idx:next_idx])
                    next_idx += 1

                if len(chunk.summary) == 0 or len(previous_sentences_text) == 0 or len(next_sentences_text) == 0:
                    continue

                ret.append({
                    "instruction": chunk.summary,
                    "input": previous_sentences_text,
                    "output": next_sentences_text,
                })

    return ret


def load_summary_book(json_string: str) -> SummaryBook:
    data = json.loads(json_string)

    chapters = []
    for chapter_data in data['chapters']:
        chunks = [SummaryChunk(**chunk) for chunk in chapter_data['chunks']]
        chapter = SummaryChapter(title=chapter_data['title'], chunks=chunks)
        chapters.append(chapter)

    return SummaryBook(
        title=data['title'],
        file_id=data['file_id'],
        chapters=chapters,
    )


def transform_input_data(local_folder: str, max_files: int) -> [{str, str}]:
    ret: [{str, str}] = []
    transformed = 0
    pbar = tqdm(os.listdir(local_folder))
    for file in pbar:
        pbar.set_description(f"Transforming {file}")
        book_summary_json = mycrypt.load_file_txt(local_folder + '/' + file)
        book_summary = load_summary_book(book_summary_json)

        data = transform_book(book_summary)
        ret.extend(data)

        if 0 < max_files < transformed:
            break

        transformed += 1

    return ret


def load_novel_dataset(local_temp, formatting_prompts_func, test_data_size_percent=0.15, num_proc=1, max_files=-1, split=True):
    os.makedirs(local_temp, exist_ok=True)
    download_input_data('/output_llm_dataset/', local_temp, max_files)
    data = transform_input_data(local_temp, max_files)
    print("Generated ", len(data))
    ds = Dataset.from_list(data)

    if formatting_prompts_func is not None:
        ds = ds.map(formatting_prompts_func, batched=True, num_proc=num_proc)

    if split:
        ds = ds.train_test_split(test_size=test_data_size_percent, shuffle=True, seed=0xAFFE)
        print("Train size: ", len(ds['train']), "Test size: ", len(ds['test']))
    return ds


if __name__ == '__main__':
    local_temp = "../temp/data/"
    my_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    Continue the novel given at the input, write in third person and use direct speech. The synopsis is as follows: {}

    ### Input:
    {}

    ### Response:
    {}"""
    EOS_TOKEN = "TODO"


    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = my_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts, }


    ds = load_novel_dataset(local_temp, formatting_prompts_func)
