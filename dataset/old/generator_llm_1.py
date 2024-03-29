import dataclasses
import datetime
import json
import os
import random
import socket

import zstandard

from entities import Book, TrainingData
from book_parser import load_book_from_text

from tqdm.auto import tqdm, trange
import mycrypt
from webdav3.client import Client, RemoteResourceNotFound
from nltk import sent_tokenize

SUMMARIZER = "LLM"
import sumi.llama_summarizer_v2 as summarizer

WEBDAV_CLIENT = Client({
    'webdav_hostname': os.environ.get('WEBDAV_HOSTNAME'),
    'webdav_login': os.environ.get('WEBDAV_USER'),
    'webdav_password': os.environ.get('WEBDAV_PASS'),
    'disable_check': True
})
print("Connected to webdav, root contents: ", WEBDAV_CLIENT.list("/"))

CONTAINER_LABEL = os.environ.get("VAST_CONTAINERLABEL") or "UNKNOWN"

PREVIOUS_SENTENCES = 3
MIN_PREV_LENGTH = 50

NEXT_SENTENCES = 2
MIN_NEXT_LENGTH = 50

WINDOW_STEP_SIZE = 3


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't even have to be reachable
        s.connect(('google.de', 1))
        ip = str(s.getsockname()[0]).replace(".", "_")
    except Exception:
        ip = 'localhost'
    finally:
        s.close()
    return ip


ALIVE_FILE = f"im_alive_{os.getpid()}_{CONTAINER_LABEL}.txt"


def inform_alive(current_md5):
    with open(ALIVE_FILE, "w+") as f:
        f.write(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " " +
            summarizer.performance_info() + " " +
            current_md5 + " " +
            SUMMARIZER + " " +
            CONTAINER_LABEL + "\n"
        )
    WEBDAV_CLIENT.upload_file(ALIVE_FILE, ALIVE_FILE)


# precompiled_summaries : ("md5", "chapter title") : "summary"
def convert_to_trainingdata(
        book: Book, title: str, file_id: str
) -> [TrainingData]:
    ret = []
    chapter_bar = tqdm(book.chapters, leave=False)
    for chapter in chapter_bar:
        chapter_bar.set_description("Chapter " + str(chapter.title))

        chunks = summarizer.split_into_chunks(" ".join(chapter.sentences))

        inform_alive(file_id)

        for chunk in tqdm(chunks):
            chunk_summary = summarizer.summarize_text(chunk)

            chunk_sentences = sent_tokenize(chunk)

            for i in range(0, len(chunk_sentences), WINDOW_STEP_SIZE):
                # Fill until MIN_PREV_LENGTH is reached
                previous_sentences_text = ""
                prev_idx = i - PREVIOUS_SENTENCES
                while len(previous_sentences_text) < MIN_PREV_LENGTH and prev_idx >= 0:
                    previous_sentences_text = " ".join(chapter.sentences[prev_idx:i])
                    prev_idx -= 1

                # Fill until MIN_NEXT_LENGTH is reached
                next_sentences_text = ""
                next_idx = i + NEXT_SENTENCES
                while len(next_sentences_text) < MIN_NEXT_LENGTH and next_idx < len(chapter.sentences):
                    next_sentences_text = " ".join(chapter.sentences[i:next_idx])
                    next_idx += 1

                # With Input Text
                temp = TrainingData(
                    book_title=title,
                    chapter_title=chapter.title,
                    summary=chunk_summary,
                    previous_sentences=previous_sentences_text,
                    expected_answer=next_sentences_text,
                    sum_type=SUMMARIZER
                )
                ret.append(temp)

    return ret


def write_and_compress(training_datas: [TrainingData], outfile: str):
    json_data = "\n".join(json.dumps(dataclasses.asdict(dc)) for dc in training_datas)

    mycrypt.save_file_txt(json_data, outfile)


def generate_for(input_file: str, output_file: str, title: str, precompiled_summaries: {(str, str): str}, md5):
    input_file_text = mycrypt.load_file_txt(input_file)
    book = load_book_from_text(input_file_text)
    tds = convert_to_trainingdata(book, title, md5)
    write_and_compress(tds, output_file)


def webdav_exists_file(file_path: str) -> bool:
    try:
        info = WEBDAV_CLIENT.info(file_path)
        return info['size'] != 0
    except RemoteResourceNotFound:
        return False


def convert_all(temp_dir: str):
    summarizer.ensure_model()

    os.makedirs(temp_dir, exist_ok=True)

    # Collect all output, if it's only .jsonl.zst
    # precompiled_summaries = load_old_summaries(output_dir)
    SERVER_INPUT_PATH = "/literotica_inputdata/"
    # SERVER_INPUT_PATH = "/dedup_inputdata/"
    SERVER_OUTPUT_PATH = "/llm_output_dataset/"

    files_to_do = WEBDAV_CLIENT.list(SERVER_INPUT_PATH)
    random.shuffle(files_to_do)
    pbar = tqdm(files_to_do)

    for file in pbar:
        if not file.endswith(".txt.zst.enc"):
            continue

        file_id = file.split("___")[0]
        if webdav_exists_file(SERVER_OUTPUT_PATH + file_id + ".json.zst.enc"):
            continue

        title = file.split("___")[1]
        pbar.set_description("File: " + file_id)

        pbar.set_description(f"Downloading: {file_id}")
        WEBDAV_CLIENT.download_file(
            SERVER_INPUT_PATH + file,
            temp_dir + file
        )
        pbar.set_description(f"Processing: {file_id}")

        input_file = os.path.join(temp_dir, file)
        output_file = os.path.join(temp_dir, file_id + "___" + SUMMARIZER.lower() + ".json")

        generate_for(input_file, output_file, title, {}, file_id)
        pbar.set_description(f"Uploading: {file_id}")
        WEBDAV_CLIENT.upload_file(
            SERVER_OUTPUT_PATH + file_id + ".json.zst.enc",
            output_file + ".zst.enc"
        )
        os.remove(input_file)
        os.remove(output_file + ".zst.enc")

    if os.path.exists(ALIVE_FILE):
        os.remove(ALIVE_FILE)


if __name__ == '__main__':
    convert_all("../../temp/")
