import dataclasses
import datetime
import json
import os
import random
import re
import socket

import zstandard

from entities import Book, SummaryBook, SummaryChunk, SummaryChapter
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
PROCESSED_BOOKS = 0

NEEDED_WORDS = os.environ.get("NEEDED_WORDS")
if NEEDED_WORDS != "":
    temp = {}
    for word in NEEDED_WORDS.split(","):
        temp[str(word).lower()] = True
    NEEDED_WORDS = temp


def count_words(dictionary: {str: bool}, text: str) -> ({str: int}, int):
    counts = {word: 0 for word in dictionary}

    # Split the text into words
    words = re.findall(r'\b\w+\b', text.lower())
    sum = 0

    # Count occurrences for each word in dictionary
    for word in words:
        if word in counts:
            counts[word] += 1
            sum += 1

    return counts, sum


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


ALIVE_FILE = f"{CONTAINER_LABEL}_{os.getpid()}_im_alive.txt"


def inform_alive(current_md5):
    with open(ALIVE_FILE, "w+") as f:
        f.write(json.dumps({
            "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summarizer_performance": summarizer.performance_info(),
            "file_id": current_md5,
            "summarizer": SUMMARIZER,
            "container_label": CONTAINER_LABEL,
            "processed_books": PROCESSED_BOOKS,
        }, indent=2))
    WEBDAV_CLIENT.upload_file(ALIVE_FILE, ALIVE_FILE)


def convert_to_summary_book(
        book: Book, title: str, file_id: str
) -> [SummaryBook]:
    summary_book = SummaryBook(title=title, file_id=file_id, chapters=[])
    chapter_bar = tqdm(book.chapters, leave=False)
    for chapter in chapter_bar:
        chapter_bar.set_description("Chapter " + str(chapter.title) + f" {summarizer.performance_info()}")

        chunks = summarizer.split_into_chunks(" ".join(chapter.sentences))

        inform_alive(file_id)

        summary_chapter = SummaryChapter(chapter.title, [])
        summary_book.chapters.append(summary_chapter)
        for chunk in tqdm(chunks):
            chunk_summary = summarizer.summarize_text(chunk)

            chunk_sentences = sent_tokenize(chunk)

            summary_chapter.chunks.append(SummaryChunk(chunk_summary, chunk_sentences))

    return summary_book


def write_and_compress(summary_book: [SummaryBook], outfile: str):
    json_data = json.dumps(dataclasses.asdict(summary_book), indent=True)
    mycrypt.save_file_txt(json_data, outfile)


def generate_for(input_file: str, output_file: str, title: str, precompiled_summaries: {(str, str): str}, md5) -> bool:
    input_file_text = mycrypt.load_file_txt(input_file)
    hist, count = count_words(NEEDED_WORDS, input_file_text)
    print("Histogram", hist, "count", count)

    if not count > 10:
        return False

    book = load_book_from_text(input_file_text)
    summary_book = convert_to_summary_book(book, title, md5)
    write_and_compress(summary_book, output_file)
    return True


def webdav_exists_file(file_path: str) -> bool:
    try:
        info = WEBDAV_CLIENT.info(file_path)
        return info['size'] != 0
    except RemoteResourceNotFound:
        return False


def convert_all(temp_dir: str):
    global PROCESSED_BOOKS
    summarizer.ensure_model()

    os.makedirs(temp_dir, exist_ok=True)

    # SERVER_INPUT_PATH = "/literotica_inputdata/"
    SERVER_INPUT_PATH = "/dedup_inputdata/"
    SERVER_OUTPUT_PATH = "/output_llm_dataset/"

    files_to_do = WEBDAV_CLIENT.list(SERVER_INPUT_PATH)
    random.shuffle(files_to_do)
    pbar = tqdm(files_to_do)

    input_file = ""
    output_file = ""
    try:
        for file in pbar:
            if not file.endswith(".txt.zst.enc"):
                continue

            file_id = file.split("___")[0]
            title = file.split("___")[1]

            if webdav_exists_file(SERVER_OUTPUT_PATH + file_id + ".json.zst.enc"):
                continue

            pbar.set_description(f"Downloading: {file_id}")
            WEBDAV_CLIENT.download_file(
                SERVER_INPUT_PATH + file,
                temp_dir + file
            )
            pbar.set_description(f"Processing: {file_id}")

            input_file = os.path.join(temp_dir, file)
            output_file = os.path.join(temp_dir, file_id + "___" + SUMMARIZER.lower() + ".json")

            generated = generate_for(input_file, output_file, title, {}, file_id)

            if generated:
                pbar.set_description(f"Uploading: {file_id}")
                WEBDAV_CLIENT.upload_file(
                    SERVER_OUTPUT_PATH + file_id + ".json.zst.enc",
                    output_file + ".zst.enc"
                )

                PROCESSED_BOOKS += 1
    finally:
        if os.path.exists(input_file):
            os.remove(input_file)
        output_file += ".zst.enc"

        if os.path.exists(output_file):
            os.remove(output_file)

        if os.path.exists(ALIVE_FILE):
            os.remove(ALIVE_FILE)


if __name__ == '__main__':
    convert_all("../temp/")
