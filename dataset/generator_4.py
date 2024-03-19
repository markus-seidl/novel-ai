import dataclasses
import datetime
import json
import os
import random
import socket

import zstandard

from entities import Book, TrainingData
from book_parser import load_book_from_text
import sumi.textsumi as summarizer
# import sumi.nop_summarizer as summarizer
from tqdm.auto import tqdm, trange
import mycrypt
from webdav3.client import Client, RemoteResourceNotFound

WEBDAV_CLIENT = Client({
    'webdav_hostname': os.environ.get('WEBDAV_HOSTNAME'),
    'webdav_login': os.environ.get('WEBDAV_USER'),
    'webdav_password': os.environ.get('WEBDAV_PASS'),
    'disable_check': True
})
print("Connected to webdav, root contents: ", WEBDAV_CLIENT.list("/"))

PREVIOUS_SENTENCES = 5
MIN_PREV_LENGTH = 100

NEXT_SENTENCES = 3
MIN_NEXT_LENGTH = 100

WINDOW_STEP_SIZE = 3

SUMMARY_LENGTH = 3000


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't even have to be reachable
        s.connect(('bearo.de', 1))
        ip = str(s.getsockname()[0]).replace(".", "_")
    except Exception:
        ip = 'localhost'
    finally:
        s.close()
    return ip


ALIVE_FILE = f"im_alive_{os.getpid()}_{get_local_ip()}.txt"


def inform_alive(current_md5):
    with open(ALIVE_FILE, "w+") as f:
        f.write(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " " +
            summarizer.performance_info() + " " +
            current_md5 +
            str(os.environ.get("VAST_CONTAINERLABEL")) + "\n"
        )
    WEBDAV_CLIENT.upload_file(ALIVE_FILE, ALIVE_FILE)


# precompiled_summaries : ("md5", "chapter title") : "summary"
def convert_to_trainingdata(
        book: Book, title: str, precompiled_summaries: {(str, str): str}, md5: str
) -> [TrainingData]:
    ret = []
    chapter_bar = tqdm(book.chapters, leave=False)
    for chapter in chapter_bar:
        chapter_bar.set_description("Chapter " + str(chapter.title))

        # create chapter summary
        hash = (md5, chapter.title)
        if hash in precompiled_summaries:
            chapter_summary = precompiled_summaries[hash]
        else:
            chapter_text = " ".join(chapter.sentences)
            chapter_summary = summarizer.summarize_text(chapter_text, SUMMARY_LENGTH)

        chapter_bar.set_description("Chapter " + str(chapter.title) + f" ({summarizer.performance_info()})")
        inform_alive(md5)

        sentence_bar = trange(
            PREVIOUS_SENTENCES, len(chapter.sentences) - NEXT_SENTENCES, WINDOW_STEP_SIZE, leave=False
        )
        for i in sentence_bar:
            sentence_bar.set_description("Sentence Index: " + str(i))

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
                summary=chapter_summary,
                previous_sentences=previous_sentences_text,
                expected_answer=next_sentences_text
            )
            ret.append(temp)

            # Without Input Text
            temp = TrainingData(
                book_title=title,
                chapter_title=chapter.title,
                summary=chapter_summary,
                previous_sentences="",
                expected_answer=next_sentences_text
            )
            ret.append(temp)

    return ret


def write_and_compress(training_datas: [TrainingData], outfile: str):
    json_data = "\n".join(json.dumps(dataclasses.asdict(dc)) for dc in training_datas)

    mycrypt.save_file_txt(json_data, outfile)


def generate_for(input_file: str, output_file: str, title: str, precompiled_summaries: {(str, str): str}, md5):
    input_file_text = mycrypt.load_file_txt(input_file)
    book = load_book_from_text(input_file_text)
    tds = convert_to_trainingdata(book, title, precompiled_summaries, md5)
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
    SERVER_INPUT_PATH = "/dedup_inputdata/"
    SERVER_OUTPUT_PATH = "train_data/"

    files_to_do = WEBDAV_CLIENT.list(SERVER_INPUT_PATH)
    random.shuffle(files_to_do)
    pbar = tqdm(files_to_do)

    for file in pbar:
        if not file.endswith(".txt.zst.enc"):
            continue

        md5 = file.split("___")[0]
        if webdav_exists_file(SERVER_OUTPUT_PATH + md5 + ".json.zst.enc"):
            print(f"Server file for {md5} already exists")
            continue

        title = file.split("___")[1]
        author = file.split("___")[2]
        pbar.set_description("File: " + md5)

        pbar.set_description(f"Downloading: {md5}")
        WEBDAV_CLIENT.download_file(
            SERVER_INPUT_PATH + file,
            temp_dir + file
        )
        pbar.set_description(f"Processing: {md5}")

        input_file = os.path.join(temp_dir, file)
        output_file = os.path.join(temp_dir, md5 + ".json")

        generate_for(input_file, output_file, title, {}, md5)
        pbar.set_description(f"Uploading: {md5}")
        WEBDAV_CLIENT.upload_file(
            SERVER_OUTPUT_PATH + md5 + ".json.zst.enc",
            output_file + ".zst.enc"
        )
        os.remove(input_file)
        os.remove(output_file + ".zst.enc")

    if os.path.exists(ALIVE_FILE):
        os.remove(ALIVE_FILE)


def load_old_summaries(output_dir) -> {(str, str): str}:
    precompiled_summaries: {(str, str): str} = {}
    pbar = tqdm(os.listdir(output_dir))
    for file in pbar:
        if not file.endswith(".jsonl.zst"):
            continue

        pbar.set_description(f"Reading existing summaries: {file}")

        with open(os.path.join(output_dir, file), 'rb') as fh:
            dctx = zstandard.ZstdDecompressor()
            reader = dctx.stream_reader(fh)
            decompressed_string = reader.read().decode('utf-8')
            json_lines = decompressed_string.strip().split("\n")

            basename = os.path.basename(file).replace(".jsonl.zst", "")

            for json_line in json_lines:
                data = json.loads(json_line)
                precompiled_summaries[(basename, data["chapter_title"])] = data['summary']

    return precompiled_summaries


if __name__ == '__main__':
    convert_all("../temp/")
