import dataclasses
import json
import os

import zstandard

from entities import Book, TrainingData
from book_parser import load_book_from_text
import sumi.textsumi as summarizer
# import sumi.nop_summarizer as summarizer
from tqdm.auto import tqdm, trange
import mycrypt

PREVIOUS_SENTENCES = 5
MIN_PREV_LENGTH = 100

NEXT_SENTENCES = 3
MIN_NEXT_LENGTH = 100

WINDOW_STEP_SIZE = 3

SUMMARY_LENGTH = 3000


# precompiled_summaries : ("md5", "chapter title") : "summary"
def convert_to_trainingdata(book: Book, title: str, precompiled_summaries: {(str, str): str}, md5: str) -> [TrainingData]:
    ret = []
    chapter_bar = tqdm(book.chapters, leave=False)
    for chapter in chapter_bar:
        chapter_bar.set_description("Chapter " + str(chapter.title) + " generating summary.")

        # create chapter summary
        hash = (md5, chapter.title)
        if hash in precompiled_summaries:
            chapter_summary = precompiled_summaries[hash]
        else:
            chapter_text = " ".join(chapter.sentences)
            chapter_summary = summarizer.summarize_text(chapter_text, SUMMARY_LENGTH)

            chapter_bar.set_description("Chapter " + str(chapter.title) + " generating sentences.")

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


def convert_all(input_dir: str, output_dir: str):
    summarizer.ensure_model()

    os.makedirs(output_dir, exist_ok=True)

    # Collect all output, if it's only .jsonl.zst
    precompiled_summaries = load_old_summaries(output_dir)

    known_summary_files: {str: bool} = {}
    for t in precompiled_summaries.keys():
        known_summary_files[t[0]] = True

    pbar = tqdm(os.listdir(input_dir))
    for file in pbar:
        if not file.endswith(".txt.zst.enc"):
            continue

        md5 = file.split("___")[0]
        if md5 not in known_summary_files:
            continue

        pbar.set_description("File: " + md5)
        input_file = os.path.join(input_dir, file)

        title = file.split("___")[1]
        author = file.split("___")[2]
        output_file = os.path.join(output_dir, md5 + ".json")

        if not os.path.exists(output_file):
            generate_for(input_file, output_file, title, precompiled_summaries, md5)


def load_old_summaries(output_dir) -> {(str, str) : str}:
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
    convert_all("../dedup_inputdata/", "../train_data/")
