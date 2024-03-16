import dataclasses
import json
import os

from entities import Book, TrainingData
from book_parser import load_book
import sumi.textsumi as summarizer
# import sumi.nop_summarizer as summarizer
import tqdm
import zstandard as zstd

PREVIOUS_SENTENCES = 5
NEXT_SENTENCES = 3

SUMMARY_LENGTH = 300
SUMMARY_PREV = 20
SUMMARY_NEXT = 20


def convert_to_trainingdata(book: Book, title: str) -> [TrainingData]:
    ret = []
    chapter_bar = tqdm.tqdm(book.chapters)
    for chapter in chapter_bar:
        chapter_bar.set_description(chapter.title)

        for i in tqdm.trange(PREVIOUS_SENTENCES, len(chapter.sentences) - NEXT_SENTENCES):
            previous_sentences = chapter.sentences[i - PREVIOUS_SENTENCES:i]
            summary_sentences = chapter.sentences[
                                max(0, i - SUMMARY_PREV):min(i + SUMMARY_NEXT, len(chapter.sentences))]
            summary_text = " ".join(summary_sentences)

            summary = summarizer.summarize_text(summary_text, SUMMARY_LENGTH)

            temp = TrainingData(
                book_title=title,
                chapter_title=chapter.title,
                summary=summary,
                previous_sentences=" ".join(previous_sentences),
                expected_answer=" ".join(chapter.sentences[i:i + NEXT_SENTENCES])
            )

            ret.append(temp)
    return ret


def write_and_compress(training_datas: [TrainingData], outfile: str):
    json_data = "\n".join(json.dumps(dataclasses.asdict(dc)) for dc in training_datas).encode("utf-8")

    cctx = zstd.ZstdCompressor()
    if True:
        compressed_data = cctx.compress(json_data)
    else:
        compressed_data = json_data

    with open(outfile, 'wb') as file:
        file.write(compressed_data)


def generate_for(input_file: str, output_file: str, title: str):
    summarizer.ensure_model()

    book = load_book(input_file)
    tds = convert_to_trainingdata(book, title)
    write_and_compress(tds, output_file)


def convert_all(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith(".txt"):
            md5 = file.split("___")[0]
            input_file = os.path.join(input_dir, file)

            title = file.split("___")[1]
            author = file.split("___")[2]
            output_file = os.path.join(output_dir, md5 + ".jsonl.zst")

            if md5 == "fabb39e33d3519a7d442d6777baba500":
                generate_for(input_file, output_file, title)


if __name__ == '__main__':
    convert_all("../anna/converted/", "../train_data/")
