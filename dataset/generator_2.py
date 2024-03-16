import dataclasses
import json
import os

from entities import Book, TrainingData
from book_parser import load_book
import sumi.textsumi as summarizer
# import sumi.nop_summarizer as summarizer
from tqdm.auto import tqdm, trange
import zstandard as zstd

PREVIOUS_SENTENCES = 5
NEXT_SENTENCES = 3

SUMMARY_LENGTH = 3000


def convert_to_trainingdata(book: Book, title: str) -> [TrainingData]:
    ret = []
    chapter_bar = tqdm(book.chapters)
    for chapter in chapter_bar:
        chapter_bar.set_description("Chapter " + str(chapter))

        # create chapter summary
        chapter_text = " ".join(chapter.sentences)
        chapter_summary = summarizer.summarize_text(chapter_text, SUMMARY_LENGTH)

        sentence_bar = trange(PREVIOUS_SENTENCES, len(chapter.sentences) - NEXT_SENTENCES)
        for i in sentence_bar:
            sentence_bar.set_description("Sentence Index: " + str(i))
            previous_sentences = chapter.sentences[i - PREVIOUS_SENTENCES:i]
            temp = TrainingData(
                book_title=title,
                chapter_title=chapter.title,
                summary=chapter_summary,
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

    pbar = tqdm(os.listdir(input_dir))
    for file in pbar:
        if not file.endswith(".txt"):
            continue

        md5 = file.split("___")[0]
        pbar.set_description("File: " + md5)
        input_file = os.path.join(input_dir, file)

        title = file.split("___")[1]
        author = file.split("___")[2]
        output_file = os.path.join(output_dir, md5 + ".jsonl.zst")

        if not os.path.exists(output_file):
            generate_for(input_file, output_file, title)


if __name__ == '__main__':
    convert_all("../anna/converted/", "../train_data/")
