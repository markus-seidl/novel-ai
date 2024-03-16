import dataclasses
import json

from entities import Book, TrainingData
from book_parser import load_book
import sumi.textsumi as summarizer
import tqdm
import zstandard as zstd

PREVIOUS_SENTENCES = 5
NEXT_SENTENCES = 3

SUMMARY_LENGTH = 300
SUMMARY_PREV = 20
SUMMARY_NEXT = 20


def convert_to_trainingdata(book: Book) -> [TrainingData]:
    ret = []
    chapter_bar = tqdm.tqdm(book.chapters)
    for chapter in chapter_bar:
        chapter_bar.set_description(chapter.title)

        for i in tqdm.trange(PREVIOUS_SENTENCES, len(chapter.sentences) - NEXT_SENTENCES):
            previous_sentences = chapter.sentences[i - PREVIOUS_SENTENCES:i]
            summary_sentences = chapter.sentences[max(0, i - SUMMARY_PREV):min(i + SUMMARY_NEXT, len(chapter.sentences))]
            summary_text = " ".join(summary_sentences)

            summary = summarizer.summarize_text(summary_text, SUMMARY_LENGTH)

            temp = TrainingData(
                book_title=book.title,
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
    compressed_data = cctx.compress(json_data)

    with open(outfile, 'wb') as file:  # + '.zst'
        file.write(json_data)


def generate_for(input_file: str, output_file: str):
    summarizer.ensure_model()

    book = load_book(input_file)
    tds = convert_to_trainingdata(book)
    write_and_compress(tds, output_file)


if __name__ == '__main__':
    # /Volumes/Dia/ai-data/anna-manual/
    # book_file = "aae8a9a0b14d2b900704cfc1e2ac3eb9.txt"
    book_file = "b0845a13375a4fb410e753ec526a8e3f.txt"
    # book_file = "055cc96d3c8a23505a6e6b353b773cd2.txt"
    # book_file = "a2a8b19cdddea509540191833a1364fc.txt"
    book = load_book(book_file)
    tds = convert_to_trainingdata(book)

    for td in tds:
        print(td.previous_sentences, "... [", td.summary, "] -> ", td.expected_answer, "")

    write_and_compress(tds, "temp.zsl")
