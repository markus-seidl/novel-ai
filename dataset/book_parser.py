import re
from nltk import sent_tokenize
from entities import *
from sumi.nop_summarizer import summarize_text

CHAPTER_REGEX = re.compile(r'[a-zA-Z0-9 ,’\'\\*]+')
SUMMARY_LENGTH = 0


def find_corpus(text: str) -> (int):
    """
    Find the text corpus. Only lines with a dot are considered to be corpus.
    :param text:
    :return:
    """
    lines_needed_to_start_corpus = 5
    lines = str(text).split("\n")
    start_of_book = -1

    last_line_was_sentence = 0
    lines_visited = 0
    for i in range(len(lines)):
        line = lines[i].strip()

        if len(line) == 0:
            lines_visited += 1
            continue

        if "." in line or "\"" in line or "“" in line:
            last_line_was_sentence += 1
            lines_visited += 1
        else:
            lines_visited = 0
            last_line_was_sentence = 0

        if last_line_was_sentence >= lines_needed_to_start_corpus:
            start_of_book = i - lines_visited
            break

    return start_of_book


def load_txt_file(filename):
    with open(filename, 'r', encoding="utf-8") as file:
        text = file.read()
    return text


def simple_cleaning(text: str, start_corpus_line_no: int) -> str:
    lines_in = text.split("\n")
    ret = list()
    for i in range(start_corpus_line_no, len(lines_in)):
        line = lines_in[i]
        line = line.strip()
        if len(line) == 0:
            continue
        ret.append(line)
    return "\n".join(ret)


def is_chapter(line: str) -> bool:
    return (str(line).count(" ") < 10
            and CHAPTER_REGEX.fullmatch(line)
            and line.lower() != "the end"
            and not line.startswith("By"))


def convert_to_chapter(lines: List[str]) -> Chapter:
    title = lines[0]
    text = " ".join(lines[1:])
    sentences = sent_tokenize(text)
    return Chapter(title=title, sentences=sentences, summary=summarize_text(text, SUMMARY_LENGTH))


def split_into_chapters(text: str) -> Book:
    lines_in = text.split("\n")

    chapters = []
    current_chapter = []
    for line in lines_in:
        if is_chapter(line):
            if current_chapter:
                temp = convert_to_chapter(current_chapter)
                if len(temp.sentences) > 3:
                    chapters.append(temp)
            current_chapter = [line]
        else:
            current_chapter.append(line)

    temp = convert_to_chapter(current_chapter)
    if len(temp.sentences) > 3:
        chapters.append(temp)

    return Book(lines_in[0], chapters)


def load_book(filename: str) -> Book:
    file_text = load_txt_file(filename)
    start_corpus_line_no = find_corpus(file_text)

    file_text = simple_cleaning(file_text, start_corpus_line_no)
    book = split_into_chapters(file_text)

    for chapter in book.chapters:
        fl = chapter.sentences.copy()
        len_before = len(fl)
        fl = list(filter(lambda line: not re.match(r'^[\* \d]+$', line.strip()), fl))

        chapter.sentences = fl

    return book


def load_book_from_text(file_text: str) -> Book:
    start_corpus_line_no = find_corpus(file_text)

    file_text = simple_cleaning(file_text, start_corpus_line_no)
    book = split_into_chapters(file_text)

    for chapter in book.chapters:
        fl = chapter.sentences.copy()
        len_before = len(fl)
        fl = list(filter(lambda line: not re.match(r'^[\* \d]+$', line.strip()), fl))

        chapter.sentences = fl

    return book


if __name__ == '__main__':
    # /Volumes/Dia/ai-data/anna-manual/
    # book_file = "aae8a9a0b14d2b900704cfc1e2ac3eb9.txt"
    # book_file = "b0845a13375a4fb410e753ec526a8e3f.txt"
    # book_file = "055cc96d3c8a23505a6e6b353b773cd2.txt"
    book_file = "a2a8b19cdddea509540191833a1364fc.txt"
    book = load_book(book_file)

    for chap in book.chapters:
        print(chap.title, " ".join(chap.sentences))
    # sentences =
    # print(sentences[0:100])
    # print(start_corpus_line_no)
