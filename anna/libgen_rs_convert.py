import multiprocessing
import os
import re
import subprocess
from libgen_rs_torrent_dl import load_json, Row, get_all_files_in_directory
from tqdm import tqdm
import sys
import random
from multiprocessing import Pool
from functools import partial

sys.path.insert(1, '../dataset/')
import mycrypt

CONVERT_CMD = "/opt/homebrew/bin/ebook-convert"

DENYLIST = {
    "e65e54659b850dadcf079e876017ca17": True
}


def convert_book_to_txt(infile, outfile):
    if not os.path.exists(outfile):
        try:
            result = subprocess.run([CONVERT_CMD, infile, outfile], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # if result.returncode != 0:
            #     print(f"Error while decoding {infile}: {result.stderr.decode()}")
        except subprocess.CalledProcessError as e:
            print(f"Error while converting {infile}: {e}")


def clean_string(txt: str) -> str:
    cleaned = re.sub(r'\W+', '_', txt)
    ret = cleaned.strip('_')
    if len(ret) > 50:
        return ret[0:50]
    return ret


def convert_all(rows: [Row], book_files: [(str, str)], enc_output_directory: str):
    rows_dict: {str: Row} = {}
    for row in rows:
        rows_dict[row.md5] = row

    random.shuffle(book_files)
    pbar = tqdm(book_files)
    for md5 in pbar:
        if md5[0] not in rows_dict:
            continue

        if md5 in DENYLIST:
            continue

        row = rows_dict[md5[0]]
        if row.extension in ["pdf", "cbr"]:  # we do not OCR
            continue

        if row.extension in ["zip"]:  # TODO ZIP files need extraction before conversion!
            continue

        pbar.set_description(f"{md5[0]}")
        clean_title = clean_string(row.title)
        clean_author = clean_string(row.author)

        outfile = f"./converted/{md5[0]}___{clean_title}___{clean_author}.txt"
        infile = md5[1]

        convert_book_to_txt(infile, outfile)

        if os.path.exists(outfile):
            with open(outfile, "r") as f:
                data = f.read()
            outfile = f"{enc_output_directory}/{md5[0]}___{clean_title}___{clean_author}.txt"
            mycrypt.save_file_txt(data, outfile)


def to_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def worker(rows, enc_output_directory, book_chunk: list):
    convert_all(rows, book_chunk, enc_output_directory)


if __name__ == "__main__":
    book_files = get_all_files_in_directory("./dl/")
    rows = load_json("libgenrs_fiction.json")
    enc_output_directory = "../enc_inputdata/"
    # convert_all(rows, book_files, enc_output_directory)

    num_chunks = multiprocessing.cpu_count()
    book_files_chunks = list(to_chunks(book_files, num_chunks))

    partial_func = partial(worker, rows, enc_output_directory)

    with multiprocessing.Pool() as pool:
        # Process each chunk concurrently
        pool.map(partial_func, book_files_chunks)
