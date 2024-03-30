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
import hashlib

sys.path.insert(1, '../dataset/')
import mycrypt

CONVERT_CMD = "/opt/homebrew/bin/ebook-convert"


def calculate_md5(filename):
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    ret = hash_md5.hexdigest()
    return ret


def convert_book_to_txt(infile, outfile):
    if not os.path.exists(outfile):
        try:
            result = subprocess.run([CONVERT_CMD, infile, outfile], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                print(f"Error while decoding {infile}: {result.stderr.decode()}")
        except subprocess.CalledProcessError as e:
            print(f"Error while converting {infile}: {e}")


def clean_string(txt: str) -> str:
    cleaned = re.sub(r'\W+', '_', txt)
    ret = cleaned.strip('_')
    if len(ret) > 50:
        return ret[0:50]
    return ret


MISSED_ARCHIVES = 0


def convert_all(book_files: [(str, str)], enc_output_directory: str, temp_dir: str):
    pbar = tqdm(book_files)
    for file_pair in pbar:
        file_name = file_pair[0]
        in_file_path = file_pair[1]

        md5 = calculate_md5(in_file_path)
        filename_parts = str(file_name).split("-")
        raw_title = filename_parts[0]
        raw_author = filename_parts[1] if len(filename_parts) > 1 else "UNKNOWN"

        pbar.set_description(f"{md5}")
        clean_title = clean_string(raw_title)
        clean_author = clean_string(raw_author)

        outfile = f"{temp_dir}/{md5}___{clean_title}___{clean_author}.txt"

        convert_book_to_txt(in_file_path, outfile)

        if os.path.exists(outfile):
            with open(outfile, "r") as f:
                data = f.read()
            outfile = f"{enc_output_directory}/{md5}___{clean_title}___{clean_author}.txt"
            mycrypt.save_file_txt(data, outfile)


if __name__ == "__main__":
    book_files = get_all_files_in_directory("/Volumes/Dia/ai-data/RAW/anna-manual/In/")
    enc_output_directory = "../train_data/enc_inputdata/"
    temp_dir = "../train_data/conv_temp/"
    convert_all(book_files, enc_output_directory, temp_dir)

    # num_chunks = multiprocessing.cpu_count()
    # book_files_chunks = list(to_chunks(book_files, num_chunks))

    # partial_func = partial(worker, rows, enc_output_directory)

    # with multiprocessing.Pool() as pool:
    #     # Process each chunk concurrently
    #     pool.map(partial_func, book_files_chunks)
    # partial_func(book_files)

    print(MISSED_ARCHIVES)
