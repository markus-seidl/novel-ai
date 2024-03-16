import os
import re
import subprocess
from libgen_rs_torrent_dl import load_json, Row, get_all_files_in_directory

CONVERT_CMD = "/opt/homebrew/bin/ebook-convert"


def convert_book_to_txt(infile, outfile):
    if not os.path.exists(outfile):
        try:
            result = subprocess.run([CONVERT_CMD, infile, outfile], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if result.returncode != 0:
                print(f"Error while decoding {infile}: {result.stderr.decode()}")
        except subprocess.CalledProcessError as e:
            print(f"Error while converting {infile}: {e}")


def clean_string(txt: str) -> str:
    clean_title = re.sub(r'\W+', '_', txt)
    return clean_title.strip('_')


def convert_all(index: [Row], book_files: [(str, str)]):
    rows_dict: {str: Row} = {}
    for row in rows:
        rows_dict[row.md5] = row

    for md5 in book_files:
        if md5[0] not in rows_dict:
            continue

        row = rows_dict[md5[0]]

        clean_title = clean_string(row.title)
        clean_author = clean_string(row.author)

        outfile = f"./converted/{md5[0]}___{clean_title}___{clean_author}.txt"
        infile = md5[1]

        print("###########################")
        convert_book_to_txt(infile, outfile)


if __name__ == "__main__":
    book_files = get_all_files_in_directory("./dl/")
    rows = load_json("libgenrs_fiction.json")
    convert_all(rows, book_files)
