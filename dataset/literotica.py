import hashlib
import json
import os
import re
from typing import Iterable
from tqdm.auto import tqdm
import mycrypt


# text, title, author, views, favorites
def stream_from_file(file: str) -> Iterable[{str: any}]:
    with open(file, "r") as f:
        for line in f:
            jobj = json.loads(line)

            yield {
                'text': jobj['text'],
                'title': jobj['meta']['title'],
                'author': jobj['meta']['author'],
                'views': jobj['meta']['n_views'],
                'favorites': jobj['meta']['n_favorites'],
            }


def clean_string(txt: str) -> str:
    cleaned = re.sub(r'\W+', '_', txt)
    ret = cleaned.strip('_')
    if len(ret) > 50:
        return ret[0:50]
    return ret


def generate_sha256_hash(text):
    hash_object = hashlib.sha256(text.encode())
    hex_dig = hash_object.hexdigest()
    return hex_dig


def generate_files_from_highest_views():
    views = []
    OUTPUT_DIR = "../__data/literotica/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ONLYVIEWS_MIN = 695_782
    written = 0
    highest_length = 0
    for obj in tqdm(stream_from_file("/Users/augunrik/temp/erotica/Literotica.jsonl")):
        if obj['views'] < ONLYVIEWS_MIN:
            continue

        highest_length = max(highest_length, len(obj['text']))

        if len(obj['text']) < 30_000:
            continue

        outfile = OUTPUT_DIR + "/lit-" + generate_sha256_hash(obj['text']) + "___" + clean_string(
            obj['title']) + "___" + clean_string(obj['author']) + ".txt"

        mycrypt.save_file_txt(obj['text'], outfile)

        written += 1

    print("Written", written)
    print("Highest length", highest_length)

    # views.sort()
    # views.reverse()
    # print(views[1000])


if __name__ == "__main__":
    generate_files_from_highest_views()
