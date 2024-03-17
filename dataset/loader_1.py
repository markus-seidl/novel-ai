import os
import zstandard
import json
from tqdm.auto import tqdm


def load_dataset(directory):
    all_data = []

    pbar = tqdm(os.listdir(directory))
    for filename in pbar:
        pbar.set_description(filename)
        if filename.endswith(".jsonl.zst"):
            with open(os.path.join(directory, filename), 'rb') as fh:
                dctx = zstandard.ZstdDecompressor()
                reader = dctx.stream_reader(fh)
                decompressed_string = reader.read().decode('utf-8')
                json_lines = decompressed_string.strip().split("\n")

                for json_line in json_lines:
                    data = json.loads(json_line)
                    all_data.append(data)

    return all_data


if __name__ == '__main__':
    dataset = load_dataset("../train_data/")
    print(len(dataset))

    print(dataset[0])
