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
                    all_data.append({
                        "instruction": data['summary'],
                        "input": data['previous_sentences'],
                        "output": data['expected_answer'],
                    })

    return all_data


def write_and_compress(training_datas: [], outfile: str):
    json_data = "\n".join(json.dumps(t) for t in training_datas).encode("utf-8")

    cctx = zstandard.ZstdCompressor()
    if True:
        compressed_data = cctx.compress(json_data)
    else:
        compressed_data = json_data

    with open(outfile, 'wb') as file:
        file.write(compressed_data)


if __name__ == '__main__':
    dataset = load_dataset("../train_data/")
    print(len(dataset))

    print(dataset[0])

    write_and_compress(dataset, "compressed_data.jsonl.zst")
