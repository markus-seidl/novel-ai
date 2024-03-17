import os
import json
from tqdm.auto import tqdm
import mycrypt


def load_dataset(directory):
    all_data = []

    pbar = tqdm(os.listdir(directory))
    for filename in pbar:
        pbar.set_description(filename)
        if filename.endswith(".json.zst.enc"):
            in_file = os.path.join(directory, filename)
            json_lines = mycrypt.load_file_txt(in_file).split("\n")

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

    mycrypt.save_file(json_data, outfile)


if __name__ == '__main__':
    dataset = load_dataset("../train_data/")
    print(len(dataset))

    print(dataset[0])

    write_and_compress(dataset, "compressed_data.json")
