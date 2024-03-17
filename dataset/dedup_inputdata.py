import os
import shutil
from tqdm.auto import tqdm

if __name__ == "__main__":
    INFOLDER = "../enc_inputdata/"
    OUTFOLDER = "../dedup_inputdata/"

    # collect all filenames
    hashes: {(str, str): str} = {}
    pbar = tqdm(os.listdir(INFOLDER))
    for filename in pbar:
        pbar.set_description(filename)
        in_filepath = os.path.join(INFOLDER, filename)
        out_filepath = os.path.join(OUTFOLDER, filename)
        if ".txt.zst.enc" not in in_filepath:
            continue

        parts = filename.split("___")
        hash = (parts[1], parts[2])
        if hash not in hashes:
            shutil.copy(in_filepath, out_filepath)
            hashes[hash] = parts[0]
        else:
            print(f"Skipping duplicate file: {filename}, because {hashes[hash]} is already there.")

