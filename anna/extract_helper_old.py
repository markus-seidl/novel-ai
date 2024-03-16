import os
import subprocess

import py7zr
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

PKZIP = "/opt/homebrew/bin/7zz"


def extract_cli(archive_file, file_to_extract, output_dir):
    # 7z executable command
    command = [PKZIP, "x", f"-o{output_dir}", archive_file, file_to_extract]

    # Run the command
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Check if the command was successful
    if result.returncode != 0:
        print(f"An error occurred while compressing the directory. Error message: {result.stderr.decode()}")


def extract_and_search_thread(file_info):
    file_name = file_info['filename']

    # Check file size and skip directories
    if file_info['maxsize'] == 0:
        return

    archive_file, out_dir, keywords = file_info['archive_file'], file_info['out_dir'], file_info['keywords']

    # seven_z_file.extract(path=out_dir, targets=[file_name])  # slow!
    if "/" in file_name:
        extract_cli(archive_file, file_name, out_dir)  # assuming extract_cli is defined elsewhere

    # we just extracted a directory, skip
    if os.path.isdir(os.path.join(out_dir, file_name)) or not os.path.exists(os.path.join(out_dir, file_name)):
        return

    # Open the extracted file and search for the keywords
    with open(os.path.join(out_dir, file_name), 'r') as f:
        file_content = f.read()

    found = sum(keyword in file_content for keyword in keywords)

    if found <= 2:
        os.remove(os.path.join(out_dir, file_name))


def extract_and_search(archive_file: str, keywords: [str], out_dir):
    # Create a temp directory if it does not already exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print("Find relevant files...")
    with py7zr.SevenZipFile(archive_file, mode='r') as seven_z_file:
        file_infos = [
            {
                'filename': f['filename'], 'maxsize': f['maxsize'], 'archive_file': archive_file, 'out_dir': out_dir,
                'keywords': keywords
            } for f in seven_z_file.files.files_list if f['maxsize'] != 0
        ]

        print("... start extracting and searching...")
        no_cpus = cpu_count() - 1
        with Pool(processes=no_cpus) as pool:
            for _ in tqdm(pool.imap_unordered(extract_and_search_thread, file_infos), total=len(file_infos)):
                pass


if __name__ == "__main__":
    keywords = [
        "cum ", "anal ", "creampie", "seed", "pussy", "femdom", "sissy", "bussy", "horny", "sex ", "fuck", "tits",
        "dick", "penis", "breeding"
    ]
    extract_and_search('/Volumes/Dia/ai-data/lg_pdf2txt.7z', keywords, "/Volumes/Dia/ai-data/the-eye-books/")
