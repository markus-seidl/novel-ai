import os
from tqdm import tqdm, trange
from dataclasses import dataclass
from typing import List
import json
import pickle


@dataclass
class TorrentEntry:
    idx: int
    filename: str
    torrent_file: str
    md5: str


@dataclass
class Row:
    md5: str
    title: str
    author: str
    extension: str
    torrent_file: TorrentEntry = None


def load_torrents(torrent_dir: str) -> ({str: [TorrentEntry]}, {str: TorrentEntry}):
    torrents = {}
    md5_dict = {}

    for filename in tqdm(os.listdir(torrent_dir)):
        file_path = os.path.join(torrent_dir, filename)
        t = torf.Torrent.read(file_path)
        t_fs = t.files
        for fidx in range(len(t_fs)):
            t_f = str(os.path.basename(t_fs[fidx])).lower()
            md5 = os.path.splitext(t_f)[0]

            e = TorrentEntry(
                idx=fidx,
                filename=t_f,
                torrent_file=file_path,
                md5=md5
            )
            torrents.setdefault(filename, [])
            torrents[filename].append(e)

            md5_dict[md5] = e

    return torrents, md5_dict


def load_json(filename: str) -> [Row]:
    print(f"Loading {filename}...")
    with open(filename, 'r') as f:
        data = json.load(f)

    ret = []
    for e in tqdm(data):
        ret.append(
            Row(
                md5=str(e['MD5']).lower(),
                title=str(e['Title']),
                author=str(e['Author']),
                extension=str(e['Extension']).lower()
            )
        )

    return ret


def create_download_cmds(
        torrent_files: {str: [TorrentEntry]},
        torrent_md5s: {str: TorrentEntry},
        rows: [Row]
) -> {str: [int]}:
    dl_dict: {str: [int]} = {}

    for row in rows:
        if row.md5 not in torrent_md5s:
            print(f"No torrent for {row.md5}")
            continue

        tf = torrent_md5s[row.md5]
        dl_dict.setdefault(tf.torrent_file, [])
        dl_dict[tf.torrent_file].append(tf.idx + 1)

    return dl_dict


def action_generate_dl_script():
    if False:
        torrent_files, torrent_md5s = load_torrents("./torrents/")
        with open("torrent_index.pickle", "wb+") as p:
            pickle.dump((torrent_files, torrent_md5s), p)
    else:
        with open("torrent_index.pickle", "rb") as p:
            torrent_files, torrent_md5s = pickle.load(p)

    print(len(torrent_files))  # 2787

    print("Loading table...")
    rows = load_json("libgenrs_fiction.json")
    print(len(rows))

    print("Creating download cmds...")
    t_idx_map = create_download_cmds(torrent_files, torrent_md5s, rows)

    for tf, idxs in t_idx_map.items():
        print(
            "aria2c", "--seed-time=0", "--select-file=" + ",".join(map(str, idxs)), "--dir=./dl/", tf
        )


def get_all_files_in_directory(directory):
    files_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            files_list.append((
                os.path.splitext(file)[0], os.path.join(root, file)
            ))
    return files_list


if __name__ == '__main__':
    # action_generate_dl_script()
    rows = load_json("libgenrs_fiction.json")
    rows_dict: {str: Row} = {}
    for row in rows:
        rows_dict[row.md5] = row

    dl_md5s = get_all_files_in_directory("./dl/")

    for md5 in dl_md5s:
        if md5[0] not in rows_dict:
            continue

        row = rows_dict[md5[0]]
        print(row.title, row.author, md5)
