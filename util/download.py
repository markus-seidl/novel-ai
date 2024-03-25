import os
from webdav3.client import Client
from tqdm.auto import tqdm
from webdav3.exceptions import RemoteResourceNotFound


def file_callback(current, total, fbar: tqdm):
    fbar.update(current - fbar.n)
    fbar.total = total


def download_files(remote_path: str, local_path: str):
    options = {
        'webdav_hostname': os.environ.get('WEBDAV_HOSTNAME'),
        'webdav_login': os.environ.get('WEBDAV_USER'),
        'webdav_password': os.environ.get('WEBDAV_PASS'),
        'disable_check': True
    }

    client = Client(options)

    with tqdm(unit_scale=True, unit="bytes") as fbar:
        fbar.set_description(f"Downloading file (?)")
        client.download_sync(
            remote_path,
            local_path,
            progress=file_callback,
            progress_args=(fbar,)
        )


if __name__ == '__main__':
    download_files("/models/", "./models/")
