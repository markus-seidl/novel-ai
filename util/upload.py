import os
from webdav3.client import Client
from tqdm.auto import tqdm
from webdav3.exceptions import RemoteResourceNotFound


def file_callback(current, total, fbar: tqdm):
    fbar.update(current - fbar.n)
    fbar.total = total


def upload_files(local_directory, remote_directory):
    options = {
        'webdav_hostname': os.environ.get('WEBDAV_HOSTNAME'),
        'webdav_login': os.environ.get('WEBDAV_USER'),
        'webdav_password': os.environ.get('WEBDAV_PASS'),
        'disable_check': True
    }

    client = Client(options)

    client.mkdir(remote_directory)

    pbar = tqdm(os.walk(local_directory))
    for dirpath, dirs, files in pbar:
        for filename in files:
            local_path = os.path.join(dirpath, filename)
            relative_path = os.path.relpath(local_path, local_directory)
            remote_path = os.path.join(remote_directory, relative_path)
            pbar.set_description(f"Upload {local_path} to {remote_path}")

            # Ensure the remote subdirectory exists
            remote_subdir = os.path.dirname(remote_path)
            client.mkdir(remote_subdir)

            try:
                remote_size = client.info(remote_path)['size']
            except RemoteResourceNotFound:
                remote_size = 0

            # get size of filename
            local_size = os.path.getsize(local_path)

            if remote_size == local_size:
                print(f"Skipping {local_path}")

            # Upload file
            with tqdm(unit_scale=True, unit="bytes") as fbar:
                fbar.set_description(f"Uploading {local_path}")

                client.upload_file(
                    remote_path=remote_path,
                    local_path=local_path,
                    progress=file_callback,
                    progress_args=(fbar,)
                )
