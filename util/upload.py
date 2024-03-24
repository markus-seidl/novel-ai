import os
from webdav3.client import Client
from tqdm.auto import tqdm


def upload_files(local_directory, remote_directory):
    options = {
        'webdav_hostname': os.environ.get('WEBDAV_HOSTNAME'),
        'webdav_login': os.environ.get('WEBDAV_USER'),
        'webdav_password': os.environ.get('WEBDAV_PASS'),
        'disable_check': True
    }

    client = Client(options)

    client.mkdir(remote_directory)

    for filename in tqdm(os.listdir(local_directory)):
        local_path = os.path.join(local_directory, filename)
        remote_path = os.path.join(remote_directory, filename)

        client.upload_sync(remote_path=remote_path, local_path=local_path)
