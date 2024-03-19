import os
import zstandard
from Cryptodome.Cipher import AES
from Cryptodome.Util.Padding import pad, unpad

KEY = "4hBJCbUgKRbEiUrW6NDa4yNbhVarXrEt".encode("utf-8")  # A 256 bit (32 byte) key


def compress_and_encrypt(data: bytes) -> bytes:
    z = zstandard.ZstdCompressor()
    compressed_data = z.compress(data)
    cipher = AES.new(KEY, AES.MODE_ECB)
    return cipher.encrypt(pad(compressed_data, AES.block_size))


def decompress_and_decrypt(encrypted_data: bytes) -> bytes:
    cipher = AES.new(KEY, AES.MODE_ECB)
    decrypted_data = unpad(cipher.decrypt(encrypted_data), AES.block_size)

    z = zstandard.ZstdDecompressor()
    return z.decompress(decrypted_data)


def load_file(file: str) -> bytes:
    with open(file, 'rb') as f:
        data = f.read()
        return decompress_and_decrypt(data)


def load_file_txt(file: str) -> str:
    with open(file, 'rb') as f:
        data = f.read()
        return decompress_and_decrypt(data).decode("utf-8")


def save_file(data: bytes, outfile: str, overwrite_existing_file=False) -> bool:
    outfile = ensure_zstd_enc_ext(outfile)
    if os.path.exists(outfile) and not overwrite_existing_file:
        return False

    with open(outfile, 'wb') as f:
        encrypted_data = compress_and_encrypt(data)
        f.write(encrypted_data)
        return True


def save_file_txt(data: str, outfile: str, overwrite_existing_file=False) -> bool:
    outfile = ensure_zstd_enc_ext(outfile)
    if os.path.exists(outfile) and not overwrite_existing_file:
        return False

    with open(outfile, 'wb') as f:
        encrypted_data = compress_and_encrypt(data.encode("utf-8"))
        f.write(encrypted_data)
        return True


def ensure_zstd_enc_ext(file_path):
    base_name = os.path.basename(file_path)

    # check if the filename ends with .zstd or .enc
    if base_name.endswith('.zst') or base_name.endswith('.enc'):
        name, extension = os.path.splitext(base_name)
        new_filename = name + ".zst.enc"
        return os.path.join(os.path.dirname(file_path), new_filename)
    elif base_name.endswith('.zst.enc'):
        pass
    else:
        new_filename = base_name + ".zst.enc"
        return os.path.join(os.path.dirname(file_path), new_filename)
    return file_path


if __name__ == "__main__":
    print(load_file_txt("../train_data/294b6512b7bdb24412eba75bdf6398c1.json.zst.enc"))
