import wget
import tarfile
import ssl
from pathlib import Path

file_dict = {
    'TCSA.h5': 'https://learner.csie.ntu.edu.tw/~boyochen/TCSA/TCSA.h5'
}

compressed_postfix = '.tar.gz'


def download_compressed_file(data_folder_path, file_name):
    file_url = file_dict[file_name]
    file_path = data_folder_path / f'{file_name}{compressed_postfix}'
    ssl._create_default_https_context = ssl._create_unverified_context
    wget.download(file_url, out=str(file_path))


def uncompress_file(data_folder_path, file_name):
    compressed_file_path = data_folder_path / f'{file_name}{compressed_postfix}'
    if not compressed_file_path.exists():
        download_compressed_file(data_folder_path, file_name)

    with tarfile.open(compressed_file_path) as tar:
        tar.extractall(path=data_folder_path)


def verify_data(data_folder_path):
    for file_name in file_dict:
        file_path = data_folder_path / file_name
        if not file_path.exists():
            print('data download failed!')
            return False
    return True


def download_data(data_folder):
    data_folder_path = Path(data_folder)
    if not data_folder_path.exists():
        data_folder_path.mkdir()

    for file_name in file_dict:
        file_path = data_folder_path / file_name
        if not file_path.exists():
            uncompress_file(data_folder_path, file_name)

    return verify_data(data_folder_path)
