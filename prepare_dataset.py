import os
import io
import requests
import zipfile
import shutil
from content_based_recomendation.scripts.movie_lens_content_based_recomendation import filter_ratings
from utils.features_extraction.movie_lens_features_extractor import FeaturesExtractor
from settings import PATH_TO_DATA


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    # save_response_content(response, destination)
    return response


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def unpack_starts_with(zip_file, zip_skip, save_path):
    members = [x for x in zip_file.NameToInfo.keys() if x.startswith(zip_skip) and len(x) > len(zip_skip)]
    for mem in members:
        path = save_path + mem[len(zip_skip):]
        if not path.endswith('/'):
            read_file = zip_file.open(mem)
            with open(path, 'wb') as write_file:
                shutil.copyfileobj(read_file, write_file)
        else:
            os.makedirs(path, exist_ok=True)


def main():
    eas_path = './dataset/raw/the-movies-dataset/'
    eas_zip_skip = ''
    eas_gdrive_id = '1Qx9FAqaIG9PbMRJ6coT_NNA9Bck3-jSZ'

    os.makedirs(eas_path, exist_ok=True)
    print('Downloading...')
    r = download_file_from_google_drive(eas_gdrive_id, None)
    print('Unzip...')
    z = zipfile.ZipFile(io.BytesIO(r.content))
    unpack_starts_with(z, eas_zip_skip, eas_path)
    print('Filtering')
    dataset_path = f'{PATH_TO_DATA}/raw/the-movies-dataset'
    features_extractor = FeaturesExtractor(dataset_path)
    data = features_extractor.run()
    filter_ratings(dataset_path, data)
    print('Done')


if __name__ == '__main__':
    main()


