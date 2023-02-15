import pathlib
from multiprocessing import Pool

import typer
from pyicloud import PyiCloudService


def process_media(api, media: pathlib.Path):
    api.photos.upload_file(str(media))


def main(username: str, folder: pathlib.Path, pattern: str = "*.jpeg"):
    api = PyiCloudService(username)

    with Pool(4) as p:
        p.map(process_media, ((api, k) for k in folder.glob(pattern)))

if __name__ == "__main__":
    typer.run(main)