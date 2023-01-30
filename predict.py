import os
import pathlib
import shutil
from multiprocessing import Pool

import coremltools as ct
import typer
from PIL import Image

# Load the model
model = ct.models.MLModel("MyImageClassifier 4.mlmodel")


def process(path: pathlib.Path):
    data = Image.open(str(path))
    return model.predict({"image": data})


def process_image(image: pathlib.Path):
    result = process(image)
    p = result["classLabelProbs"]["not"]
    print(f"process {image} -> {p:.0%}")
    level = int(p * 10) * 10

    os.makedirs(f"output/{level}", exist_ok=True)
    shutil.copy(image, f"output/{level}/{image.name}")


def main(folder: pathlib.Path):
    with Pool(4) as p:
        p.map(process_image, folder.glob("*.jpeg"))


if __name__ == "__main__":
    typer.run(main)
