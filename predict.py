import pathlib
import shutil

import coremltools as ct
import typer
from PIL import Image
import os

# Load the model
model = ct.models.MLModel('MyImageClassifier 3.mlmodel')

def process(path: pathlib.Path):
    data = Image.open(str(path))
    return model.predict({'image': data})


def main(folder: pathlib.Path):
    for image in folder.glob('*.jpeg'):
        result = process(image)
        p = result["classLabelProbs"]["not"]
        print(f"process {image} -> {p:.0%}")
        level = int(p * 10) * 10

        os.makedirs(f"output/{level}", exist_ok=True)

        shutil.copy(image, f"output/{level}/{image.name}")

if __name__ == "__main__":
    typer.run(main)