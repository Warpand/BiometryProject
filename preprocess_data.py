import os.path
from pathlib import Path

import cv2
import pandas as pd
import tqdm.contrib.concurrent

VARIANCE_THRESHOLD = 100.0
DATASET_PATH = "dataset/casia-webface"


def variance_filter(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    variance = cv2.Laplacian(img, cv2.CV_64F).var()
    return variance > VARIANCE_THRESHOLD


if __name__ == "__main__":
    images = list(Path(DATASET_PATH).rglob("*.jpg"))
    is_not_blurred_map = tqdm.contrib.concurrent.process_map(
        variance_filter, images, chunksize=8
    )
    metadata = {"file": [], "id": []}
    for path, _ in filter(lambda t: t[1], zip(images, is_not_blurred_map)):
        _id = path.parts[2].lstrip("0")
        _id = int(_id) if _id else 0
        filename = os.path.join(*path.parts[1:])
        metadata["file"].append(filename)
        metadata["id"].append(_id)
    num_filtered = len(images) - len(metadata["id"])
    df = pd.DataFrame(metadata)
    df.to_csv("dataset/metadata.csv", index=False)
    print(f"Filtered {num_filtered} out of {len(images)} images.")
    print(f"Filtered {num_filtered * 100.0 / len(images)}% images.")
