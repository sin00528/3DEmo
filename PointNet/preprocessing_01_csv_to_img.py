import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# for multiprocessing
from multiprocessing import Pool

FER2013_PATH = "data/fer2013.csv"
FER2013_WIDTH = 48
FER2013_HEIGHT = 48
OUT_DIR = "data/images/"

data = pd.read_csv(FER2013_PATH)
data = data.reset_index().rename(columns={"index":"filename"})
data['filename'] = data['filename'].apply(lambda x : "{0:08d}.png".format(x))

Emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

data["emotion"].value_counts(sort=False)

def saveImg(index):
    image = np.reshape(data.at[index, "pixels"].split(" "), (FER2013_WIDTH, FER2013_HEIGHT)).astype("uint8")
    im = Image.fromarray(image, mode='L')
    im.save(OUT_DIR + data.at[index, "filename"])

with Pool(processes=32) as p:
    max_ = len(data)
    with tqdm(total=max_) as pbar:
        for i, _ in tqdm(enumerate(p.imap_unordered(saveImg, range(max_)))):
            pbar.update()
