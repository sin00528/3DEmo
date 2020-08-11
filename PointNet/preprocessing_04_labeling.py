import os
import numpy as np
import pandas as pd

NPZ_PATH = "data/npys/"

csv = pd.read_csv('./data/fer2013.csv')
csv = csv.reset_index().rename(columns={"index":"filename"})
csv['filename'] = csv['filename'].apply(lambda x : "{0:08d}_0.npy".format(x))

Emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]  # indices 0 to 6

npy = []
for (path, _, files) in os.walk(NPZ_PATH):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        if ext == '.npy':
            npy.append("{}".format(filename))

npy = pd.DataFrame(npy, columns=['filename'])
data = pd.merge(csv, npy, left_on='filename', right_on='filename') # 셀프 조인
data = data.drop(columns='pixels')

# 레이블 데이터 저장
data.to_csv('./data/label.csv', index=False)
data = pd.read_csv('./data/label.csv')
