import os
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from tqdm import tqdm

NPY_PATH = "data/npys/"
OUT_PATH = 'data/'
nClasses = 7

NUM_POINTS = 68
Emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]  # indices 0 to 6

data = pd.read_csv('./data/label.csv')
print(data)

pts = []

for (path, _, files) in os.walk(NPY_PATH):
    for filename in tqdm(files):
        ext = os.path.splitext(filename)[-1]
        if ext == '.npy':
            pt = np.load(path + filename)
            pts.append(pt)
            #pt.close()

print(pts)
pts_srs = pd.Series(pts)
data["pts"] = pts_srs

# TODO : data 데이터 프레임에 pts 라는 새로운 칼럼 추가 후, 훈련/검증/테스트 데이터로 분리하기
train_set = data[data.Usage == "Training"]
val_set = data[data.Usage == "PublicTest"]
test_set = data[data.Usage == "PrivateTest"]

# 데이터 구성 확인
print(data["Usage"].value_counts())


# TODO : 데이터 세트를 (Batch_size, numbers, 3) 형태로 3D Array화 시키기
def fer2013_to_X(table):
    X = []
    pixels_list = table["pts"].values

    for xyz in tqdm(pixels_list):
        X = np.append(X, xyz)

    X = X.reshape((-1, NUM_POINTS, 3))

    return X


# 레이블 추출
def fer2013_to_y(table):
    y = []
    labeled_list = table["emotion"].values
    labeled_list = to_categorical(labeled_list, nClasses)
    y.append(labeled_list)
    y = np.squeeze(y)

    return y


# 변환
train_X = fer2013_to_X(train_set)
val_X = fer2013_to_X(val_set)
test_X = fer2013_to_X(test_set)

train_y = fer2013_to_y(train_set)
val_y = fer2013_to_y(val_set)
test_y = fer2013_to_y(test_set)

# x 형태 출력
print("train_X shape : ", np.shape(train_X))
print("val_X shape : ", np.shape(val_X))
print("test_X shape : ", np.shape(test_X))

# 레이블 형태 출력
print("train_y shape : ", np.shape(train_y))
print("val_y shape : ", np.shape(val_y))
print("test_y shape : ", np.shape(test_y))

# TODO : feature와 lable을 npy 파일로 저장하기 (3개 세트 * 2)
# 저장
np.save(OUT_PATH + "train_X", train_X)
np.save(OUT_PATH + "val_X", val_X)
np.save(OUT_PATH + "test_X", test_X)

np.save(OUT_PATH + "train_y", train_y)
np.save(OUT_PATH + "val_y", val_y)
np.save(OUT_PATH + "test_y", test_y)
