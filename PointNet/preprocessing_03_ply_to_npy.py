import os
import open3d as o3d
import numpy as np
from tqdm import tqdm

# for multiprocessing
from multiprocessing import Pool

PLY_PATH = "data/plys/"
OUT_DIR = "data/npys/"
#OUT_DIR = "data/npzs/"

data = []
filenames = []
for (path, _, files) in os.walk(PLY_PATH):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        if ext == '.ply':
            #print("{}{}".format(path, filename))
            data.append("{}{}".format(path, filename))
            filenames.append("{}".format(filename))


def saveNpz(index):
    pc = o3d.io.read_point_cloud(data[index])
    pts = np.array(pc.points)
    #downpcd = pts.voxel_down_sample(voxel_size=5)
    np.save(OUT_DIR + "{}".format(filenames[index].split('.')[-2]), pts)
    #np.savez_compressed(OUT_DIR + "{}".format(data[index].split('/')[-1].split('.')[-2]), pts)
    #np.savez_compressed(OUT_DIR + "{}".format(data[index].split('/')[-1]), pts)


with Pool(processes=32) as p:
    max_ = len(data)
    with tqdm(total=max_) as pbar:
        for i, _ in tqdm(enumerate(p.imap_unordered(saveNpz, range(max_)))):
            pbar.update()

"""
#print(dirs)
#pc = o3d.io.read_point_cloud(PLY_PATH +)

# 픽셀 수 확인
#print(pc)

# ply to numpy
#pts = np.array(pc.points)

# 화면에 표시
#o3d.visualization.draw_geometries([pc])

# numpy파일을 npz 파일로 저장
#np.save('test', pts)
#np.savez_compressed('test', pts)
"""