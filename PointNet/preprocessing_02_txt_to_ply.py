import os
import open3d as o3d
import numpy as np
from tqdm import tqdm

# for multiprocessing
from multiprocessing import Pool

TXT_PATH = "data/txts/"
OUT_DIR = "data/plys/"

data = []
filenames = []
for (path, _, files) in os.walk(TXT_PATH):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        if ext == '.txt':
            #print("{}{}".format(path, filename))
            data.append("{}{}".format(path, filename))
            filenames.append("{}".format(filename))


def savePly(index):
    xyz = np.loadtxt(data[index], delimiter=' ', unpack=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud(OUT_DIR + "{}.ply".format(filenames[index].split('.')[-2]), pcd)
    #np.save(OUT_DIR + "{}".format(filenames[index].split('.')[-2]), pts)
    #np.savez_compressed(OUT_DIR + "{}".format(data[index].split('/')[-1].split('.')[-2]), pts)
    #np.savez_compressed(OUT_DIR + "{}".format(data[index].split('/')[-1]), pts)


with Pool(processes=32) as p:
    max_ = len(data)
    with tqdm(total=max_) as pbar:
        for i, _ in tqdm(enumerate(p.imap_unordered(savePly, range(max_)))):
            pbar.update()


"""
xyz = np.loadtxt("./data/txts/00000000_0.txt", delimiter=' ', unpack=True)
print(xyz)
print(xyz.shape)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
o3d.io.write_point_cloud("test.ply", pcd)
"""

# TODO: ply visualize
"""
pcd_load = o3d.io.read_point_cloud("test.ply")
xyz_load = np.asarray(pcd_load.points)
print(xyz_load)
o3d.visualization.draw_geometries([pcd_load])
"""