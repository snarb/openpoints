"""Modified from DeepGCN and DGCNN
Reference: https://github.com/lightaime/deep_gcns_torch/tree/master/examples/classification
"""
import os
import glob
import h5py
import numpy as np
import pickle
import logging
import ssl
import urllib
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import extract_archive, check_integrity
from ..build import DATASETS
import trimesh
from openpoints.utils.utils_3shape import *

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def get_plane(plane_fname):
    #MEAN_ORIG = np.array([-1.9013895 , 12.77964562, -4.90804974])
    #MEAN_NORMAL = np.array([-0.34115871,  0.84640487,  0.17356184])

    crimefile = open(plane_fname, 'r')
    line = crimefile.readlines()[0].replace('\n', '')
    data = np.array(line.split(',')).astype(float).reshape(2, 3)
    normals = data[0, :]
    origin = data[1, :]
    return origin, normals
    #return origin - MEAN_ORIG, normals - MEAN_NORMAL

def load_data(data_dir, partition):
    # mesh_o = trimesh.load_mesh(r"C:\temp\Segmented\{}.ply".format(fname))
    POINT_CLOUD_SCALE_FACTOR = 16.6
    all_data = []
    all_label = []
    all_path = []
    all_normals = []
    for plane_fpath in glob.glob(os.path.join(data_dir, partition, 'planes', '*.txt')):
        origin, normals = get_plane(plane_fpath)
        fname = os.path.splitext(os.path.basename(plane_fpath))[0]
        mesh_path = os.path.join(data_dir, partition, 'meshes', fname + '.ply')
        all_path.append((mesh_path, plane_fpath))
        mesh = trimesh.load_mesh(mesh_path)
        point_cloud = np.array(mesh.vertices)
        point_cloud /= POINT_CLOUD_SCALE_FACTOR
        point_cloud = point_cloud.astype(np.float32)

        R = normal_to_rotation_matrix(normals)
        T = np.eye(4)
        # Substitute the rotation and translation components
        T[:3, :3] = R
        T[:3, 3] = origin

        all_data.append(point_cloud)
        all_label.append(T)
        all_normals.append(normals)
        # if len(all_data) > 100:
        #     break
    all_label = np.stack(all_label, axis=0)
    return all_data, all_label, all_path, all_normals


@DATASETS.register_module()
class PlaneRegression(Dataset):
    """
    This is the data loader for ModelNet 40
    ModelNet40 contains 12,311 meshed CAD models from 40 categories.
    num_points: 1024 by default
    data_dir
    paritition: train or test
    """
    dir_name = 'modelnet40_ply_hdf5_2048'

    def __init__(self,
                 num_points=1024,
                 data_dir="./data/PlaneRegression",
                 split='train',
                 transform=None
                 ):
        data_dir = os.path.join(os.getcwd(), data_dir) if data_dir.startswith('.') else data_dir
        self.partition = 'train' if split.lower() == 'train' else 'val'  # val = test
        self.data, self.label, self.path, self.normals = load_data(data_dir, self.partition)
        self.num_points = num_points
        logging.info(f'==> sucessfully loaded {self.partition} data')
        self.transform = transform

    def __getitem__(self, item):
        pointcloud = farthest_point_sample(self.data[item], self.num_points)

        #pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        mesh_path = self.path[item][0]
        plane_path = self.path[item][1]

        if self.partition == 'train':
            np.random.shuffle(pointcloud)
        data = {'pos': pointcloud,
                'y': label
                }
        if self.transform is not None:
            data = self.transform(data)

        data['x'] = data['pos']
        data['mesh_path'] = mesh_path
        data['plane_path'] = plane_path
        data['normals'] = self.normals[item]
        return data

    def __len__(self):
        return len(self.data)

