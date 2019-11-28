from torch.utils.data import Dataset
import os
from utils import find_nearest_point
import numpy as np
import torch
import cv2
import pandas as pd
import albumentations as A



class FingerDataset(Dataset):
    """
    Dataset with cache. Examples come from cache
    """
    def __init__(self, names, image_dir, point_dir, transform, cache_size, patch_size, inter_dist):
        #store filenames
        self.image_names = [os.path.join(image_dir, f+'.jpg') for f in names]
        self.point_names = [os.path.join(point_dir, f+'.txt') for f in names]
        self.transform = transform
        self.random_crop = A.Compose([A.RandomCrop(*patch_size)], keypoint_params = A.KeypointParams(format='xy'))
        self.det_crop = A.Compose([A.Crop()], keypoint_params = A.KeypointParams(format='xy'))
        self.patch_size = patch_size
        self.center = (patch_size[1]//2, patch_size[0]//2)
        self.inter_dist = inter_dist
        self._cache_patch = []
        self._cache_points = []
        self._cache_size = cache_size
        self.start_index = 0
        # fill cache
        self.update_n_samples = 20
        for i in range((cache_size + self.update_n_samples - 1) // self.update_n_samples):
            self.update_cache()


    def __len__(self):
        #return size of dataset
        return len(self._cache_patch)


    def _write_to_cache(self, patch, points):
        if len(self._cache_patch) == self._cache_size:
            self._cache_patch[self.start_index] = patch
            self._cache_points[self.start_index] = points
            self.start_index = (self.start_index+1) % self._cache_size
        else:
            self._cache_patch.append(patch)
            self._cache_points.append(points)


    def _create_vec(self, points):
        if len(points) == 0:
            return (0,0,0)
        dist,x,y = find_nearest_point(*self.center,points)
        if dist <= self.inter_dist:
            vec = (1,(x-self.center[0])/self.inter_dist,(y-self.center[1])/self.inter_dist)
        else:
            vec = (0,0,0)
        return vec


    def __getitem__(self, index):
        patch = self.transform(image=self._cache_patch[index], keypoints=self._cache_points[index])
        vec = self._create_vec(patch['keypoints'])
        return torch.tensor(patch['image'], dtype=torch.float).unsqueeze(0), torch.tensor(vec, dtype=torch.float)


    def update_cache(self, ratio = 0.7):
        index = np.random.randint(len(self.image_names))
        image = (cv2.imread(self.image_names[index],-1)/255).astype('float32')
        points = pd.read_csv(self.point_names[index],delimiter="\t").values
        #print_with_points(image,points)
        r = int(self.update_n_samples*ratio)
        #random
        for i in range(r):
            patch = self.random_crop(image=image,keypoints=points)
            self._write_to_cache(patch['image'], patch['keypoints'])
        # with keypoints
        for i in range(self.update_n_samples-r):
            c = points[np.random.randint(len(points))]
            shift  = np.random.randint(-self.inter_dist,self.inter_dist+1, size=2)
            c = np.clip(shift+c,self.center,(image.shape[1]-self.center[0]-1,image.shape[0]-self.center[1]-1))
            self.det_crop[0].x_min, self.det_crop[0].y_min = c-self.center
            self.det_crop[0].x_max = self.det_crop[0].x_min + self.patch_size[1]
            self.det_crop[0].y_max = self.det_crop[0].y_min + self.patch_size[0]
            patch = self.det_crop(image=image,keypoints=points)
            #print_with_points(patch['image'],patch['keypoints'])
            self._write_to_cache(patch['image'], patch['keypoints'])
