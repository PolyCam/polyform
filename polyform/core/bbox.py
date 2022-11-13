'''
File: bbox.py
Polycam Inc.

Created by Chris Heinrich on Saturday, 5th November 2022
Copyright © 2022 Polycam Inc. All rights reserved.
'''

import numpy as np
from typing import List

class BBox3D:
    def __init__(self, bbox_min: np.ndarray, bbox_max: np.ndarray):
        """
        Initializes a 3D axis-aligned bbox from the min and max points
        
        Args:
            bbox_min: the minimum point, expected to be a 
        """
        assert bbox_min.shape == np.shape([0,0,0]), "bbox_min must be a numpy array of shape (3,)"
        assert bbox_max.shape == np.shape([0,0,0]), "bbox_max must be a numpy array of shape (3,)"
        self.min = bbox_min
        self.max = bbox_max
    
    def center(self) -> np.ndarray:
        return (self.min + self.max) / 2.0
    
    def size(self) -> np.ndarray:
        return (self.max - self.min)

    def __str__(self):
        return "\n *** BBox3D *** \nmin: {}\nmax: {}\nsize: {}\ncenter: {}\n".format(self.min, self.max, self.size(), self.center())


def bbox_from_points(points: List[np.ndarray]) -> BBox3D:
    """
    Generates a bounding box from a list of points (i.e. 3D numpy arrays)
    """
    finfo = np.finfo(np.float32)
    bbox_min = np.asarray([finfo.max, finfo.max, finfo.max])
    bbox_max = np.asarray([finfo.min, finfo.min, finfo.min])
    for point in points:
        # x
        if point[0] < bbox_min[0]:
            bbox_min[0] = point[0]
        if point[0] > bbox_max[0]:
            bbox_max[0] = point[0]
        # y
        if point[1] < bbox_min[1]:
            bbox_min[1] = point[1]
        if point[1] > bbox_max[1]:
            bbox_max[1] = point[1]
         #z
        if point[2] < bbox_min[2]:
            bbox_min[2] = point[2]
        if point[2] > bbox_max[2]:
            bbox_max[2] = point[2]

    return BBox3D(bbox_min, bbox_max)