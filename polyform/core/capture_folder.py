"""
The CaptureFolder and associated types are the main wrappers around Polycam data as it is laid out on the filesystem. 

Initializing a CapureFolder from the absolute path of the data folder on the filesystem is usually the first step to 
working with a raw Polycam dataset.
"""

import os
import json
from polyform.utils.logging import logger
from enum import Enum
from typing import List


class CaptureArtifact(Enum):
    IMAGES = "keyframes/images"
    CORRECTED_IMAGES = "keyframes/corrected_images"
    CAMERAS = "keyframes/cameras"
    CORRECTED_CAMERAS = "keyframes/corrected_cameras"
    DEPTH_MAPS = "keyframes/depth"
    CONFIDENCE_MAPS = "keyframes/confidence"
    MESH_INFO = "mesh_info.json"
    ANCHORS = "anchors.json"
    PREVIEW_MESH = "mesh.obj"


class Camera:
    def __init__(self, j: dict):
        """ Initializes a Camera object from the Polycam camera json format """
        self.fx = j["fx"]
        self.fy = j["fy"]
        self.cx = j["cx"]
        self.cy = j["cy"]
        self.width = j["width"]
        self.height = j["height"]
        self.blur_score = j["blur_score"]
        self.transform_rows = [
                [j["t_00"], j["t_01"], j["t_02"], j["t_03"]],
                [j["t_10"], j["t_11"], j["t_12"], j["t_13"]],
                [j["t_20"], j["t_21"], j["t_22"], j["t_23"]],
                [0,0,0,1]]

class Keyframe:
    """
    A Keyframe includes the camera information (extrinsics and intrinsics) as well as the 
    path to the associated data on disk (images, depth map, confidence)
    """
    def __init__(self, folder: str, timestamp: int):
        self.folder = folder
        self.timestamp = timestamp
        self.image_path = os.path.join(
            folder, "{}/{}.jpg".format(CaptureArtifact.IMAGES.value, timestamp))
        self.corrected_image_path = os.path.join(
            folder, "{}/{}.jpg".format(CaptureArtifact.CORRECTED_IMAGES.value, timestamp))
        self.camera_path = os.path.join(
            folder, "{}/{}.json".format(CaptureArtifact.CAMERAS.value, timestamp))
        self.corrected_camera_path = os.path.join(
            folder, "{}/{}.json".format(CaptureArtifact.CORRECTED_CAMERAS.value, timestamp))
        self.depth_path = os.path.join(
            folder, "{}/{}.png".format(CaptureArtifact.DEPTH_MAPS.value, timestamp))
        self.camera = Camera(self.get_best_camera_json())

    def is_valid(self) -> bool:
        if not os.path.isfile(self.camera_path):
            return False
        if not os.path.isfile(self.image_path):
            return False
        if not os.path.isfile(self.depth_path):
            return False
        return True

    def is_optimized(self) -> bool:
        if not os.path.isfile(self.corrected_camera_path):
            return False
        if not os.path.isfile(self.corrected_image_path):
            return False
        return True

    def get_best_camera_json(self) -> dict:
        """ Returns the camera json for the optimized camera if it exists, othewise returns the ARKit camera """
        if self.is_optimized():
            return CaptureFolder.load_json(self.corrected_camera_path)
        else:
            return CaptureFolder.load_json(self.camera_path)


    def __str__(self):
        return "keyframe:{}".format(self.timestamp)


class CaptureFolder:
    def __init__(self, root: str):
        self.root = root
        self.id = os.path.basename(os.path.normpath(root))
        logger.warning("Camera poses have not been optimized, the ARKit poses will be used as a fallback")
        if not self.has_optimized_poses():
            logger.warning("Camera poses have not been optimized, the ARKit poses will be used as a fallback")

    def get_artifact_path(self, artifact: CaptureArtifact) -> str:
        return os.path.join(self.root, artifact.value)

    def get_artifact_paths(self, folder_artifact: CaptureArtifact,
                           file_extension: str) -> List[str]:
        """ 
        Returns all of the artifacts located in the folder_artifact with the provided file_extension
        """
        paths = []
        folder_path = self.get_artifact_path(folder_artifact)
        if not os.path.isdir(folder_path):
            return paths
        return [os.path.join(folder_path, path) for path in sorted(os.listdir(folder_path)) if path.endswith(file_extension)]

    def has_artifact(self, artifact: CaptureArtifact) -> bool:
        return os.path.exists(self.get_artifact_path(artifact))

    def has_optimized_poses(self) -> bool:
        return (self.has_artifact(CaptureArtifact.CORRECTED_CAMERAS) and self.has_artifact(CaptureArtifact.CORRECTED_IMAGES))

    def get_image_paths(self) -> List[str]:
        return self.get_artifact_paths(CaptureArtifact.IMAGES, "jpg")

    def get_camera_paths(self) -> List[str]:
        return self.get_artifact_paths(CaptureArtifact.CAMERAS, "json")

    def get_depth_paths(self) -> List[str]:
        return self.get_artifact_paths(CaptureArtifact.DEPTH_IMAGES, "png")

    def get_keyframe_timestamps(self) -> List[int]:
        timestamps = []
        folder_path = self.get_artifact_path(CaptureArtifact.CAMERAS)
        if not os.path.isdir(folder_path):
            return timestamps
        return [int(path.replace(".json", "")) for path in sorted(os.listdir(folder_path)) if path.endswith("json")]

    def get_keyframes(self) -> List[Keyframe]:
        """
        Returns all valid keyframes associated with this dataset
        """
        keyframes = []
        for ts in self.get_keyframe_timestamps():
            keyframe = Keyframe(self.root, ts)
            if keyframe.is_valid():
                keyframes.append(keyframe)
        return keyframes

    @staticmethod
    def load_json(path) -> dict:
        name, ext = os.path.splitext(path)
        if not os.path.exists(path) or ext.lower() != ".json":
            print("File at path {} did not exist or was not a json file. Returning empty dict".format(path))
            return {}
        else:
            with open(path) as f:
                return json.load(f)

    def load_json_artifact(self, folder_artifact: CaptureArtifact) -> dict:
        path = self.get_artifact_path(folder_artifact)
        return CaptureFolder.load_json(path)


