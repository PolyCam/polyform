'''
File: instant_ngp.py
Polycam Inc.

Created by Chris Heinrich on Tuesday, 1st November 2022
Copyright © 2022 Polycam Inc. All rights reserved.
'''
import os
import json
import numpy as np
from polyform.utils.logging import logger
from polyform.core.capture_folder import *
from polyform.convertors.convertor_interface import ConvertorInterface

class InstantNGPMultiFileConvertor(ConvertorInterface):
    """Converts Polycam data to the format expected by Instant-NGP with each camera having its own JSON file
        so that Instant-NGP will use the per-frame intrinsics"""
    def convert(self, folder: CaptureFolder, output_path: str = ""):
        """
        Converts a Polycam CaptureFolder into an instant-ngp dataset by writing out each camera
        to it's own file (so we can have per-frame camera intrinsics).

        Args:
            folder: the capture folder to convert
            output_path: the path to write the transforms.json file. By default this is
                written at the root of the CaptureFolder's data directory.
        """

        keyframes = folder.get_keyframes()
        bbox = CaptureFolder.camera_bbox(keyframes)
        ## Use camera bbox to compute scale, and offset 
        ## See https://github.com/NVlabs/instant-ngp/blob/master/docs/nerf_dataset_tips.md#scaling-existing-datasets
        scale = 1.0 / np.max(bbox.size()) * 0.6
        offset = np.asarray([0.5,0.5,0.5]) - scale * bbox.center()
        print(bbox)
        for keyframe in keyframes:
            data = {}
            cam = keyframe.camera
            data["fl_x"] = cam.fx
            data["fl_y"] = cam.fy
            data["cx"] = cam.cx
            data["cy"] = cam.cy
            data["w"] = cam.width
            data["h"] = cam.height
            data["scale"] = float(scale)
            data["offset"] = [float(offset[0]), float(offset[1]), float(offset[2])]
            data["aabb_scale"] = 2

            ## add the frame
            data["frames"] = [self._convert_keyframe(keyframe)]

            # Write the output file to json
            output_file_path = os.path.join(folder.root, "{}.json".format(keyframe.timestamp))
            with open(output_file_path, "w") as f:
                json.dump(data, f, indent=2)
        logger.info("Successfuly wrote {} camera files".format(len(keyframes)))


    def _convert_keyframe(self, keyframe: Keyframe) -> dict:
        """ Converts Polycam keyframe into a dictionary to be serialized as json """
        frame = {}
        # add the image
        sub_path = CaptureArtifact.CORRECTED_IMAGES.value if keyframe.is_optimized() else CaptureArtifact.IMAGES.value
        frame["file_path"] = "./{}/{}.jpg".format(sub_path, keyframe.timestamp)
        if keyframe.camera.blur_score:
            frame["sharpness"] = keyframe.camera.blur_score
        frame["transform_matrix"] = keyframe.camera.transform_rows
        return frame