import os
import json
from polyform.utils.logging import logger
from polyform.core.capture_folder import *
from polyform.convertors.convertor_interface import ConvertorInterface

class InstantNGPConvertor(ConvertorInterface):
    """Converts Polycam data to the format expected by Instant-NGP"""
    def convert(self, folder: CaptureFolder, output_path: str = ""):
        """
        Converts a Polycam CaptureFolder into an instant-ngp dataset by writing out the
        transforms.json file

        Args:
            folder: the capture folder to convert
            output_path: the path to write the transforms.json file. By default this is
                written at the root of the CaptureFolder's data directory.
        """
        data = {}
        keyframes = folder.get_keyframes()
        if len(keyframes) == 0:
            logger.error("Capture folder does not have any data! Aborting conversion to Instant NGP")
            return
        """
        NOTE
        We use the intrinsics from the first camera to populate the camera data in the transforms.json file.
        However the intrinsics do vary slightly (a pixel or two) frame-to-frame, but instant-ngp does not support
        having different intrinsics for each frame. If you are working with a NeRF implementation that does then
        it would be better to use the frame-dependent intrinsics.
        """
        cam = keyframes[0].camera
        data["fl_x"] = cam.fx
        data["fl_y"] = cam.fy
        data["cx"] = cam.cx
        data["cy"] = cam.cy
        data["w"] = cam.width
        data["h"] = cam.height

        """
        TODO
        - estimate the real aabb_scale
        - add the camera angle formulation (if required)
        """
        data["aabb_scale"] = 16 # constant value until we estimate this from the data

        ## add the frames
        frames = []
        for keyframe in keyframes:
            frames.append(self._convert_keyframe(keyframe))
        data["frames"] = frames

        # Write the output file to json
        output_file_path = os.path.join(folder.root, "transforms.json")
        with open(output_file_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Successfuly wrote the data to {}".format(output_file_path))


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