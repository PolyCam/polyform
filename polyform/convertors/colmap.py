'''
File: colmap.py
Polycam Inc.

Converts Polycam raw capture data into the COLMAP text model format
(cameras.txt / images.txt / points3D.txt), so that Polycam captures can be
consumed by any tool in the COLMAP ecosystem (e.g. 3D Gaussian Splatting,
nerfstudio's colmap dataparser, Open3D, MVS pipelines, ...).

See https://colmap.github.io/format.html#text-format for the file spec.

Addresses: https://github.com/PolyCam/polyform/issues/9
"How to get the raw data into Colmap format to use 3DGaussian?"
'''
import os
import numpy as np
from polyform.utils.logging import logger
from polyform.core.capture_folder import *
from polyform.convertors.convertor_interface import ConvertorInterface

# Polycam's raw camera-to-world transform (rotate=False) is expressed in
# ARKit / OpenGL camera-local axes: +X right, +Y up, +Z out of the screen
# (i.e. the camera looks down its local -Z axis).
#
# COLMAP (and OpenCV) use the computer-vision convention: +X right, +Y down,
# +Z into the scene (i.e. the camera looks down its local +Z axis).
#
# The two conventions only differ by a 180 degree rotation about the local
# X axis, so converting one to the other is a matter of flipping the local
# Y and Z axes.
_ARKIT_TO_CV = np.diag([1.0, -1.0, -1.0]).astype(np.float32)


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Converts a 3x3 rotation matrix into a (qw, qx, qy, qz) unit quaternion,
    using COLMAP's convention (scalar-first, Hamilton quaternions).

    Uses Shepperd's method, which picks whichever diagonal term is largest to
    avoid dividing by a near-zero number (a plain "qw = sqrt(trace)/2" formula
    is numerically unstable when qw is close to 0, i.e. for near-180-degree
    rotations, which do show up in real scan trajectories when the camera
    loops back on itself).
    """
    m = R
    if not np.allclose(m @ m.T, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(m), 1.0, atol=1e-3):
        raise ValueError(
            "Input is not a valid rotation matrix (must be orthogonal with determinant 1); "
            "got matrix with det={:.4f}".format(np.linalg.det(m))
        )

    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (m[2, 1] - m[1, 2]) * s
        qy = (m[0, 2] - m[2, 0]) * s
        qz = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s

    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    return q / np.linalg.norm(q)


def arkit_pose_to_colmap(transform: np.ndarray):
    """
    Converts a Polycam/ARKit camera-to-world 4x4 transform (rotate=False
    convention, see Camera in capture_folder.py) into the COLMAP world-to-camera
    rotation (as a wxyz quaternion) and translation expected by images.txt.

    Args:
        transform: 4x4 camera-to-world matrix in ARKit/OpenGL camera-local axes

    Returns:
        (qvec, tvec): qvec is a length-4 np.ndarray [qw, qx, qy, qz], tvec is a
        length-3 np.ndarray [tx, ty, tz]
    """
    R_c2w_arkit = transform[0:3, 0:3]
    t_c2w = transform[0:3, 3]

    # Re-express the camera-to-world rotation in OpenCV/COLMAP camera axes
    R_c2w_cv = R_c2w_arkit @ _ARKIT_TO_CV

    # COLMAP stores WORLD-TO-CAMERA pose, i.e. the inverse of the camera-to-world
    # transform. For a rotation matrix the inverse is just the transpose.
    R_w2c = R_c2w_cv.T
    t_w2c = -R_w2c @ t_c2w

    qvec = _rotation_matrix_to_quaternion(R_w2c)
    return qvec, t_w2c.astype(np.float64)


class COLMAPConvertor(ConvertorInterface):
    """
    Converts Polycam data into the COLMAP text model (cameras.txt, images.txt,
    points3D.txt), so it can be loaded by any COLMAP-compatible tool (e.g. the
    original 3D Gaussian Splatting reference implementation, nerfstudio's
    `colmap` dataparser, or COLMAP itself for further processing).

    NOTE on points3D.txt: Polycam's raw export does not include a sparse
    point cloud with per-image 2D/3D correspondences (the kind COLMAP's own
    feature-matching + triangulation pipeline produces), so we write an empty
    points3D.txt. This is sufficient input for tools that only need camera
    poses/intrinsics (e.g. 3D Gaussian Splatting can initialize from random
    points when points3D.txt is empty). A natural follow-up (left as future
    work, see PR description) would be to back-project Polycam's per-frame
    depth maps into a seed point cloud.
    """

    def __init__(self, shared_camera: bool = False, corrected_image_padding: int = 5):
        """
        Args:
            shared_camera: if True, all images reference a single COLMAP camera
                (using the first keyframe's intrinsics). If False (default), each
                image gets its own camera entry, since Polycam's per-frame
                intrinsics vary by a pixel or two frame-to-frame (same tradeoff
                noted in InstantNGPConvertor).
            corrected_image_padding: cropping applied to the corrected/optimized
                images, mirroring InstantNGPConvertor's handling of the black
                border left by undistortion.
        """
        self.shared_camera = shared_camera
        self.corrected_image_padding = corrected_image_padding

    def convert(self, folder: CaptureFolder, output_path: str = ""):
        """
        Converts a Polycam CaptureFolder into a COLMAP text-format sparse model
        by writing cameras.txt, images.txt and points3D.txt.

        Args:
            folder: the capture folder to convert
            output_path: directory to write the COLMAP model into. Defaults to
                a `colmap_text` directory at the root of the CaptureFolder.
        """
        keyframes = folder.get_keyframes(rotate=False)
        if len(keyframes) == 0:
            logger.error("Capture folder does not have any data! Aborting conversion to COLMAP")
            return

        if not output_path:
            output_path = os.path.join(folder.root, "colmap_text")
        os.makedirs(output_path, exist_ok=True)

        use_corrected = folder.has_optimized_poses()
        camera_lines = []
        image_lines = []

        shared_camera_id = 1
        for idx, keyframe in enumerate(keyframes):
            image_id = idx + 1
            camera_id = shared_camera_id if self.shared_camera else image_id

            cam = keyframe.camera
            if use_corrected:
                width = cam.width - 2 * self.corrected_image_padding
                height = cam.height - 2 * self.corrected_image_padding
                cx = cam.cx - self.corrected_image_padding
                cy = cam.cy - self.corrected_image_padding
                image_name = "{}/{}.jpg".format(CaptureArtifact.CORRECTED_IMAGES.value, keyframe.timestamp)
            else:
                width = cam.width
                height = cam.height
                cx = cam.cx
                cy = cam.cy
                image_name = "{}/{}.jpg".format(CaptureArtifact.IMAGES.value, keyframe.timestamp)

            if not self.shared_camera or idx == 0:
                # PINHOLE model params are: fx, fy, cx, cy
                camera_lines.append(
                    "{} PINHOLE {} {} {} {} {} {}".format(
                        camera_id, width, height, cam.fx, cam.fy, cx, cy
                    )
                )

            qvec, tvec = arkit_pose_to_colmap(keyframe.camera.transform)
            image_lines.append(
                "{} {} {} {} {} {} {} {} {} {}".format(
                    image_id,
                    qvec[0], qvec[1], qvec[2], qvec[3],
                    tvec[0], tvec[1], tvec[2],
                    camera_id,
                    image_name,
                )
            )
            # COLMAP's images.txt alternates: one line of pose data, then one
            # line of (possibly empty) 2D keypoint / 3D point correspondences.
            image_lines.append("")

        self._write_cameras_txt(output_path, camera_lines)
        self._write_images_txt(output_path, image_lines, num_images=len(keyframes))
        self._write_points3D_txt(output_path)

        logger.info("Successfully wrote COLMAP text model to {}".format(output_path))

    @staticmethod
    def _write_cameras_txt(output_path: str, camera_lines):
        path = os.path.join(output_path, "cameras.txt")
        with open(path, "w") as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write("# Number of cameras: {}\n".format(len(camera_lines)))
            for line in camera_lines:
                f.write(line + "\n")

    @staticmethod
    def _write_images_txt(output_path: str, image_lines, num_images: int):
        path = os.path.join(output_path, "images.txt")
        with open(path, "w") as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            f.write("# Number of images: {}, mean observations per image: 0\n".format(num_images))
            for line in image_lines:
                f.write(line + "\n")

    @staticmethod
    def _write_points3D_txt(output_path: str):
        path = os.path.join(output_path, "points3D.txt")
        with open(path, "w") as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            f.write("# Number of points: 0\n")
