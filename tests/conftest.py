import json
import os
import numpy as np
import pytest
from PIL import Image

from polyform.core.capture_folder import CaptureFolder, CaptureArtifact


def _write_camera_json(path, fx, fy, cx, cy, width, height, transform, blur_score=1.0):
    """
    Writes a Polycam-style camera json. `transform` is a 3x4 camera-to-world
    matrix (row-major) using the ARKit raw (t_00..t_23) key layout.
    """
    j = {
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "width": width, "height": height, "blur_score": blur_score,
    }
    for row in range(3):
        for col in range(4):
            j["t_{}{}".format(row, col)] = float(transform[row, col])
    with open(path, "w") as f:
        json.dump(j, f)


def _make_capture_folder(tmp_path, num_frames=3, optimized=False, image_size=(64, 48)):
    """
    Builds a minimal, valid Polycam raw-data folder on disk with `num_frames`
    synthetic keyframes, following the on-disk layout CaptureFolder expects
    (see CaptureArtifact in polyform/core/capture_folder.py).
    """
    root = tmp_path / "capture"
    for artifact in [CaptureArtifact.IMAGES, CaptureArtifact.CAMERAS, CaptureArtifact.DEPTH_MAPS]:
        os.makedirs(root / artifact.value, exist_ok=True)
    if optimized:
        for artifact in [CaptureArtifact.CORRECTED_IMAGES, CaptureArtifact.CORRECTED_CAMERAS]:
            os.makedirs(root / artifact.value, exist_ok=True)

    width, height = image_size
    rng = np.random.default_rng(seed=0)

    for i in range(num_frames):
        timestamp = 1000 + i

        # Build a simple camera-to-world transform: identity rotation with a
        # translation that moves along +X per frame, plus one frame with a
        # non-trivial rotation so the quaternion math is actually exercised.
        R = np.eye(3, dtype=np.float32)
        if i == 1:
            # 90 degree rotation about the camera-local Y axis
            R = np.asarray([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)
        t = np.asarray([float(i), 0.0, 0.0], dtype=np.float32)
        transform = np.concatenate([R, t.reshape(3, 1)], axis=1)

        cam_path = root / CaptureArtifact.CAMERAS.value / "{}.json".format(timestamp)
        _write_camera_json(
            cam_path, fx=500 + i, fy=500 + i, cx=width / 2, cy=height / 2,
            width=width, height=height, transform=transform,
        )

        img_path = root / CaptureArtifact.IMAGES.value / "{}.jpg".format(timestamp)
        Image.fromarray((rng.random((height, width, 3)) * 255).astype(np.uint8)).save(img_path)

        depth_path = root / CaptureArtifact.DEPTH_MAPS.value / "{}.png".format(timestamp)
        Image.fromarray((rng.random((height, width)) * 1000).astype(np.uint16)).save(depth_path)

        if optimized:
            corrected_cam_path = root / CaptureArtifact.CORRECTED_CAMERAS.value / "{}.json".format(timestamp)
            _write_camera_json(
                corrected_cam_path, fx=500 + i, fy=500 + i, cx=width / 2, cy=height / 2,
                width=width, height=height, transform=transform,
            )
            corrected_img_path = root / CaptureArtifact.CORRECTED_IMAGES.value / "{}.jpg".format(timestamp)
            Image.fromarray((rng.random((height, width, 3)) * 255).astype(np.uint8)).save(corrected_img_path)

    return CaptureFolder(str(root))


@pytest.fixture
def raw_capture_folder(tmp_path):
    return _make_capture_folder(tmp_path, num_frames=3, optimized=False)


@pytest.fixture
def optimized_capture_folder(tmp_path):
    return _make_capture_folder(tmp_path, num_frames=3, optimized=True, image_size=(64, 48))


@pytest.fixture
def empty_capture_folder(tmp_path):
    root = tmp_path / "empty_capture"
    os.makedirs(root, exist_ok=True)
    return CaptureFolder(str(root))
