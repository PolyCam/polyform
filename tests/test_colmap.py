import os
import numpy as np
import pytest

from polyform.convertors.colmap import (
    COLMAPConvertor,
    arkit_pose_to_colmap,
    _rotation_matrix_to_quaternion,
)


def _quat_to_rotation_matrix(q):
    """Reference (independent) quaternion -> rotation matrix implementation,
    used only in tests to check round-tripping of _rotation_matrix_to_quaternion."""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)],
        [2 * (qx*qy + qz*qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy*qz - qx*qw)],
        [2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx**2 + qy**2)],
    ])


class TestRotationMatrixToQuaternion:
    @pytest.mark.parametrize("R", [
        np.eye(3),
        # 90 degrees about Z
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
        # 90 degrees about X
        np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
        # 180 degrees about Y (trace is 0 -> exercises the trace<=0 branch)
        np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
        # 180 degrees about X
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
        # an arbitrary rotation (about a normalized axis, via Rodrigues' formula)
        None,
    ])
    def test_round_trip(self, R):
        if R is None:
            axis = np.array([1.0, 2.0, 3.0])
            axis = axis / np.linalg.norm(axis)
            theta = 0.7
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

        q = _rotation_matrix_to_quaternion(R.astype(np.float64))

        # quaternion must be unit norm
        assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-9)

        # converting back must reproduce the original rotation matrix
        R_reconstructed = _quat_to_rotation_matrix(q)
        np.testing.assert_allclose(R_reconstructed, R, atol=1e-6)

    def test_rejects_non_rotation_matrix(self):
        with pytest.raises(ValueError):
            _rotation_matrix_to_quaternion(np.zeros((3, 3)))

    def test_rejects_reflection_matrix(self):
        # orthogonal but det=-1 (a reflection, not a rotation) must also be rejected
        with pytest.raises(ValueError):
            _rotation_matrix_to_quaternion(np.diag([1.0, 1.0, -1.0]))


class TestArkitPoseToColmap:
    def test_identity_pose_looks_down_positive_z_in_colmap_frame(self):
        """
        A camera at the world origin with an identity ARKit rotation looks down
        its local -Z axis (ARKit/OpenGL convention). After conversion to COLMAP's
        convention the same camera must look down its local +Z axis, i.e. the
        world-to-camera rotation should equal diag(1, -1, -1).
        """
        transform = np.eye(4)
        qvec, tvec = arkit_pose_to_colmap(transform)
        R_reconstructed = _quat_to_rotation_matrix(qvec)
        np.testing.assert_allclose(R_reconstructed, np.diag([1, -1, -1]), atol=1e-6)
        np.testing.assert_allclose(tvec, [0, 0, 0], atol=1e-6)

    def test_translation_is_inverted_into_camera_frame(self):
        """A camera translated along world +X (with identity rotation) should end
        up with a world-to-camera translation of -X in COLMAP's frame."""
        transform = np.eye(4)
        transform[0, 3] = 5.0
        qvec, tvec = arkit_pose_to_colmap(transform)
        np.testing.assert_allclose(tvec, [-5.0, 0.0, 0.0], atol=1e-6)


class TestCOLMAPConvertor:
    def test_writes_expected_files(self, raw_capture_folder, tmp_path):
        out_dir = tmp_path / "colmap_out"
        COLMAPConvertor().convert(raw_capture_folder, output_path=str(out_dir))

        assert (out_dir / "cameras.txt").exists()
        assert (out_dir / "images.txt").exists()
        assert (out_dir / "points3D.txt").exists()

    def test_per_image_cameras_by_default(self, raw_capture_folder, tmp_path):
        out_dir = tmp_path / "colmap_out"
        COLMAPConvertor().convert(raw_capture_folder, output_path=str(out_dir))

        cameras = _read_data_lines(out_dir / "cameras.txt")
        images = _read_data_lines(out_dir / "images.txt", stride=2)

        # 3 synthetic keyframes were created in the fixture
        assert len(cameras) == 3
        assert len(images) == 3

        for cam_line in cameras:
            fields = cam_line.split()
            assert fields[1] == "PINHOLE"

    def test_shared_camera_option(self, raw_capture_folder, tmp_path):
        out_dir = tmp_path / "colmap_out"
        COLMAPConvertor(shared_camera=True).convert(raw_capture_folder, output_path=str(out_dir))

        cameras = _read_data_lines(out_dir / "cameras.txt")
        images = _read_data_lines(out_dir / "images.txt", stride=2)

        assert len(cameras) == 1
        assert len(images) == 3
        for image_line in images:
            camera_id = image_line.split()[8]
            assert camera_id == "1"

    def test_quaternions_are_unit_norm(self, raw_capture_folder, tmp_path):
        out_dir = tmp_path / "colmap_out"
        COLMAPConvertor().convert(raw_capture_folder, output_path=str(out_dir))

        for image_line in _read_data_lines(out_dir / "images.txt", stride=2):
            fields = image_line.split()
            q = np.array([float(x) for x in fields[1:5]])
            assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-6)

    def test_points3d_is_empty(self, raw_capture_folder, tmp_path):
        out_dir = tmp_path / "colmap_out"
        COLMAPConvertor().convert(raw_capture_folder, output_path=str(out_dir))
        assert _read_data_lines(out_dir / "points3D.txt") == []

    def test_optimized_folder_uses_corrected_images_and_cropped_intrinsics(
        self, optimized_capture_folder, tmp_path
    ):
        out_dir = tmp_path / "colmap_out"
        convertor = COLMAPConvertor(corrected_image_padding=5)
        convertor.convert(optimized_capture_folder, output_path=str(out_dir))

        cameras = _read_data_lines(out_dir / "cameras.txt")
        images = _read_data_lines(out_dir / "images.txt", stride=2)
        assert len(cameras) == len(images) == 3

        first_cam_fields = cameras[0].split()
        width, height = int(first_cam_fields[2]), int(first_cam_fields[3])
        # fixture images are 64x48, padding is 5 on each side
        assert width == 64 - 2 * 5
        assert height == 48 - 2 * 5

        for image_line in images:
            name = image_line.split()[9]
            assert "corrected_images" in name

    def test_empty_folder_does_not_raise(self, empty_capture_folder, tmp_path):
        out_dir = tmp_path / "colmap_out"
        # Should log an error and return early rather than raising/crashing
        COLMAPConvertor().convert(empty_capture_folder, output_path=str(out_dir))
        assert not out_dir.exists()

    def test_default_output_path_is_under_folder_root(self, raw_capture_folder):
        COLMAPConvertor().convert(raw_capture_folder)
        default_out = os.path.join(raw_capture_folder.root, "colmap_text")
        assert os.path.exists(os.path.join(default_out, "cameras.txt"))


def _read_data_lines(path, stride=1):
    """Reads a COLMAP text file, stripping comment (#) lines and blank lines,
    returning every `stride`-th remaining line (use stride=2 for images.txt,
    where each entry is a pose line followed by a POINTS2D line)."""
    with open(path) as f:
        raw_lines = [line.rstrip("\n") for line in f if not line.startswith("#")]
    # drop fully-blank trailing artifacts but keep intentional empty POINTS2D lines
    while raw_lines and raw_lines[-1] == "" and stride == 1:
        raw_lines.pop()
    if stride > 1:
        return raw_lines[::stride]
    return [line for line in raw_lines if line != ""]
