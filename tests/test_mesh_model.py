"""Tests for the skinned rat mesh (rpimocap.model.mesh_model)."""
import numpy as np

from rpimocap.model.mesh_model import (RatMesh, build_rat_mesh,
                                       render_mesh_pose_silhouette, skin_mesh)
from rpimocap.model.rat_skeleton import RatPose
from tests.test_body_model import _P


class TestMeshModel:

    @classmethod
    def setup_class(cls):
        cls.mesh = build_rat_mesh(voxel=4.0)          # coarse grid for speed

    def test_mesh_nonempty_and_weights_normalized(self):
        m = self.mesh
        assert isinstance(m, RatMesh)
        assert len(m.verts_rest) > 100 and len(m.faces) > 100
        assert m.weights.shape[0] == len(m.verts_rest)
        assert np.allclose(m.weights.sum(1), 1.0, atol=1e-6)   # partition of unity

    def test_root_translation_is_rigid(self):
        m = self.mesh
        v0 = skin_mesh(m, RatPose(root_pos=np.array([0.0, 0.0, 0.0])))
        v1 = skin_mesh(m, RatPose(root_pos=np.array([100.0, 0.0, 0.0])))
        assert np.allclose(v1 - v0, np.array([100.0, 0.0, 0.0]), atol=1e-6)

    def test_joint_bend_deforms_non_rigidly(self):
        m = self.mesh
        rest = skin_mesh(m, RatPose())
        bent = skin_mesh(m, RatPose(joint_angles={"SpineL": (0.0, 0.6, 0.0)}))
        disp = np.linalg.norm(bent - rest, axis=1)
        assert disp.max() > 3.0            # the rear half swings
        assert disp.min() < 0.5            # the head/front stays put (non-rigid)

    def test_scale_grows_mesh(self):
        m = self.mesh
        small = skin_mesh(m, RatPose(scale=0.8))
        big = skin_mesh(m, RatPose(scale=1.4))
        assert np.ptp(big, axis=0).sum() > np.ptp(small, axis=0).sum()

    def test_render_nonempty(self):
        sil = render_mesh_pose_silhouette(
            self.mesh, RatPose(root_pos=np.array([0.0, 0.0, 60.0])),
            _P(), image_shape=(500, 600))
        assert sil.dtype == np.uint8 and int((sil > 0).sum()) > 500
