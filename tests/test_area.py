import unittest
import numpy as np

from area import apply_affine_transform, PolygonArea, CosineWaveArea


class TestArea(unittest.TestCase):
    def test_affine_transform_identity(self):
        pts = np.array([[0.0, 0.0], [2.0, 3.0], [1.5, -1.0]])
        M = np.eye(3)
        out = apply_affine_transform(pts, M)
        np.testing.assert_allclose(out, pts)

    def test_polygon_mask_simple_square(self):
        # 10x10 grid, square from (2,2) to (7,7)
        nx, ny = 10, 10
        verts = np.array([[2, 2], [7, 2], [7, 7], [2, 7]], dtype=float)
        area = PolygonArea("sq", verts, config={})
        mask = area.compute_mask(nx, ny)
        self.assertEqual(mask.shape, (ny, nx))
        self.assertTrue(mask[3:7, 3:7].all())
        self.assertFalse(mask[0, 0])

    def test_cosine_wave_inside_ellipse(self):
        nx, ny = 20, 20
        verts = np.array([[5, 5], [15, 5], [15, 15], [5, 15]], dtype=float)
        area = CosineWaveArea("cos", verts, config={"eta0": 2.0})
        mask = area.compute_mask(nx, ny)
        wave = area.compute_wave(mask)
        self.assertEqual(wave.shape, mask.shape)
        # Values are NaN outside
        self.assertTrue(np.isnan(wave[0, 0]))
        # Max equals eta0 (cos(0) -> 1)
        self.assertAlmostEqual(np.nanmax(wave), 2.0, places=6)


if __name__ == '__main__':
    unittest.main()

