import unittest
from diser.coords import xy_to_rc, rc_to_xy, Size


class TestCoords(unittest.TestCase):
    def test_xy_rc_roundtrip(self):
        for x, y in [(0, 0), (5, 2), (19, 33)]:
            r, c = xy_to_rc(x, y)
            xx, yy = rc_to_xy(r, c)
            self.assertEqual((xx, yy), (x, y))

    def test_size_dataclass(self):
        s = Size(10, 20)
        self.assertEqual((s.x, s.y), (10, 20))


if __name__ == '__main__':
    unittest.main()

