import os
import tempfile
import unittest

from loader import load_bath_grid


class TestLoaderMOST(unittest.TestCase):
    def test_load_bath_grid_parses_header_and_shape(self):
        # Create a tiny MOST-like file: nx ny, then nx x-coords, ny y-coords, then grid values
        nx, ny = 4, 3
        xs = ' '.join(str(i) for i in range(nx))
        ys = ' '.join(str(j) for j in range(ny))
        # values from 0..(nx*ny-1)
        vals = ' '.join(str(v) for v in range(nx * ny))
        content = f"{nx} {ny}\n{xs}\n{ys}\n{vals}\n"

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'grid.most')
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)

            grid = load_bath_grid(path)
            self.assertEqual(grid.shape, (ny, nx))
            self.assertEqual(int(grid[0, 0]), 0)
            self.assertEqual(int(grid[-1, -1]), nx * ny - 1)


if __name__ == '__main__':
    unittest.main()

