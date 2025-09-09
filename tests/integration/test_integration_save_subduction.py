import os
import tempfile
import unittest
import numpy as np
import builtins

from save_subduction import save_functions


class TestIntegrationSaveSubduction(unittest.TestCase):
    def test_save_functions_creates_wave_file(self):
        # Use real config and areas; write to a temporary file
        self.assertTrue(os.path.exists('data/config.json'))
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, 'functions.wave')
            # Suppress printing to avoid Windows console encoding issues
            real_print = builtins.print
            try:
                builtins.print = lambda *a, **k: None
                save_functions('data/config.json', out)
            finally:
                builtins.print = real_print
            self.assertTrue(os.path.exists(out))
            arr = np.loadtxt(out)
            # Dimensions should match config size
            import json
            cfg = json.load(open('data/config.json', 'r', encoding='utf-8'))
            self.assertEqual(arr.shape, (cfg['size']['y'], cfg['size']['x']))


if __name__ == '__main__':
    unittest.main()
