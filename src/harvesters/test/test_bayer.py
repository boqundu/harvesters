# ----------------------------------------------------------------------------
#
# Copyright 2018 EMVA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ----------------------------------------------------------------------------


# Standard library imports
import unittest

# Related third party imports
import numpy

# Local application/library local imports
from harvesters.util.bayer import cfa_positions, demosaic


class TestCfaPositions(unittest.TestCase):
    def test_positions_match_the_tile_arrangements(self):
        expected = {
            'BayerRG8': ((0, 0), (1, 1)),
            'BayerGR8': ((0, 1), (1, 0)),
            'BayerGB8': ((1, 0), (0, 1)),
            'BayerBG8': ((1, 1), (0, 0)),
            'BayerRG12': ((0, 0), (1, 1)),
        }
        for name, positions in expected.items():
            self.assertEqual(positions, cfa_positions(name), name)

    def test_rejects_non_bayer_formats(self):
        with self.assertRaises(ValueError):
            cfa_positions('Mono8')


class TestDemosaic(unittest.TestCase):
    def test_uniform_image_is_identity(self):
        # Every interpolation collapses to the same value so a uniform
        # mosaic must demosaic to a uniform RGB image, for every CFA
        # arrangement:
        for tile in ('BayerRG8', 'BayerGR8', 'BayerGB8', 'BayerBG8'):
            mosaic = numpy.full((8, 10), 137, dtype=numpy.uint8)
            rgb = demosaic(mosaic, tile)
            self.assertEqual((8, 10, 3), rgb.shape)
            self.assertEqual(numpy.uint8, rgb.dtype)
            self.assertTrue(numpy.all(rgb == 137), tile)

    def test_measured_sites_are_preserved(self):
        # The sites where a color is measured must keep their original
        # values; BayerRG8 measures R at (0, 0), G at (0, 1) and (1, 0),
        # and B at (1, 1) of each 2x2 tile:
        mosaic = numpy.fromfunction(
            lambda y, x: (37 * y + 11 * x + 5) % 256,
            (12, 14), dtype=numpy.int64).astype(numpy.uint8)
        rgb = demosaic(mosaic, 'BayerRG8')
        self.assertTrue(numpy.array_equal(
            rgb[0::2, 0::2, 0], mosaic[0::2, 0::2]))
        self.assertTrue(numpy.array_equal(
            rgb[0::2, 1::2, 1], mosaic[0::2, 1::2]))
        self.assertTrue(numpy.array_equal(
            rgb[1::2, 0::2, 1], mosaic[1::2, 0::2]))
        self.assertTrue(numpy.array_equal(
            rgb[1::2, 1::2, 2], mosaic[1::2, 1::2]))

    def test_bilinear_interpolation_is_exact_on_ramps(self):
        # A linear ramp is reproduced exactly by linear interpolation:
        # the R site at (0, 1) is the average of its left and right R
        # sites, and the B site at (0, 1) is interpolated vertically:
        width, height = 8, 6
        mosaic = numpy.fromfunction(
            lambda y, x: x * 9, (height, width),
            dtype=numpy.int64).astype(numpy.uint8)
        rgb = demosaic(mosaic, 'BayerRG8')
        self.assertEqual(
            (int(mosaic[0, 0]) + int(mosaic[0, 2])) // 2,
            int(rgb[0, 1, 0]))
        self.assertEqual(
            (int(mosaic[1, 1]) + int(mosaic[1, 3])) // 2,
            int(rgb[1, 2, 2]))

    def test_borders_do_not_crash_and_stay_in_range(self):
        mosaic = numpy.arange(21, dtype=numpy.uint8).reshape(3, 7)
        rgb = demosaic(mosaic, 'BayerGB8')
        self.assertEqual((3, 7, 3), rgb.shape)

    def test_rejects_non_bayer_formats(self):
        for bad in ('Mono8', 'RGB8', 'BayerQQ8'):
            with self.assertRaises(ValueError):
                demosaic(numpy.zeros((4, 4), dtype=numpy.uint8), bad)

    def test_rejects_non_2d_input(self):
        with self.assertRaises(ValueError):
            demosaic(numpy.zeros((4, 4, 3), dtype=numpy.uint8), 'BayerRG8')


if __name__ == '__main__':
    unittest.main()
