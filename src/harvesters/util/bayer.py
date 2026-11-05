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

# Related third party imports
import numpy

# Local application/library specific imports


_BAYER_TILES = {
    'RG': 'RGGB',
    'GR': 'GRBG',
    'GB': 'GBRG',
    'BG': 'BGGR',
}


def cfa_positions(data_format):
    """
    Returns the sites of the red and the blue pixels within the 2x2 CFA
    tile of the given Bayer pixel format name as ``((ry, rx), (by, bx))``,
    where the origin is the top-left texel and ``y`` addresses the rows.

    :param data_format: The PFNC symbolic name, such as ``BayerRG8``.

    :return: A tuple of two ``(y, x)`` tuples, red first, blue second.

    :raises ValueError: If ``data_format`` is not a Bayer format name.
    """
    name = str(data_format)
    prefix = name.find('Bayer')
    arrangement = ''
    if prefix >= 0:
        arrangement = name[prefix + 5:prefix + 7].upper()
    if arrangement not in _BAYER_TILES:
        raise ValueError(
            'not a Bayer pixel format: {}'.format(data_format))
    tile_str = _BAYER_TILES[arrangement]
    red = tile_str.index('R')
    blue = tile_str.index('B')
    # The tile string is row-major: index i sits at (i // 2, i % 2):
    return (red // 2, red % 2), (blue // 2, blue % 2)


def demosaic(mosaic, data_format):
    """
    Returns an RGB image as a NumPy array that has been demosaiced from
    the given Bayer CFA image ``mosaic`` using bilinear interpolation.

    The mosaic site of each color is derived from the Bayer arrangement
    that the given PFNC pixel format name, such as ``BayerRG8``, implies
    (``RG`` means an RGGB tile whose R site sits at the origin). The
    measured sites keep their original values and the missing sites are
    interpolated from their same-color neighbors; the image borders are
    handled with edge replication.

    Note that this function is intended for visualization; it returns an
    8-bit image and is not a color-coefficient-corrected conversion.

    :param mosaic: A 2D NumPy array that holds the Bayer image.
    :param data_format: The PFNC symbolic name of the pixel format.

    :return: A ``(height, width, 3)`` uint8 RGB array.

    :raises ValueError: If ``data_format`` is not a Bayer format name.
    """
    name = str(data_format)
    prefix = name.find('Bayer')
    arrangement = ''
    if prefix >= 0:
        arrangement = name[prefix + 5:prefix + 7].upper()
    if arrangement not in _BAYER_TILES:
        raise ValueError(
            'not a Bayer pixel format: {}'.format(data_format))
    tile_str = _BAYER_TILES[arrangement]

    # The 2x2 CFA tile, for example RGGB: R at (0, 0), G at (0, 1) and
    # (1, 0), and B at (1, 1):
    tile = {(0, 0): tile_str[0], (0, 1): tile_str[1],
            (1, 0): tile_str[2], (1, 1): tile_str[3]}

    mosaic = numpy.asarray(mosaic)
    if mosaic.ndim != 2:
        raise ValueError(
            'the mosaic must be a 2D array: {}'.format(mosaic.shape))
    height, width = mosaic.shape

    m = mosaic.astype(numpy.uint16)
    p = numpy.pad(m, 1, mode='edge')

    up, down = p[0:-2, 1:-1], p[2:, 1:-1]
    left, right = p[1:-1, 0:-2], p[1:-1, 2:]
    h_average = (left + right) >> 1
    v_average = (up + down) >> 1
    d_average = (p[0:-2, 0:-2] + p[0:-2, 2:] +
                 p[2:, 0:-2] + p[2:, 2:]) >> 2
    n4_average = (up + down + left + right) >> 2

    rgb = numpy.empty((height, width, 3), dtype=numpy.uint16)
    for c, color in enumerate('RGB'):
        for py in (0, 1):
            for px in (0, 1):
                if tile[(py, px)] == color:
                    source = m
                elif color == 'G':
                    # At an R or B site the green neighbors sit up,
                    # down, left, and right:
                    source = n4_average
                elif tile[(py, 1 - px)] == color:
                    # The same-color sites sit left and right:
                    source = h_average
                elif tile[(1 - py, px)] == color:
                    # The same-color sites sit up and down:
                    source = v_average
                else:
                    # The same-color sites sit on the diagonals:
                    source = d_average
                y, x = slice(py, height, 2), slice(px, width, 2)
                rgb[y, x, c] = source[y, x]

    return rgb.astype(numpy.uint8)
