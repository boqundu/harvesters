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
from types import SimpleNamespace

# Related third party imports
from unittest.mock import MagicMock

from genicam.gentl import NotAvailableException

# Local application/library specific imports
from harvesters.core import ImageAcquirer


_MONO8 = 0x01080001


def _make_acquirer(node_map):
    acquirer = ImageAcquirer.__new__(ImageAcquirer)
    device = MagicMock()
    device.node_map = node_map
    acquirer._remote_device = device
    return acquirer


class TestDetermineBufferSize(unittest.TestCase):
    def test_data_stream_size_used_when_available(self):
        ds = MagicMock()
        ds.defines_payload_size.return_value = True
        ds.payload_size = 1000
        acquirer = _make_acquirer(node_map=MagicMock())
        self.assertEqual(1000, acquirer._determine_buffer_size(ds))

    def test_falls_back_when_inquiry_fails(self):
        # A producer that declares a data stream payload size but fails
        # the inquiry (as reported with a GEV 3.0 producer):
        ds = MagicMock()
        ds.defines_payload_size.return_value = True
        ds.payload_size = MagicMock()
        type(ds).payload_size = property(
            lambda self: (_ for _ in ()).throw(
                NotAvailableException('not available')))

        node_map = MagicMock()
        node_map.PayloadSize.value = 4321
        acquirer = _make_acquirer(node_map=node_map)
        self.assertEqual(4321, acquirer._determine_buffer_size(ds))

    def test_falls_back_to_calculation(self):
        ds = MagicMock()
        ds.defines_payload_size.return_value = False

        node_map = SimpleNamespace(
            Width=SimpleNamespace(value=4),
            Height=SimpleNamespace(value=3),
            PixelFormat=SimpleNamespace(
                get_int_value=lambda: _MONO8),
        )
        acquirer = _make_acquirer(node_map=node_map)
        self.assertEqual(12, acquirer._determine_buffer_size(ds))

    def test_raises_when_undeterminable(self):
        ds = MagicMock()
        ds.defines_payload_size.return_value = False

        node_map = SimpleNamespace(
            Width=SimpleNamespace(value=0),
            Height=SimpleNamespace(value=3),
            PixelFormat=SimpleNamespace(
                get_int_value=lambda: _MONO8),
        )
        acquirer = _make_acquirer(node_map=node_map)
        with self.assertRaises(Exception):
            acquirer._determine_buffer_size(ds)


if __name__ == '__main__':
    unittest.main()
