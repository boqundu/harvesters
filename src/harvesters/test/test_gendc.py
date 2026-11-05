#!/usr/bin/env python3
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
import struct
import unittest
from unittest.mock import MagicMock

# Related third party imports
import numpy

# Local application/library specific imports
from harvesters._private.core.gendc import (
    GenDCParseError,
    parse,
)
from harvesters.core import (
    Buffer,
    PayloadGenDC,
    PayloadImage,
    PayloadUnknown,
    _PAYLOAD_TYPE_GENDC,
)


# GenDC v1.2 predefined values (see GenDC.h):
GDC_SIGNATURE = 0x43444E47  # "GNDC"
GDC_CONTAINER_HEADER = 0x1000
GDC_COMPONENT_HEADER = 0x2000
GDC_PART_ARRAY_2D = 0x4200
GDC_PART_GENICAM_CHUNK = 0x4000
GDC_INTENSITY = 0x0001
GDC_METADATA = 0x8001

# PFNC values:
MONO8 = 0x01080001
RGB8 = 0x02180014

# Struct layouts; every offset is relative to the container start and all
# integers are little-endian (GenDC v1.2, pack(1)):
_CONTAINER_HEADER = '<I3BBHHIQHHIQqII'  # + 'q' * component_count
_COMPONENT_HEADER = '<HHIHHHHIIqQIHH'  # + 'q' * part_count
_PART_HEADER_2D = '<HHIIHHqQqIIHHI'
_PART_HEADER_1D = '<HHIIHHqQqQHHIQ'

_CONTAINER_HEADER_SIZE = 56
_COMPONENT_HEADER_SIZE = 48


def _pack_container_header(
        component_offsets, *, signature=GDC_SIGNATURE,
        version=(1, 2, 0), header_type=GDC_CONTAINER_HEADER, flags=0,
        header_size=None, container_id=0, variable_fields=0, data_size=None,
        data_offset=None, descriptor_size=None, component_count=None):
    count = len(component_offsets)
    if header_size is None:
        header_size = _CONTAINER_HEADER_SIZE + 8 * count
    if component_count is None:
        component_count = count
    return struct.pack(
        _CONTAINER_HEADER + 'q' * count,
        signature, *version, 0, header_type, flags, header_size,
        container_id, variable_fields, 0, 0,
        0 if data_size is None else data_size,
        0 if data_offset is None else data_offset,
        0 if descriptor_size is None else descriptor_size,
        component_count,
        *component_offsets)


def _pack_component_header(
        part_offsets, *, header_type=GDC_COMPONENT_HEADER, flags=0,
        header_size=None, group_id=0, source_id=0, region_id=0,
        region_offset_x=0, region_offset_y=0, timestamp=0,
        type_id=GDC_INTENSITY, fmt=0, part_count=None):
    count = len(part_offsets)
    if header_size is None:
        header_size = _COMPONENT_HEADER_SIZE + 8 * count
    if part_count is None:
        part_count = count
    return struct.pack(
        _COMPONENT_HEADER + 'q' * count,
        header_type, flags, header_size, 0, group_id, source_id, region_id,
        region_offset_x, region_offset_y, timestamp, type_id, fmt, 0,
        part_count,
        *part_offsets)


def _pack_part_2d(
        *, part_type=GDC_PART_ARRAY_2D, flags=0, header_size=56, fmt=MONO8,
        flow_id=0, data_size=0, data_offset=0, size_x=0, size_y=0,
        padding_x=0, padding_y=0):
    return struct.pack(
        _PART_HEADER_2D, part_type, flags, header_size, fmt, 0, flow_id, 0,
        data_size, data_offset, size_x, size_y, padding_x, padding_y, 0)


def _pack_part_1d(
        *, part_type=GDC_PART_GENICAM_CHUNK, flags=0, header_size=64,
        fmt=MONO8, flow_id=0, data_size=0, data_offset=0, size_x=0,
        padding_x=0, layout_id=0):
    return struct.pack(
        _PART_HEADER_1D, part_type, flags, header_size, fmt, 0, flow_id, 0,
        data_size, data_offset, size_x, padding_x, 0, 0, layout_id)


def build_container(components, *, version=(1, 2, 0), signature=GDC_SIGNATURE):
    """
    Builds a synthetic GenDC v1.2 container from a list of components.

    Every component is a dict with optional keys ``type_id`` (default
    :const:`GDC_INTENSITY`), ``source_id``, ``timestamp`` and ``parts``;
    every part is a dict carrying the part header fields plus its payload
    ``data`` (bytes, already padded by the caller when relevant).
    """
    # First pass: determine every header size so that the offsets can be
    # calculated.
    part_header_size = {
        '2d': 56,
        '1d': 64,
    }
    cursor = _CONTAINER_HEADER_SIZE + 8 * len(components)
    component_offsets = []
    component_blocks = []
    data_blocks = []
    for component in components:
        parts = component.get('parts', [])
        component_offsets.append(cursor)
        cursor += _COMPONENT_HEADER_SIZE + 8 * len(parts)
        component_blocks.append(
            (component, [_COMPONENT_HEADER_SIZE + 8 * len(parts)]))
        for part in parts:
            kind = component_blocks[-1][0]  # noqa: F841 (readability)
            size = part_header_size[part.get('kind', '2d')]
            part['_header_size'] = size
            part['_offset'] = cursor
            cursor += size

    # The data region starts right after the descriptor unless a part
    # explicitly dictates its own data offset.
    data_offset_base = cursor
    cursor_data = data_offset_base
    for component in components:
        for part in component.get('parts', []):
            data = part.get('data', b'')
            if 'data_offset' not in part:
                part['data_offset'] = cursor_data
                cursor_data += len(data)
            if 'data_size' not in part:
                part['data_size'] = len(data)

    descriptor = bytearray()
    descriptor += _pack_container_header(
        component_offsets, version=version, signature=signature,
        descriptor_size=data_offset_base,
        data_size=cursor_data - data_offset_base,
        data_offset=data_offset_base)
    for i, component in enumerate(components):
        parts = component.get('parts', [])
        descriptor += _pack_component_header(
            [p['_offset'] for p in parts],
            source_id=component.get('source_id', 0),
            timestamp=component.get('timestamp', 0),
            type_id=component.get('type_id', GDC_INTENSITY))
        for part in parts:
            if part.get('kind', '2d') == '2d':
                descriptor += _pack_part_2d(
                    fmt=part.get('fmt', MONO8),
                    data_size=part['data_size'],
                    data_offset=part['data_offset'],
                    size_x=part.get('size_x', 0),
                    size_y=part.get('size_y', 0),
                    padding_x=part.get('padding_x', 0),
                    padding_y=part.get('padding_y', 0))
            else:
                descriptor += _pack_part_1d(
                    part_type=part.get('part_type', GDC_PART_GENICAM_CHUNK),
                    fmt=part.get('fmt', MONO8),
                    data_size=part['data_size'],
                    data_offset=part['data_offset'],
                    size_x=part.get('size_x', 0),
                    padding_x=part.get('padding_x', 0))

    for component in components:
        for part in component.get('parts', []):
            descriptor += part.get('data', b'')

    return bytes(descriptor)


def _padded_rows(pixels, width, height, pixel_bytes, padding_x, padding_y=0):
    """Packs pixel rows inserting ``padding_x`` bytes after every line."""
    rows = []
    for y in range(height):
        start = y * width * pixel_bytes
        rows.append(pixels[start:start + width * pixel_bytes])
        rows.append(b'\xAA' * padding_x)
    rows.append(b'\x55' * padding_y)
    return b''.join(rows)


class _FakeBufferModule:
    """
    Mimics the attributes of a GenTL Buffer module that the payload
    building code touches.
    """
    def __init__(self, raw_buffer, *, payload_type=_PAYLOAD_TYPE_GENDC,
                 pixel_format=MONO8, width=4, height=3):
        self.parent = None
        self._raw_buffer = raw_buffer
        self.payload_type = payload_type
        self.pixel_format = pixel_format
        self.width = width
        self.height = height
        self.padding_x = 0
        self.padding_y = 0
        self.offset_x = 0
        self.offset_y = 0
        self.delivered_image_height = height

    @property
    def raw_buffer(self):
        return self._raw_buffer


class TestGenDCParser(unittest.TestCase):
    def _mono8_container(self):
        pixels = bytes(range(12))  # 4x3 Mono8
        return build_container(
            [{'type_id': GDC_INTENSITY, 'source_id': 7, 'timestamp': 12345,
              'parts': [{'kind': '2d', 'fmt': MONO8, 'size_x': 4,
                         'size_y': 3, 'data': pixels}]}]), pixels

    def test_mono8_container(self):
        raw, pixels = self._mono8_container()
        container = parse(raw)
        self.assertEqual(GDC_SIGNATURE, container.signature)
        self.assertEqual(1, container.version_major)
        self.assertEqual(2, container.version_minor)
        self.assertEqual(1, container.component_count)
        self.assertEqual(1, len(container.components))
        component = container.components[0]
        self.assertEqual(GDC_INTENSITY, component.type_id)
        self.assertEqual(7, component.source_id)
        self.assertEqual(12345, component.timestamp)
        self.assertEqual(1, component.part_count)
        part = component.parts[0]
        self.assertTrue(part.is_2d())
        self.assertTrue(part.is_array2d())
        self.assertEqual(MONO8, part.format)
        self.assertEqual(4, part.width)
        self.assertEqual(3, part.height)
        self.assertEqual(0, part.padding_x)
        self.assertEqual(len(pixels), part.data_size)
        self.assertEqual(raw[part.data_offset:part.data_offset +
                             part.data_size], pixels)

    def test_rgb8_container(self):
        pixels = bytes(range(18))  # 3x2 RGB8
        raw = build_container(
            [{'parts': [{'kind': '2d', 'fmt': RGB8, 'size_x': 3,
                         'size_y': 2, 'data': pixels}]}])
        container = parse(raw)
        part = container.components[0].parts[0]
        self.assertEqual(RGB8, part.format)
        self.assertEqual(3, part.width)
        self.assertEqual(2, part.height)
        self.assertEqual(18, part.data_size)

    def test_container_with_chunk_part(self):
        pixels = bytes(range(6))  # 2x3 Mono8
        raw = build_container([
            {'type_id': GDC_INTENSITY,
             'parts': [{'kind': '2d', 'fmt': MONO8, 'size_x': 2,
                        'size_y': 3, 'data': pixels}]},
            {'type_id': GDC_METADATA,
             'parts': [{'kind': '1d', 'fmt': MONO8, 'size_x': 8,
                        'data': b'\x01\x02\x03\x04\x05\x06\x07\x08'}]},
        ])
        container = parse(raw)
        self.assertEqual(2, container.component_count)
        first = container.components[0]
        second = container.components[1]
        self.assertEqual(GDC_INTENSITY, first.type_id)
        self.assertEqual(GDC_METADATA, second.type_id)
        self.assertTrue(first.parts[0].is_array2d())
        chunk = second.parts[0]
        self.assertFalse(chunk.is_2d())
        # A GenICam chunk part belongs to the metadata part family:
        self.assertFalse(chunk.is_1d())
        self.assertEqual(0x4000, chunk.part_type & 0x4F00)
        self.assertFalse(chunk.is_array2d())
        # The first image component selection rule:
        self.assertIs(first, container.first_image_component())

    def test_first_image_component_skips_chunk_only(self):
        raw = build_container([
            {'type_id': GDC_METADATA,
             'parts': [{'kind': '1d', 'data': b'\x00' * 4}]},
            {'type_id': GDC_INTENSITY,
             'parts': [{'kind': '2d', 'fmt': MONO8, 'size_x': 4,
                        'size_y': 3, 'data': bytes(12)}]},
        ])
        container = parse(raw)
        self.assertIs(container.components[1],
                      container.first_image_component())

    def test_padding_fields_parsed(self):
        pixels = bytes(range(12))
        padded = _padded_rows(pixels, 4, 3, 1, padding_x=8)
        raw = build_container(
            [{'parts': [{'kind': '2d', 'fmt': MONO8, 'size_x': 4,
                         'size_y': 3, 'padding_x': 8, 'data': padded}]}])
        part = parse(raw).components[0].parts[0]
        self.assertEqual(8, part.padding_x)
        self.assertEqual(0, part.padding_y)
        self.assertEqual(len(padded), part.data_size)

    def test_corrupt_containers_rejected(self):
        raw, _ = self._mono8_container()

        cases = {}
        # Truncated buffer:
        cases['too_small'] = raw[:32]
        # Bad signature:
        cases['bad_signature'] = b'XXXX' + raw[4:]
        # Unsupported major version:
        cases['bad_version'] = \
            raw[:4] + bytes((2, 2, 0)) + raw[7:]
        # Wrong container header type:
        cases['bad_header_type'] = raw[:8] + struct.pack('<H', 0x2000) + \
            raw[10:]
        # Descriptor larger than the buffer:
        cases['descriptor_too_large'] = raw[:48] + struct.pack(
            '<I', len(raw) + 100) + raw[52:]
        # No component:
        cases['zero_components'] = raw[:52] + struct.pack('<I', 0) + raw[56:]
        # Component offset outside the descriptor:
        cases['component_offset_out'] = raw[:56] + struct.pack(
            '<q', 8) + raw[64:]

        for name, blob in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(GenDCParseError):
                    parse(blob)

    def test_corrupt_component_rejected(self):
        pixels = bytes(12)
        raw = bytearray(
            build_container(
                [{'parts': [{'kind': '2d', 'fmt': MONO8, 'size_x': 4,
                             'size_y': 3, 'data': pixels}]}]))
        # Wrong component header type (expected 0x2000):
        raw[64:66] = struct.pack('<H', 0x1000)
        with self.assertRaises(GenDCParseError):
            parse(bytes(raw))

    def test_corrupt_part_rejected(self):
        pixels = bytes(12)
        raw = bytearray(
            build_container(
                [{'parts': [{'kind': '2d', 'fmt': MONO8, 'size_x': 4,
                             'size_y': 3, 'data': pixels}]}]))
        # Wrong part family (0x2000 does not belong to 0x4000-0x4FFF):
        raw[120:122] = struct.pack('<H', 0x2000)
        with self.assertRaises(GenDCParseError):
            parse(bytes(raw))

        # Part data outside the buffer:
        raw = bytearray(
            build_container(
                [{'parts': [{'kind': '2d', 'fmt': MONO8, 'size_x': 4,
                             'size_y': 3, 'data': pixels}]}]))
        raw[120 + 24:120 + 32] = struct.pack('<Q', 1 << 20)  # DataSize
        with self.assertRaises(GenDCParseError):
            parse(bytes(raw))

    def test_data_offset_before_descriptor_rejected(self):
        pixels = bytes(12)
        raw = bytearray(
            build_container(
                [{'parts': [{'kind': '2d', 'fmt': MONO8, 'size_x': 4,
                             'size_y': 3, 'data': pixels}]}]))
        # Container DataOffset pointing into the descriptor:
        raw[40:48] = struct.pack('<q', 4)
        with self.assertRaises(GenDCParseError):
            parse(bytes(raw))


class TestPayloadGenDC(unittest.TestCase):
    def _node_map(self):
        return MagicMock()

    def test_mono8_payload(self):
        pixels = bytes(range(12))
        raw = build_container(
            [{'parts': [{'kind': '2d', 'fmt': MONO8, 'size_x': 4,
                         'size_y': 3, 'data': pixels}]}])
        module = _FakeBufferModule(raw)
        payload = PayloadGenDC(buffer=module, node_map=self._node_map())
        self.assertEqual(1, len(payload.components))
        component = payload.components[0]
        self.assertIsNotNone(component)
        self.assertEqual(4, component.width)
        self.assertEqual(3, component.height)
        self.assertEqual(MONO8, component.data_format_value)
        self.assertEqual('Mono8', component.data_format)
        self.assertTrue(
            numpy.array_equal(numpy.frombuffer(pixels, dtype=numpy.uint8),
                              component.data))

    def test_rgb8_payload(self):
        pixels = bytes(range(18))
        raw = build_container(
            [{'parts': [{'kind': '2d', 'fmt': RGB8, 'size_x': 3,
                         'size_y': 2, 'data': pixels}]}])
        module = _FakeBufferModule(raw, pixel_format=RGB8, width=3, height=2)
        payload = PayloadGenDC(buffer=module, node_map=self._node_map())
        component = payload.components[0]
        self.assertEqual(3, component.num_components_per_pixel)
        self.assertEqual(18, component.data.size)
        self.assertTrue(
            numpy.array_equal(numpy.frombuffer(pixels, dtype=numpy.uint8),
                              component.data))

    def test_chunk_part_not_displayable(self):
        pixels = bytes(range(6))
        raw = build_container([
            {'type_id': GDC_INTENSITY,
             'parts': [{'kind': '2d', 'fmt': MONO8, 'size_x': 2,
                        'size_y': 3, 'data': pixels}]},
            {'type_id': GDC_METADATA,
             'parts': [{'kind': '1d', 'data': b'\x01' * 8}]},
        ])
        module = _FakeBufferModule(raw)
        payload = PayloadGenDC(buffer=module, node_map=self._node_map())
        # Only the 2D part yields a displayable component:
        self.assertEqual(1, len(payload.components))
        self.assertEqual(2, payload.components[0].width)

    def test_padding_stripped(self):
        pixels = bytes(range(12))
        padded = _padded_rows(pixels, 4, 3, 1, padding_x=8)
        raw = build_container(
            [{'parts': [{'kind': '2d', 'fmt': MONO8, 'size_x': 4,
                         'size_y': 3, 'padding_x': 8, 'data': padded}]}])
        module = _FakeBufferModule(raw)
        payload = PayloadGenDC(buffer=module, node_map=self._node_map())
        component = payload.components[0]
        # The padding bytes must not leak into the component data:
        self.assertEqual(12, component.data.size)
        expected = numpy.frombuffer(pixels, dtype=numpy.uint8)
        self.assertTrue(numpy.array_equal(expected, component.data))

    def test_padding_y_stripped(self):
        pixels = bytes(range(12))
        padded = _padded_rows(pixels, 4, 3, 1, padding_x=0, padding_y=16)
        raw = build_container(
            [{'parts': [{'kind': '2d', 'fmt': MONO8, 'size_x': 4,
                         'size_y': 3, 'padding_y': 16, 'data': padded}]}])
        module = _FakeBufferModule(raw)
        payload = PayloadGenDC(buffer=module, node_map=self._node_map())
        self.assertEqual(12, payload.components[0].data.size)

    def test_no_displayable_component_yields_empty_list(self):
        raw = build_container(
            [{'type_id': GDC_METADATA,
              'parts': [{'kind': '1d', 'data': b'\x01' * 8}]}])
        module = _FakeBufferModule(raw)
        payload = PayloadGenDC(buffer=module, node_map=self._node_map())
        self.assertEqual([], payload.components)

    def test_no_none_components(self):
        # A container whose only 2D part uses an unknown pixel format must
        # not inject None into the component list:
        pixels = bytes(12)
        raw = build_container(
            [{'parts': [{'kind': '2d', 'fmt': 0x0E000000, 'size_x': 4,
                         'size_y': 3, 'data': pixels}]},
             {'parts': [{'kind': '2d', 'fmt': MONO8, 'size_x': 4,
                         'size_y': 3, 'data': pixels}]}])
        module = _FakeBufferModule(raw)
        payload = PayloadGenDC(buffer=module, node_map=self._node_map())
        self.assertEqual(1, len(payload.components))
        self.assertIsNotNone(payload.components[0])

    def test_gui_reshape(self):
        # Simulates canvas.py: component.data.reshape(height, width):
        pixels = bytes(range(12))
        raw = build_container(
            [{'parts': [{'kind': '2d', 'fmt': MONO8, 'size_x': 4,
                         'size_y': 3, 'data': pixels}]}])
        module = _FakeBufferModule(raw)
        payload = PayloadGenDC(buffer=module, node_map=self._node_map())
        component = payload.components[0]
        content = component.data.reshape(component.height, component.width)
        self.assertTrue(
            numpy.array_equal(
                numpy.frombuffer(pixels, dtype=numpy.uint8).reshape(3, 4),
                content))

    def test_gui_reshape_padded(self):
        pixels = bytes(range(12))
        padded = _padded_rows(pixels, 4, 3, 1, padding_x=8)
        raw = build_container(
            [{'parts': [{'kind': '2d', 'fmt': MONO8, 'size_x': 4,
                         'size_y': 3, 'padding_x': 8, 'data': padded}]}])
        module = _FakeBufferModule(raw)
        payload = PayloadGenDC(buffer=module, node_map=self._node_map())
        component = payload.components[0]
        content = component.data.reshape(component.height, component.width)
        self.assertTrue(
            numpy.array_equal(
                numpy.frombuffer(pixels, dtype=numpy.uint8).reshape(3, 4),
                content))


class TestPayloadDispatch(unittest.TestCase):
    def _node_map(self):
        return MagicMock()

    def test_dispatch_by_payload_type(self):
        pixels = bytes(range(12))
        raw = build_container(
            [{'parts': [{'kind': '2d', 'fmt': MONO8, 'size_x': 4,
                         'size_y': 3, 'data': pixels}]}])
        module = _FakeBufferModule(raw, payload_type=_PAYLOAD_TYPE_GENDC)
        payload = Buffer._build_payload(buffer=module,
                                        node_map=self._node_map())
        self.assertIsInstance(payload, PayloadGenDC)
        component = payload.components[0]
        self.assertEqual(4, component.width)
        self.assertEqual(3, component.height)
        self.assertEqual(MONO8, component.data_format_value)
        self.assertEqual('Mono8', component.data_format)
        self.assertTrue(
            numpy.array_equal(numpy.frombuffer(pixels, dtype=numpy.uint8),
                              component.data))

    def test_dispatch_by_signature_sniffing(self):
        # A producer that reports an unhandled payload type but delivers
        # GenDC containers:
        pixels = bytes(range(12))
        raw = build_container(
            [{'parts': [{'kind': '2d', 'fmt': MONO8, 'size_x': 4,
                         'size_y': 3, 'data': pixels}]}])
        module = _FakeBufferModule(raw, payload_type=9)
        payload = Buffer._build_payload(buffer=module,
                                        node_map=self._node_map())
        self.assertIsInstance(payload, PayloadGenDC)

    def test_unknown_type_without_signature_takes_remedy(self):
        # payload type 9 without the GNDC magic must take the legacy
        # "assume image" remedy exactly as before:
        pixels = bytes(range(12))
        module = _FakeBufferModule(
            pixels, payload_type=9, pixel_format=MONO8, width=4, height=3)
        payload = Buffer._build_payload(buffer=module,
                                        node_map=self._node_map())
        self.assertIsInstance(payload, PayloadImage)
        self.assertTrue(
            numpy.array_equal(numpy.frombuffer(pixels, dtype=numpy.uint8),
                              payload.components[0].data))

    def test_gendc_type_with_corrupt_container_yields_unknown(self):
        # A GenDC-typed buffer that fails to parse must not raise into
        # the acquisition loop; the payload is exposed as unknown with
        # no components so that no client can mistake the container
        # bytes for an image:
        module = _FakeBufferModule(
            b'XXXX' + bytes(48), payload_type=_PAYLOAD_TYPE_GENDC,
            pixel_format=MONO8, width=4, height=3)
        payload = Buffer._build_payload(buffer=module,
                                        node_map=self._node_map())
        self.assertIsInstance(payload, PayloadUnknown)
        self.assertEqual([], payload.components)

    def test_payload_type_reports_effective_gendc(self):
        # A producer that announces a non-GENDC payload type while
        # delivering GenDC containers: the wrapper Buffer must report
        # the effective type so that clients gate on what the payload
        # actually is:
        pixels = bytes(range(12))
        raw = build_container(
            [{'parts': [{'kind': '2d', 'fmt': MONO8, 'size_x': 4,
                         'size_y': 3, 'data': pixels}]}])
        module = _FakeBufferModule(raw, payload_type=9)
        buffer = Buffer(module=module, node_map=self._node_map())
        self.assertIsInstance(buffer.payload, PayloadGenDC)
        self.assertEqual(_PAYLOAD_TYPE_GENDC, buffer.payload_type)
        self.assertEqual(9, module.payload_type)

    def test_represent_pixel_location_with_padding(self):
        # The padding is stripped from the data, so the pixel location
        # representation must be computed from the packed shape:
        pixels = bytes(range(12))
        padded = _padded_rows(pixels, 4, 3, 1, padding_x=8)
        raw = build_container(
            [{'parts': [{'kind': '2d', 'fmt': MONO8, 'size_x': 4,
                         'size_y': 3, 'padding_x': 8, 'data': padded}]}])
        module = _FakeBufferModule(raw)
        payload = Buffer._build_payload(buffer=module,
                                        node_map=self._node_map())
        component = payload.components[0]
        locations = component.represent_pixel_location()
        self.assertEqual((3, 4), locations.shape)
        self.assertTrue(
            numpy.array_equal(
                numpy.frombuffer(pixels, dtype=numpy.uint8).reshape(3, 4),
                locations))

    def test_image_payload_unchanged(self):
        # Regression: the conventional IMAGE path must behave as before:
        pixels = bytes(range(12))
        module = _FakeBufferModule(pixels, payload_type=1)
        payload = Buffer._build_payload(buffer=module,
                                        node_map=self._node_map())
        self.assertIsInstance(payload, PayloadImage)
        component = payload.components[0]
        self.assertEqual(4, component.width)
        self.assertEqual(3, component.height)
        self.assertEqual('Mono8', component.data_format)
        self.assertTrue(
            numpy.array_equal(numpy.frombuffer(pixels, dtype=numpy.uint8),
                              component.data))


if __name__ == '__main__':
    unittest.main()
