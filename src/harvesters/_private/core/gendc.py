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
from __future__ import annotations
import struct
from typing import List, Optional


# This module is a Python port of the reference C++ parser GenDCParser.h/
# GenDCParser.cpp. It parses and validates a GenDC v1.2 container that is
# laid out in a buffer; see GenDC.h for the authoritative structure
# definitions. All offsets are relative to the container start and every
# integer is little-endian (the structures are packed).

# GenDC standard values:
SIGNATURE = 0x43444E47  # "GNDC"
VERSION_MAJOR = 0x01

# Header types:
HEADER_TYPE_MASK = 0xF000
CONTAINER_HEADER = 0x1000
COMPONENT_HEADER = 0x2000
PART_HEADER_MIN = 0x4000  # 0x4000-0x4FFF are reserved for part headers.

# Part types:
GENERIC_PART_TYPE_MASK = PART_HEADER_MIN + 0x0F00  # 0x4F00
GENERIC_PART_METADATA = PART_HEADER_MIN + 0x0000
GENERIC_PART_1D = PART_HEADER_MIN + 0x0100
GENERIC_PART_2D = PART_HEADER_MIN + 0x0200
ARRAY_2D = GENERIC_PART_2D  # Uncompressed 2D part.
GENICAM_CHUNK = GENERIC_PART_METADATA  # GenICam chunk metadata part.

# Component types (from SFNC):
INTENSITY = 0x0001
METADATA = 0x8001

# Structure sizes in bytes; every derived offset depends on them:
CONTAINER_HEADER_BASE_SIZE = 56
COMPONENT_HEADER_BASE_SIZE = 48
PART_HEADER_BASE_SIZE = 40
PART_HEADER_2D_SIZE = 56
PART_HEADER_1D_SIZE = 64

_OFFSET_SIZE = 8  # sizeof(GenDCOffset)

# Struct layouts; the trailing offset arrays are appended dynamically:
_CONTAINER_HEADER_BASE = '<I3BBHHIQHHIQqII'
_COMPONENT_HEADER_BASE = '<HHIHHHHIIqQIHH'
_PART_HEADER_BASE = '<HHIIHHqQq'
_PART_HEADER_2D = _PART_HEADER_BASE + 'IIHHI'
_PART_HEADER_1D = _PART_HEADER_BASE + 'QHHIQ'


class GenDCParseError(Exception):
    """
    Is raised when a buffer does not hold a valid GenDC container. The
    message tells which validation the container failed.
    """
    pass


def _unpack(data, offset: int, fmt: str):
    """
    Raises
    ------
    GenDCParseError
        If the requested field does not fit in the buffer.
    """
    try:
        return struct.unpack_from(fmt, data, offset)
    except struct.error:
        raise GenDCParseError(
            'truncated container: cannot unpack {} at offset {}'.format(
                fmt, offset))


class Part:
    """
    Represents a GenDC part header. Only the type specific fields that
    follow the common part header base are interpreted for 1D/2D parts;
    every other field remains zero.
    """
    def __init__(
            self, *, part_type: int, flags: int, header_size: int,
            fmt: int, flow_id: int, flow_offset: int, data_size: int,
            data_offset: int, width: int = 0, height: int = 0,
            padding_x: int = 0, padding_y: int = 0,
            info_type_specific: int = 0):
        #
        self.part_type = part_type
        self.flags = flags
        self.header_size = header_size
        self.format = fmt
        self.flow_id = flow_id
        self.flow_offset = flow_offset
        self.data_size = data_size  # As declared by the header.
        self.data_offset = data_offset
        self.width = width
        self.height = height
        self.padding_x = padding_x
        self.padding_y = padding_y
        self.info_type_specific = info_type_specific

    def is_2d(self) -> bool:
        return (self.part_type & GENERIC_PART_TYPE_MASK) == GENERIC_PART_2D

    def is_1d(self) -> bool:
        return (self.part_type & GENERIC_PART_TYPE_MASK) == GENERIC_PART_1D

    def is_array2d(self) -> bool:
        """
        Returns True if the part is an uncompressed 2D array part, which
        is the only part type that can be turned into an image.
        """
        return self.part_type == ARRAY_2D

    def __repr__(self):
        return 'Part: type={}, format={:#010x}, {}x{}, data' \
               ' offset={}, data size={}'.format(
                   hex(self.part_type), self.format, self.width,
                   self.height, self.data_offset, self.data_size)


class Component:
    """
    Represents a GenDC component header together with its parts.
    """
    def __init__(
            self, *, header_type: int, flags: int, header_size: int,
            group_id: int, source_id: int, region_id: int,
            region_offset_x: int, region_offset_y: int, timestamp: int,
            type_id: int, fmt: int, parts: List[Part]):
        #
        self.header_type = header_type
        self.flags = flags
        self.header_size = header_size
        self.group_id = group_id
        self.source_id = source_id
        self.region_id = region_id
        self.region_offset_x = region_offset_x
        self.region_offset_y = region_offset_y
        self.timestamp = timestamp
        self.type_id = type_id
        self.format = fmt
        self.parts = parts

    @property
    def part_count(self) -> int:
        return len(self.parts)

    def first_2d_part(self) -> Optional[Part]:
        for part in self.parts:
            if part.is_2d():
                return part
        return None

    def __repr__(self):
        return 'Component: type id={:#06x}, source id={}, parts={}'.format(
            self.type_id, self.source_id, len(self.parts))


class Container:
    """
    Represents a validated GenDC container descriptor.
    """
    def __init__(
            self, *, signature: int, version_major: int, version_minor: int,
            version_subminor: int, header_type: int, flags: int,
            header_size: int, id: int, variable_fields: int,
            data_size: int, data_offset: int, descriptor_size: int,
            components: List[Component]):
        #
        self.signature = signature
        self.version_major = version_major
        self.version_minor = version_minor
        self.version_subminor = version_subminor
        self.header_type = header_type
        self.flags = flags
        self.header_size = header_size
        self.id = id
        self.variable_fields = variable_fields
        self.data_size = data_size
        self.data_offset = data_offset
        self.descriptor_size = descriptor_size
        self.components = components

    @property
    def component_count(self) -> int:
        return len(self.components)

    def first_image_component(self) -> Optional[Component]:
        """
        Returns the first component that holds a 2D part, mirroring
        GenDCParser::FirstImageComponent.
        """
        for component in self.components:
            if component.first_2d_part() is not None:
                return component
        return None

    def __repr__(self):
        return 'GenDC container: version {}.{}.{}, id={}, ' \
               'components={}'.format(
                   self.version_major, self.version_minor,
                   self.version_subminor, self.id, len(self.components))


def _is_in_data_range(buffer_size: int, offset: int, size: int) -> bool:
    """
    Port of GenDCParser::InDataRange; a negative offset is rejected
    because the C++ implementation rejects pointers below the container.
    """
    if offset < 0:
        return False
    return offset <= buffer_size and size <= buffer_size - offset


def _is_in_descriptor_range(
        descriptor_size: int, offset: int, size: int) -> bool:
    """
    Port of GenDCParser::InDescriptorRange.
    """
    if offset < 0:
        return False
    return offset <= descriptor_size and size <= descriptor_size - offset


def _validate_part(
        data, buffer_size: int, descriptor_size: int,
        part_offset: int) -> Part:
    """
    Port of GenDCParser::ValidatePart; validates the part header at
    ``part_offset`` and returns the parsed Part object.

    Raises
    ------
    GenDCParseError
        If the part header violates the GenDC layout.
    """
    values = _unpack(data, part_offset, _PART_HEADER_BASE)
    part_type, _flags, header_size, fmt, _reserved, flow_id, _flow_offset, \
        data_size, data_offset = values

    if (part_type & HEADER_TYPE_MASK) != PART_HEADER_MIN:
        raise GenDCParseError(
            'invalid part header type: {:#06x}'.format(part_type))
    if header_size < PART_HEADER_BASE_SIZE or not _is_in_descriptor_range(
            descriptor_size, part_offset, header_size):
        raise GenDCParseError(
            'invalid part header size: {} at offset {}'.format(
                header_size, part_offset))
    if not _is_in_data_range(buffer_size, data_offset, data_size):
        raise GenDCParseError(
            'part data out of range: offset={}, size={}'.format(
                data_offset, data_size))

    part = Part(
        part_type=part_type, flags=_flags, header_size=header_size,
        fmt=fmt, flow_id=flow_id, flow_offset=_flow_offset,
        data_size=data_size, data_offset=data_offset)

    # Interpret the type specific information that follows the common
    # part header base as the reference parser does when it casts the
    # header to a 1D/2D part header; fields that do not fit in the
    # buffer remain zero:
    if part.is_2d():
        values_2d = _unpack(
            data, part_offset + PART_HEADER_BASE_SIZE, '<IIHHI')
        size_x, size_y, padding_x, padding_y, _info_reserved = values_2d
        part.width = size_x
        part.height = size_y
        part.padding_x = padding_x
        part.padding_y = padding_y
    elif part.is_1d():
        values_1d = _unpack(
            data, part_offset + PART_HEADER_BASE_SIZE, '<QHHIQ')
        size_x, padding_x, _padding_reserved, _info_reserved, \
            info_type_specific = values_1d
        part.width = int(min(size_x, 0xFFFFFFFF))  # uint32 truncation.
        part.height = 1
        part.padding_x = padding_x
        part.info_type_specific = info_type_specific

    return part


def _validate_component(
        data, buffer_size: int, descriptor_size: int,
        component_offset: int) -> Component:
    """
    Port of GenDCParser::ValidateComponent; validates the component
    header at ``component_offset`` and returns the parsed Component
    object with its parts.

    Raises
    ------
    GenDCParseError
        If the component header violates the GenDC layout.
    """
    values = _unpack(data, component_offset, _COMPONENT_HEADER_BASE)
    header_type, _flags, header_size, _reserved, group_id, source_id, \
        region_id, region_offset_x, region_offset_y, timestamp, type_id, \
        fmt, _reserved2, part_count = values

    if header_type != COMPONENT_HEADER:
        raise GenDCParseError(
            'invalid component header type: {:#06x}'.format(header_type))
    if part_count == 0:
        raise GenDCParseError('component without any part')
    if header_size < COMPONENT_HEADER_BASE_SIZE + part_count * \
            _OFFSET_SIZE:
        raise GenDCParseError(
            'invalid component header size: {}'.format(header_size))

    parts = []
    for p in range(part_count):
        part_offset, = _unpack(
            data, component_offset + COMPONENT_HEADER_BASE_SIZE +
            p * _OFFSET_SIZE, '<q')
        if not _is_in_descriptor_range(
                descriptor_size, part_offset, PART_HEADER_BASE_SIZE):
            raise GenDCParseError(
                'part header out of descriptor: offset={}'.format(
                    part_offset))
        parts.append(
            _validate_part(
                data, buffer_size, descriptor_size, part_offset))

    return Component(
        header_type=header_type, flags=_flags, header_size=header_size,
        group_id=group_id, source_id=source_id, region_id=region_id,
        region_offset_x=region_offset_x, region_offset_y=region_offset_y,
        timestamp=timestamp, type_id=type_id, fmt=fmt, parts=parts)


def _validate_descriptor(
        data, buffer_size: int, descriptor_size: int, header_size: int,
        component_count: int) -> List[Component]:
    """
    Port of GenDCParser::ValidateDescriptor.

    Raises
    ------
    GenDCParseError
        If the descriptor violates the GenDC layout.
    """
    if component_count == 0:
        raise GenDCParseError('container without any component')
    if header_size < CONTAINER_HEADER_BASE_SIZE + component_count * \
            _OFFSET_SIZE:
        raise GenDCParseError(
            'invalid container header size: {}'.format(header_size))

    components = []
    for i in range(component_count):
        component_offset, = _unpack(
            data, CONTAINER_HEADER_BASE_SIZE + i * _OFFSET_SIZE, '<q')
        if not _is_in_descriptor_range(
                descriptor_size, component_offset,
                COMPONENT_HEADER_BASE_SIZE):
            raise GenDCParseError(
                'component header out of descriptor: offset={}'.format(
                    component_offset))
        components.append(
            _validate_component(
                data, buffer_size, descriptor_size, component_offset))
    return components


def parse(data) -> Container:
    """
    Parses a GenDC v1.2 container out of ``data`` and validates it the
    same way GenDCParser::Parse does.

    Parameter
    ---------
    data
        A bytes-like object that holds the container.

    Returns
    -------
    Container
        The parsed container.

    Raises
    ------
    GenDCParseError
        If ``data`` does not hold a valid GenDC container.
    """
    try:
        buffer_size = memoryview(data).nbytes
    except TypeError:
        data = bytes(data)
        buffer_size = len(data)

    if buffer_size < CONTAINER_HEADER_BASE_SIZE:
        raise GenDCParseError(
            'buffer too small: {} bytes'.format(buffer_size))

    values = _unpack(data, 0, _CONTAINER_HEADER_BASE)
    signature, version_major, version_minor, version_subminor, _reserved, \
        header_type, flags, header_size, id, variable_fields, _reserved16, \
        _reserved32, data_size, data_offset, descriptor_size, \
        component_count = values

    if signature != SIGNATURE:
        raise GenDCParseError(
            'invalid signature: {:#010x}'.format(signature))
    if version_major != VERSION_MAJOR:
        raise GenDCParseError(
            'unsupported version: {}.{}.{}'.format(
                version_major, version_minor, version_subminor))
    if header_type != CONTAINER_HEADER:
        raise GenDCParseError(
            'invalid container header type: {:#06x}'.format(header_type))
    if header_size < CONTAINER_HEADER_BASE_SIZE:
        raise GenDCParseError(
            'invalid container header size: {}'.format(header_size))
    if descriptor_size < header_size or descriptor_size > buffer_size:
        raise GenDCParseError(
            'invalid descriptor size: {}'.format(descriptor_size))
    if data_size > 0 and (
            data_offset < descriptor_size or data_offset > buffer_size or
            data_size > buffer_size - data_offset):
        raise GenDCParseError(
            'invalid data offset/size: offset={}, size={}'.format(
                data_offset, data_size))

    components = _validate_descriptor(
        data, buffer_size, descriptor_size, header_size, component_count)

    return Container(
        signature=signature, version_major=version_major,
        version_minor=version_minor, version_subminor=version_subminor,
        header_type=header_type, flags=flags, header_size=header_size,
        id=id, variable_fields=variable_fields, data_size=data_size,
        data_offset=data_offset, descriptor_size=descriptor_size,
        components=components)
