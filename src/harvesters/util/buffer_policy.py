#!/usr/bin/env python3
# ----------------------------------------------------------------------------
#
# Copyright 2026 harvesters_monitor contributors
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

# Local application/library specific imports


def compute_num_buffers(
        payload_size, available_memory_bytes,
        *, default=10, maximum=100, usable_fraction=0.5, minimum=2):
    #
    # Sizes the acquisition buffer queue. Frames that arrive while the
    # previous buffers are still being processed need a buffer to land
    # in; if none is free, the GenTL producer drops the frame and the
    # frame rate collapses. The queue defaults to ten buffers, and as
    # many more are announced as the memory that is currently
    # available on the host comfortably allows; when that memory is
    # too tight to pay for the default, the count is reduced to what
    # it can pay for, but never below a minimum viable queue, because
    # announcing fewer buffers than that would drop frames by design.
    # Announcing a count that the memory cannot pay for would instead
    # fail the buffer allocation when the acquisition starts, so the
    # affordable count is always what gets announced.
    #
    # Every buffer costs one payload of memory. Only half of the
    # available memory is offered to the buffers so that this process,
    # the GUI and the rest of the system keep room to work in, and a
    # cap keeps small-payload devices from announcing absurd counts.
    # If the payload size or the available memory is unknown, the
    # default is announced; the buffers are allocated when the
    # acquisition starts.
    #
    if payload_size <= 0 or available_memory_bytes <= 0:
        return default

    budget = usable_fraction * available_memory_bytes
    count = int(budget // payload_size)
    return max(minimum, min(maximum, count))
