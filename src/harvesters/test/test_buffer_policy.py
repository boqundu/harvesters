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
import os
import sys
import unittest

# Make the source tree importable when the tests are run from anywhere:
sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..')))

# Local application/library specific imports
from harvesters.util.buffer_policy import compute_num_buffers


class TestComputeNumBuffers(unittest.TestCase):
    #
    # The monitor sizes the acquisition buffer queue so that short
    # delivery bursts cannot drop frames: ten buffers by default, as
    # many more as the host memory comfortably allows, and fewer only
    # when the memory is too tight to pay for the default.
    #

    def test_defaults_to_ten_buffers(self):
        #
        # Half of 20 MiB pays for exactly ten buffers of 1 MiB; when
        # the memory pays for neither more nor less, ten is announced:
        #
        self.assertEqual(compute_num_buffers(
            payload_size=1024 * 1024, available_memory_bytes=20 * 1024 * 1024
        ), 10)

    def test_unknown_payload_size_falls_back_to_the_default(self):
        self.assertEqual(compute_num_buffers(
            payload_size=0, available_memory_bytes=8 * 1024 * 1024 * 1024
        ), 10)

    def test_unknown_available_memory_falls_back_to_the_default(self):
        self.assertEqual(compute_num_buffers(
            payload_size=1024, available_memory_bytes=0
        ), 10)

    def test_counts_the_usable_fraction_of_available_memory(self):
        #
        # Half of 512 MiB pays for 64 buffers of 4 MiB each:
        #
        self.assertEqual(compute_num_buffers(
            payload_size=4 * 1024 * 1024,
            available_memory_bytes=512 * 1024 * 1024
        ), 64)

    def test_tight_memory_reduces_the_count_below_the_default(self):
        #
        # Half of 10 MiB pays for exactly 5 buffers of 1 MiB; when the
        # memory cannot pay for the default ten, the affordable count
        # is used as it is:
        #
        self.assertEqual(compute_num_buffers(
            payload_size=1024 * 1024, available_memory_bytes=10 * 1024 * 1024
        ), 5)

    def test_memory_starvation_does_not_go_below_a_viable_queue(self):
        #
        # Half of 1 MiB would pay for half a buffer of 1 MiB; a queue
        # still has to hold a couple of buffers to be viable at all:
        #
        self.assertEqual(compute_num_buffers(
            payload_size=1024 * 1024, available_memory_bytes=1024 * 1024
        ), 2)

    def test_partial_buffers_are_floored(self):
        #
        # Half of 20 MB pays for 33.3 buffers of 300 kB; the fraction
        # of an unfinished buffer is discarded:
        #
        self.assertEqual(compute_num_buffers(
            payload_size=300000, available_memory_bytes=20 * 1000 * 1000
        ), 33)

    def test_a_small_payload_cannot_produce_an_absurd_count(self):
        #
        # Line-scan cameras report payloads of a few kB; without a cap
        # the computation would announce millions of buffers:
        #
        self.assertEqual(compute_num_buffers(
            payload_size=1024, available_memory_bytes=64 * 1024 * 1024 * 1024
        ), 100)

    def test_the_bounds_and_the_fraction_are_configurable(self):
        self.assertEqual(compute_num_buffers(
            payload_size=1000, available_memory_bytes=100 * 1000,
            default=32, maximum=64, usable_fraction=1.0
        ), 64)
        self.assertEqual(compute_num_buffers(
            payload_size=1000, available_memory_bytes=100 * 1000,
            default=32, maximum=200, usable_fraction=1.0
        ), 100)
        self.assertEqual(compute_num_buffers(
            payload_size=1000, available_memory_bytes=3000,
            usable_fraction=1.0, minimum=1
        ), 3)
        self.assertEqual(compute_num_buffers(
            payload_size=1000, available_memory_bytes=3000,
            usable_fraction=1.0, minimum=4
        ), 4)


if __name__ == '__main__':
    unittest.main()
