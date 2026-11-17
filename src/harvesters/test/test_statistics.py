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
sys.path.insert(
    0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..')
)

# Local application/library specific imports
from harvesters._private.core.statistics import Statistics


class FakeClock:
    #
    # A controllable wall clock; the statistics code measures the
    # delivered rate against the wall clock, so the tests drive it.
    #

    def __init__(self):
        self._now = 0.0

    def __call__(self):
        return self._now

    def advance(self, seconds):
        self._now += seconds


def deliver(stats, clock, num_frames, frame_period_s, burst_size=1):
    #
    # Simulates the delivery of num_frames frames that arrive in bursts
    # of burst_size frames every burst_size * frame_period_s. The core
    # increments the image count before it updates the statistics.
    #
    for _ in range(num_frames // burst_size):
        for _ in range(burst_size):
            stats.increment_num_images()
            stats.update_timestamp(buffer=None)
            clock.advance(frame_period_s)


class TestDeliveredFrameRate(unittest.TestCase):
    #
    # The delivered frame rate has to be measured from how many frames
    # reach the host per unit of wall time. A camera that streams at a
    # steady rate and a consumer that drains the arriving buffers in
    # bursts both have to yield the same number.
    #

    def test_reports_a_steady_rate(self):
        clock = FakeClock()
        stats = Statistics(perf_counter=clock)
        deliver(stats, clock, num_frames=400, frame_period_s=1 / 164)
        clock.advance(1 / 164)  # the last frame completes the window
        self.assertAlmostEqual(stats.fps, 164.0, delta=3)

    def test_a_bursty_delivery_does_not_bias_the_rate(self):
        #
        # 164 fps worth of frames that arrive in bursts of three every
        # 18.3 ms must still be reported as 164 fps, and not as the
        # rate at which the consumer happens to wake up:
        #
        clock = FakeClock()
        stats = Statistics(perf_counter=clock)
        deliver(
            stats, clock, num_frames=492, frame_period_s=1 / 164,
            burst_size=3
        )
        self.assertAlmostEqual(stats.fps, 164.0, delta=3)

    def test_the_rate_is_zero_before_the_first_frame(self):
        clock = FakeClock()
        stats = Statistics(perf_counter=clock)
        self.assertEqual(stats.fps, 0.0)

    def test_a_span_too_short_to_trust_is_not_reported(self):
        #
        # Two frames that land within a couple of milliseconds would
        # compute an absurd rate of several hundred fps; the rate is
        # withheld until the samples span long enough to mean anything:
        #
        clock = FakeClock()
        stats = Statistics(perf_counter=clock)
        stats.increment_num_images()
        stats.update_timestamp(buffer=None)
        clock.advance(0.002)
        stats.increment_num_images()
        stats.update_timestamp(buffer=None)
        self.assertEqual(stats.fps, 0.0)

    def test_a_single_frame_does_not_yet_report_a_rate(self):
        clock = FakeClock()
        stats = Statistics(perf_counter=clock)
        stats.increment_num_images()
        stats.update_timestamp(buffer=None)
        self.assertEqual(stats.fps, 0.0)
        self.assertIsNone(stats.fps_min)
        self.assertEqual(stats.fps_max, 0.0)

    def test_minimum_and_maximum_track_the_windowed_rate(self):
        #
        # 100 fps for a moment, then 200 fps: after the warm-up the
        # windowed rate has to have visited both.
        #
        clock = FakeClock()
        stats = Statistics(perf_counter=clock)
        deliver(stats, clock, num_frames=150, frame_period_s=1 / 100)
        deliver(stats, clock, num_frames=500, frame_period_s=1 / 200)
        self.assertAlmostEqual(stats.fps, 200.0, delta=5)
        self.assertIsNotNone(stats.fps_min)
        self.assertLess(stats.fps_min, 120.0)
        self.assertGreater(stats.fps_max, 180.0)

    def test_reset_clears_everything(self):
        clock = FakeClock()
        stats = Statistics(perf_counter=clock)
        deliver(stats, clock, num_frames=300, frame_period_s=1 / 164)
        stats.reset()
        self.assertEqual(stats.fps, 0.0)
        self.assertEqual(stats.num_images, 0)
        self.assertIsNone(stats.fps_min)
        self.assertEqual(stats.fps_max, 0.0)

    def test_the_sample_memory_stays_bounded(self):
        #
        # A session that runs for hours must not accumulate one sample
        # per frame forever; the window keeps a bounded number of them.
        #
        clock = FakeClock()
        stats = Statistics(perf_counter=clock)
        deliver(stats, clock, num_frames=20000, frame_period_s=1 / 164)
        self.assertLess(len(stats._samples), 400)

    def test_the_average_rate_is_derived_from_the_image_count(self):
        clock = FakeClock()
        stats = Statistics(perf_counter=clock)
        stats.increment_num_images(300)
        stats._time_elapsed = 2.0
        # The consumer of the statistics derives the average from the
        # image count and the elapsed time; both have to be available:
        self.assertEqual(stats.num_images, 300)
        self.assertEqual(stats.elapsed_time_s, 2.0)


if __name__ == '__main__':
    unittest.main()
