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
import time
from collections import deque

# Related third party imports

# Local application/library specific imports


class Statistics:
    #
    # Measures the rate at which buffers are delivered to the host.
    #
    # The rate is derived from how many buffers arrive per unit of
    # wall time within a sliding window. It is deliberately not
    # derived from the device timestamps: the buffers are delivered
    # on a worker thread whose consecutive samples are, in practice,
    # either back-to-back drains of a burst of buffers or pairs that
    # span a whole event-wait period of several frames, and a rate
    # that is computed from a timestamp gap divided by the wall time
    # between two samples then reports the cadence of the consumer
    # instead of the cadence of the camera.
    #
    _WINDOW_SECONDS = 1.0
    _WARMUP_SECONDS = 0.25
    _MIN_REPORT_SPAN_SECONDS = 0.05

    def __init__(self, perf_counter=None):
        #
        super().__init__()

        #
        self._perf_counter = perf_counter or time.perf_counter

        #
        self.reset()

    def update_timestamp(self, buffer):
        #
        # The buffer itself is not inspected; see the class comment.
        # The core updates the image count before it calls this
        # method, so every delivered buffer extends the window by one.
        #
        now = self._perf_counter()
        self._samples.append((now, self._num_images))

        # Drop the samples that have fallen out of the window; the
        # oldest sample is kept as the boundary so that the window
        # always spans at least the window period once it is filled:
        cutoff = now - self._WINDOW_SECONDS
        while len(self._samples) >= 2 and self._samples[1][0] <= cutoff:
            self._samples.popleft()

        oldest_time, oldest_count = self._samples[0]
        span = now - oldest_time
        if span <= 0:
            return

        rate = (self._num_images - oldest_count) / span
        if span >= self._MIN_REPORT_SPAN_SECONDS:
            # A rate over a span of a few milliseconds says more about
            # the scheduling of the first frames than about the
            # camera; withhold it until the span is meaningful:
            self._fps = rate

        # The extrema are only meaningful once the window has warmed
        # up; before that, a rate from a fraction of the window is
        # dominated by the delivery jitter of the first frames:
        if span >= self._WARMUP_SECONDS:
            if rate > self._fps_max:
                self._fps_max = rate
            if self._fps_min is None or rate < self._fps_min:
                self._fps_min = rate

    def reset(self):
        self._time_base = time.time()
        self._time_elapsed = 0
        self._num_images = 0
        self._samples = deque()
        self._fps = 0.
        self._fps_max = 0.
        self._fps_min = None

    def increment_num_images(self, num=1):
        self._time_elapsed = time.time() - self._time_base
        self._num_images += num

    @property
    def fps(self):
        return self._fps

    @property
    def fps_max(self):
        return self._fps_max

    @property
    def fps_min(self):
        return self._fps_min

    @property
    def num_images(self):
        return self._num_images

    @property
    def elapsed_time_s(self):
        return self._time_elapsed
