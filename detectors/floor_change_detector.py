import numpy as np
from input import data_constants as dc
from math import floor

class FloorChangeDetector:

    GOING_UP = 0
    GONE_UP = 1
    GOING_DOWN = 2
    GONE_DOWN = 3
    STATIONARY = 5

    def __init__(self, starting_floor_number, hPa_threshold = 0.3):
        self.current_floor_number = starting_floor_number
        self.hPa_threshold = hPa_threshold
        self.status = None
        self.previous_hpa = None
        self.zero_count_threshold = 25
        self.min_delta_val = 0.005
        self.partial_delta_threshold = 0.1
        self.p_Ref = None
        self.prev_value = None
        self.zero_count = 0
        self.curr_direction = 0
        self.prev_direction = 0
        self.reset = False

    def update(self, data):

        if data[dc.BAROMETER] is not None:
            hPa = data[dc.BAROMETER] * 10

            num_stories = 0.0
            deltaP = 0

            if self.reset:
                self.p_Ref = hPa
                self.reset = False

            elif self.p_Ref is None:
                self.p_Ref = hPa
            else:
                deltaP = (hPa - self.p_Ref)

                local_delta = 0
                if self.prev_value is not None:
                    local_delta = abs(hPa - self.prev_value)

                if abs(deltaP) > self.hPa_threshold:
                    if local_delta < self.min_delta_val:
                        self.reset = True
                        num_stories = -(deltaP / abs(deltaP)) * floor(abs(deltaP) / self.hPa_threshold)
                        self.current_floor_number += num_stories

                self.prev_value = hPa

            return (self.current_floor_number, num_stories)
        else:
            raise RuntimeError("ERROR: No Barometer values available. Ending navigation system.")
