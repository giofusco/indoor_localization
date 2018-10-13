import numpy as np
from input import data_constants as dc

class FloorChangeDetector:

    GOING_UP = 0
    GONE_UP = 1
    GOING_DOWN = 2
    GONE_DOWN = 3
    STATIONARY = 5

    def __init__(self, starting_floor_number, hPa_threshold = 0.35):
        self.starting_floor_number = starting_floor_number
        self.hPa_threshold = hPa_threshold
        self.status = None
        self.previous_hpa = None

    def update(self, data):

        hPa = data[dc.BAROMETER] * 10