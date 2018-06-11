# load map and compute traversability between two points
# updated to Python 3

import numpy as np
from pylab import close, colorbar, figure, gray, hist, hold, imshow, plot, savefig, show
import scipy
from scipy.misc import imread, imsave
from math import trunc  # similar to round() but it returns an int; however, trunc(3.9) is 3, etc.
from numba import jit
from numba import cuda

import time
import cv2

# binary_map = imread('Walls-binarized.bmp')+0 #a binarized version of what Alejandro sent me 7/14/16
# binary_map = imread('example-map.bmp')+0
binary_map = imread('../res/Walls.png') + 0
binary_map = binary_map[:, :, 0]

# experiment with different way of loading image:
# binary_map = cv2.imread('example-map.bmp')
# binary_map = binary_map[:,:,0]

scale = 33.56 / 1.4859  # Pixels per meter, from Alejandro


@jit
def traversable(r1, c1, r2, c2, m):
    """Traversability calculation:
    Inputs are pixel locations (r1,c1) and (r2,c2) and 2D map array.
    Draw a line from (r1,c1) to (r2,c2) and determine whether any white pixels are hit along the way."""

    dr = abs(r1 - r2)  # delta row
    dc = abs(c1 - c2)

    flag = 0  # nothing hit yet

    if dr > dc:  # more rows than columns so loop over rows
        for k in range(dr + 1):
            if dr == 0:
                frac = 0
            else:
                frac = k / (dr + 0.)  # fraction of way from one endpoint to the other
            r = trunc(r1 + frac * (r2 - r1))
            c = trunc(c1 + frac * (c2 - c1))
            if m[r, c] > 0:  # hit!
                flag = 1
                return flag

    else:  # more (or an equal # of) columns than rows so loop over columns
        for k in range(dc + 1):
            if dc == 0:
                frac = 0
            else:
                frac = k / (dc + 0.)  # fraction of way from one endpoint to the other
            r = trunc(r1 + frac * (r2 - r1))
            c = trunc(c1 + frac * (c2 - c1))
            if m[r, c] > 0:  # hit!
                flag = 1
                return flag

    return flag


@jit
def traversable2(r1, c1, r2, c2, m):  # improved version? need to test!
    """Traversability calculation:
    Inputs are pixel locations (r1,c1) and (r2,c2) and 2D map array.
    Draw a line from (r1,c1) to (r2,c2) and determine whether any white (wall) pixels are hit along the way.
    Return 1 if a wall is hit, 0 otherwise."""

    dr = abs(r1 - r2)  # delta row
    dc = abs(c1 - c2)  # delta column
    hits = 0
    span = max(dr, dc)  # if more rows than columns then loop over rows; else loop over columns
    span_float = span + 0.  # float version
    if span == 0:  # i.e., special case: (r1,c1) equals (r2,c2)
        multiplier = 0.
    else:
        multiplier = 1. / span
    for k in range(
                    span + 1):  # k goes from 0 through span; e.g., a span of 2 implies there are 2+1=3 pixels to reach in loop
        frac = k * multiplier
        # r = trunc(r1 + frac * (r2 - r1))
        # c = trunc(c1 + frac * (c2 - c1))
        hits += m[trunc(r1 + frac * (r2 - r1)), trunc(c1 + frac * (c2 - c1))]

        # if m[r, c] > 0:  # hit!
        #     return 1  # report hit and exit function
    if hits > 0:
        return 1
    return 0  # if we got to here then no hits

@jit(nopython=True)
def traversable3(r1, c1, height, width, m):  # improved version? need to test!
    """Traversability calculation:
    Inputs are pixel locations (r1,c1) and (r2,c2) and 2D map array.
    Draw a line from (r1,c1) to (r2,c2) and determine whether any white (wall) pixels are hit along the way.
    Return 1 if a wall is hit, 0 otherwise."""
    visib = binary_map * 0
    for r2 in range(height):
        for c2 in range(width):
            # r2 = r
            # c2 = c
            dr = abs(r1 - r2)  # delta row
            dc = abs(c1 - c2)  # delta column
            hits = 0
            span = max(dr, dc)  # if more rows than columns then loop over rows; else loop over columns
            span_float = span + 0.  # float version
            if span == 0:  # i.e., special case: (r1,c1) equals (r2,c2)
                multiplier = 0.
            else:
                multiplier = 1. / span
            for k in range(
                            span + 1):  # k goes from 0 through span; e.g., a span of 2 implies there are 2+1=3 pixels to reach in loop
                frac = k * multiplier
                hits += m[trunc(r1 + frac * (r2 - r1)), trunc(c1 + frac * (c2 - c1))]

                if hits > 0:  # hit!
                    visib[r2, c2] = 1
                    break
    return visib

# now create visibility map
# origin = (177,236)
origin = (195, 283)
visib = binary_map * 0
h, w = np.shape(binary_map)

print('starting...')

tic = time.time()
visib = traversable3(origin[0], origin[1], h, w, binary_map)
toc = time.time()
print('elapsed time:', toc - tic)
print('elapsed time per pixel:', (toc - tic) / (h * w))

figure()
imshow(visib)
show()

