import glob, os
import linecache
import numpy as np
import input.data_constants as dc
import cv2


class DataParser:

    VIO_FILE_REGEXP = '/VIO*.txt'
    BLOCK_LEN = 6 # use 6 for version with barometer

    def __init__(self, data_folder, image_format='jpg'):
        self.folder = data_folder
        self.vio_filename = None
        self._initialized = False
        self._line_counter = 1
        self.image_format = image_format
        self._initialize_parser()

    def _initialize_parser(self):
        self.vio_filename = glob.glob(self.folder + self.VIO_FILE_REGEXP)
        if not len(self.vio_filename) == 0:
            line = linecache.getline(self.vio_filename[0], self._line_counter)
            self._line_counter += 1
            if len(line) == 0:
                raise IOError("Could not read file:", self.vio_filename[0])
            self.initialized = True
            # we skip the first sample, sometimes it is extremely noisy due to VIO re-initialization
            self.read_next(False)
        else:
            raise IOError("Could not locate VIO data file ")

    def read_next(self, load_image=True):
        if self.initialized:
            lines = []
            for l in range(0,self.BLOCK_LEN):
                lines.append(linecache.getline(self.vio_filename[0], l+self._line_counter))
            self._line_counter += self.BLOCK_LEN
            if len(lines[0]) > 0:
                data = self._parse(lines)
                data[dc.FOLDER] = self.folder
                data[dc.IMAGE] = None
                if load_image:
                    img = cv2.imread(data[dc.IMAGE_FILENAME])
                    img = cv2.resize(img, (360, 640), img)
                    data[dc.IMAGE] = img # cv2.imread(data[dc.IMAGE_FILENAME])
                return data
            else:
                return {}
        else:
            raise RuntimeError("Trying to read when parser not yet initialized.")

    def _parse(self, lines):
        data = {}
        data[dc.TIMESTAMP] = float(lines[0])
        data[dc.VIO_STATUS] = lines[1].split('(')[0].rstrip()
        row1, line = lines[2][16:].split(')', 1)
        row2, line = line[4:].split(')', 1)
        row3, line = line[4:].split(')', 1)
        row4, line = line[4:].split(')', 1)
        matrix_string = row1 + ';' + row2 + ';' + row3 + ';' + row4
        data[dc.CAMERA_MATRIX] = np.matrix(matrix_string)
        data[dc.CAMERA_POSITION] = [float(lines[3].split(',')[0]), float(lines[3].split(',')[1]),
                                 float(lines[3].split(',')[2][0:-1])]
        data[dc.CAMERA_ROTATION] = [float(lines[4][7:].split(',')[0]), float(lines[4][7:].split(',')[1]),
                                 float(lines[4][7:].split(',')[2][0:-2])]
        data[dc.IMAGE_FILENAME] = os.path.join(self.folder, lines[0][0:-1] + "." + self.image_format)
        if self.BLOCK_LEN > 5:
            data[dc.BAROMETER] = float(lines[5])
        return data
