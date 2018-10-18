from map.annotated_map import AnnotatedMap
from collections import defaultdict
import numpy as np
import cv2

class MapManager:

    def __init__(self, map_folder, curr_floor_int):
        self.map_folder = map_folder
        self.map_file = map_folder + '/info.yml'
        self.annotated_maps = defaultdict(AnnotatedMap)
        self.curr_floor = str(curr_floor_int)
        self.parse(self.map_file)

    def set_current_floor(self, curr_floor_int):
        self.curr_floor = str(curr_floor_int)

    def get_color_map_image(self):
        map_image = self.annotated_maps[self.curr_floor].get_walls_image()
        if len(np.shape(map_image)) == 2:
            draw_map = cv2.cvtColor(map_image, cv2.COLOR_GRAY2RGB)
        else:
            draw_map = map_image.copy()
        return draw_map

    def get_walls_image(self):
        return self.annotated_maps[self.curr_floor].get_walls_image()

    def get_walkable_mask(self):
        return self.annotated_maps[self.curr_floor].get_walkable_mask()

    def get_map_landmarks_dict(self):
        return self.annotated_maps[self.curr_floor].map_landmarks_dict

    def map_size_xy(self, pos):
        return self.annotated_maps[self.curr_floor].mapsize_xy[pos]

    def uv2pixels_vectorized(self, pos):
        return self.annotated_maps[self.curr_floor].uv2pixels_vectorized(pos)

    def uv2pixels(self, pos):
        return self.annotated_maps[self.curr_floor].uv2pixels(pos)

    def get_map_landmarks(self, filter):
        return self.annotated_maps[self.curr_floor].map_landmarks_dict[filter]

    def parse(self, map_file):

        with open(map_file, 'r') as myfile:
            while True:
                line = myfile.readline()
                if not line:
                    break
                line = line.strip()
                if len(line) == 0:
                    continue
                if ':' in line:
                    label, value = self.__parse_yml_line(line)
                    if label == 'floor':
                        line = myfile.readline()
                        _, floor_id = self.__parse_yml_line(line)
                        line = myfile.readline()
                        _, features_file = self.__parse_yml_line(line)
                        line = myfile.readline()
                        _, walls_file = self.__parse_yml_line(line)
                        line = myfile.readline()
                        _, walkable_file = self.__parse_yml_line(line)
                        line = myfile.readline()
                        _, scale = (self.__parse_yml_line(line))
                        scale = float(scale)
                        self.annotated_maps[floor_id] = AnnotatedMap(self.map_folder+'/'+walls_file,
                                                                     self.map_folder+'/'+walkable_file,
                                                                     self.map_folder+'/'+features_file, scale=scale)

    @staticmethod
    def __parse_yml_line(line):
        name = line[:line.index(':')].strip()
        string = line[line.index(':') + 1:].strip()
        if len(string) == 0:
            value = []
        else:
            value = string
        return name, value