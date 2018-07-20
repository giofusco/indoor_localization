import numpy as np
import cv2
import math
from collections import defaultdict


class AnnotatedMap:

    class MapLandmark:

        def __init__(self, name='', idnumber=None, position=np.zeros((2,)), orientation=np.eye(3), normal=[0, 0]):
            self.name = name
            self.id = idnumber
            self.position = position
            self.orientation = orientation
            self.normal = normal
            self.normal_angle = None

        # set the range of particles yaw compatible with the sign orientation in the map
        def set_FOV_range(self):
            if len(self.normal) == 2:
                if (self.normal == [0, 1]).all():
                    self.normal_angle = 180*math.pi/180
                elif (self.normal == [1, 0]).all():
                    self.normal_angle = 90*math.pi/180
                elif (self.normal == [-1, 0]).all():
                    self.normal_angle = 270*math.pi/180
                else:
                    self.normal_angle = 0

    #default scale value [33.56 / 1.4859] is for SKERI building
    def __init__(self, walls_image_file, walkable_image_file, map_landmarks_file, scale=33.56 / 1.4859):
        self.tag = "AnnotatedMap"
        self.floormap = {}  # Mat for occupacy (in pixels)
        self.floorpoints = {}  # List of EnvironmentMap corners (in meters)
        self.floorsegmemts = {}  # List of EnvironmentMap segments, walls (in meters)
        self.scale = scale  # Pixels per meter
        # self.scale = 34 / 1.48  # Pixels per meter
        self.mapsize_uv = None
        self.mapsize_xy = None
        self.read_map(walls_image_file, walkable_image_file)
        self.map_landmarks_dict = self.read_map_landmarks(map_landmarks_file)

    def read_map(self, mapfile, walkfile):
        layermap = cv2.imread(mapfile, cv2.IMREAD_GRAYSCALE)
        walkmap = cv2.imread(walkfile, cv2.IMREAD_GRAYSCALE)
        mapsize = layermap.shape
        layermap = cv2.threshold(layermap, 128, 255, cv2.THRESH_BINARY)[1]
        walkmap = cv2.threshold(walkmap, 128, 255, cv2.THRESH_BINARY)[1]
        self.floormap['Walls'] = layermap
        self.floormap['Walkable'] = walkmap
        self.mapsize_uv = mapsize
        self.mapsize_xy = (mapsize[0] / self.scale, mapsize[1] / self.scale)
        # cv2.imshow("MAP", layermap)
        # print("MAP LOADED")

    def get_walkable_mask(self):
        return self.floormap['Walkable']

    def get_walls_image(self):
        return self.floormap['Walls']

    def xy2uv(self, pt):
        # Converts motion coordinates to coordinates in the EnvironmentMap (an image)
        u = self.scale * pt[0]
        v = self.mapsize_uv[0] - self.scale * pt[1]
        return np.array([int(u), int(v)])

    def xy2uv_vectorized(self, pts):
        # Converts a list of motion coordinates to coordinates in the EnvironmentMap (an image)
        u = np.trunc(self.scale * pts[:, 0]).astype(np.int)
        v = np.trunc(self.mapsize_uv[0] - self.scale * pts[:, 1]).astype(np.int)
        return np.array([u, v]).T

    def uv2xy(self, pt):
        # Converts mapp coordinates to coordinates in the real world
        x = pt[0] / self.scale
        y = (self.mapsize_uv[0] - pt[1]) / self.scale
        return np.array([x, y])

    def read_map_landmarks(self, mapfile):
        print("\nLoading map features from '{}'".format(mapfile))
        # to handle collisions create a dictionary whose entries are lists
        map_landmark_dict = defaultdict(list)
        with open(mapfile, 'r') as myfile:
            while True:
                line = myfile.readline()
                if not line:
                    break
                line = line.strip()
                if len(line) == 0:
                    continue
                if ':' in line:
                    label, value = self.__parse_yml_line(line)
                    if label == 'map_feature':
                        line = myfile.readline()
                        labelname, name = self.__parse_yml_line(line)
                        line = myfile.readline()
                        labelid, idnumber = self.__parse_yml_line(line)
                        line = myfile.readline()
                        labelpos, pos = self.__parse_yml_line(line)
                        line = myfile.readline()
                        labelrot, rot = self.__parse_yml_line(line)
                        line = myfile.readline()
                        label_normal, normal = self.__parse_yml_line(line)

                        if (labelname == 'name') & \
                                (labelid == 'id') & \
                                (labelpos == 'position') & \
                                (labelrot == 'orientation') & \
                                (label_normal == 'normal'):
                            pos = np.fromstring(pos[1:-1], dtype=np.float, sep=', ')
                            rot = np.fromstring(rot[1:-1], dtype=np.float, sep=', ').reshape((3, 3))
                            normal = np.fromstring(normal[1:-1], dtype=np.float, sep=', ')
                            newmapfeat = self.MapLandmark(name, idnumber, pos, rot, normal)
                            if idnumber == 'exit_sign':
                                newmapfeat.set_FOV_range()
                            # map_landmark_dict[idnumber].append(newmapfeat)
                            map_landmark_dict[idnumber.split()[len(idnumber.split()) - 1]].append(newmapfeat)
                        else:
                            print('YAML file error')
                            continue
        print("Done.")
        return map_landmark_dict

    @staticmethod
    def __parse_yml_line(line):
        name = line[:line.index(':')].strip()
        string = line[line.index(':') + 1:].strip()
        if len(string) == 0:
            value = []
        else:
            value = string
        return name, value
