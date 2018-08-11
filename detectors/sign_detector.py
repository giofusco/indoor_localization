# from observer import IObserver
from input import data_constants as dc
import numpy as np
import cv2
import camera_info
# import svmutil
import math
import camera_info
from classifiers import svmutil

DETECTION_WINDOW_NAME = "SignDetector"


class SignDetector:

    def __init__(self, name, camera_id = camera_info.IPHONE8_640x360):
        # init descriptors
        self.hog = cv2.HOGDescriptor('./classifiers/hog.xml')
        self.sift = cv2.xfeatures2d.SIFT_create()

        # load classifiers
        self.cascade_class = cv2.CascadeClassifier('./classifiers/exit_sign_cascade.xml')
        self.svm = svmutil.svm_load_model('./classifiers/exit_sign_model.svm')
        self.last_frame_RGB = None
        self.name = name
        self.roi = None
        self.sign_height_m = -1
        self.avg_filter = np.ones((9,9), dtype = np.float32) * 1/81

        self.camera_calibration = camera_info.get_camera_params(camera_id)

        # iPhone 8 camera calibration 360
        self.camera_matrix = self.camera_calibration[camera_info.CAMERA_MATRIX]
        self.dist_coeffs = self.camera_calibration[camera_info.DIST_COEFFS]

        self.fx = self.camera_matrix[0][0,0]
        self.fy = self.camera_matrix[1][0,1]
        self.observed_distance_to_sign = -1.


    def get_observation(self, data):
        # self.observed_distance_to_sign = -1.
        # self.update(data)
        return self.observed_distance_to_sign

    def extract_features_halves(self, img, feature_extractor):
        h, w, _ = img.shape
        sift_img = np.copy(img)
        kp = []
        des = []
        k, d = feature_extractor.detectAndCompute(img[0:h, 0:int(w / 2)], None)

        kp.append(k)
        des.append(d)

        k, d = feature_extractor.detectAndCompute(img[0:h, int(w / 2) + 1:w], None)
        for i in range(0, len(k)):
            k[i].pt = (k[i].pt[0] + int(w / 2) + 1, k[i].pt[1])

        # cv2.drawKeypoints(sift_img, k, sift_img)
        # cv2.imshow("SIFT", sift_img)
        # cv2.waitKey(-1)

        kp.append(k)
        des.append(d)
        return kp, des

    def resize_roi(self, r, offset, img_shape):
        x0 = r[0] - r[0] * offset
        w = (r[0] + r[2]) + (r[0] + r[2]) * offset
        y0 = r[1] - r[1] * offset
        h = (r[1] + r[3]) + (r[1] + r[3]) * offset

        if x0 < 0: x0 = 0
        if y0 < 0: y0 = 0
        if w > img_shape[1]: w = img_shape[1] - 1
        if h > img_shape[0]: h = img_shape[0] - 1

        return [int(x0), int(y0), int(w), int(h)]

    def compute_hog(self, img, r, hog):
        h = r[1] + r[3]
        if h > img.shape[0]:
            h = img.shape[0] - 1
        w = r[0] + r[2]
        if w > img.shape[1]:
            w = img.shape[1] - 1
        roi = img[r[1]:h, r[0]:w]  # crop the image
        roi = cv2.resize(roi, (36, 24))
        r_hog = hog.compute(roi)
        return r_hog

    def set_sign_height(self, height_m):
        self.sign_height_m = height_m

    def get_sign_info(self):
        return self.observed_distance_to_sign, self.roi


    def update(self, data):
        self.observed_distance_to_sign = None
        self.roi = None
        if not data[dc.IMAGE] is None:
            self.last_frame_RGB = np.copy(data[dc.IMAGE])

            # rotate image according to VIO
            rows, cols, ch = self.last_frame_RGB.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90+data[dc.CAMERA_ROTATION][2]*180/math.pi, 1)
            rotated_frame = cv2.warpAffine(self.last_frame_RGB, M, (cols, rows))

            gray = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("SIGNINPUT", self.last_frame_RGB )
            # cv2.imshow("ROT", rotated_frame)
            # cv2.waitKey(1)
            # print(data[dc.CAMERA_ROTATION]*180/3.14)
            rois_stage1 = self.cascade_class.detectMultiScale(gray, 1.25, minNeighbors=1, minSize=(36, 24), maxSize=(180, 240))

            if len(rois_stage1) > 0:
                self.sign_height = -1
                best_prob = 0
                for r in rois_stage1:

                    r_hog = self.compute_hog(gray, r, self.hog)
                    p_labels, p_acc, p_vals = svmutil.svm_predict([1], r_hog.transpose().tolist(), self.svm, '-b 1')
                    # print (p_vals)
                    if p_vals[0][0] > 0.5:
                        cv2.rectangle(rotated_frame, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (255, 0, 0), 2)
                        cv2.imshow("Stage2", rotated_frame)
                        if p_vals[0][0] > best_prob:
                            best_prob = p_vals[0][0]
                            self.roi = r

                if self.roi is not None:
                    self.observed_distance_to_sign = self._get_sign_height(rotated_frame)


    def _get_sign_height(self, img):

        h = self.roi[1] + self.roi[3]
        if h > img.shape[0]:
            h = img.shape[0] - 1
        w = self.roi[0] + self.roi[2]
        if w > img.shape[1]:
            w = img.shape[1] - 1
        roi = img[self.roi[1]:h, self.roi[0]:w]  # crop the image
        Ihsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV_FULL)

        center = np.array((0, 0), dtype=np.int32)
        center[0] = math.ceil(Ihsv.shape[0] / 2)
        center[1] = math.ceil(Ihsv.shape[1] / 2)

        offset_row = int(math.floor(Ihsv.shape[0] * .3))
        offset_col = int(math.floor(Ihsv.shape[1] * .15))
        # Ihsv(center(0)-offset_row : center(0) + offset_row, 1:-1)
        roi_hsv = Ihsv[(center[0] - offset_row) : center[0] + offset_row,
                  center[1] - offset_col:center[1] + offset_col,:]

        h_roi = np.ravel(roi_hsv[:, :, 0])
        s_roi = np.ravel(roi_hsv[:, :, 1])

        H, xedges, yedges = np.histogram2d(h_roi, s_roi, [10, 10])

        H_filt = cv2.filter2D(H,-1, self.avg_filter)
        # print(np.max(H_filt))
        ind = np.unravel_index(np.argmax(H_filt, axis=None), H_filt.shape)
        # cv2.imshow("ROI", Ihsv[:,:,1])
        # print(H_filt)
        ht = xedges[ind[0]]
        st = yedges[ind[1]]
        hh = Ihsv[:,:,0] >= ht
        hs = Ihsv[:, :, 1] >= st
        M = hh & hs
        # print st, ht
        rect_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closing = cv2.morphologyEx(M.astype(np.uint8)*255 , cv2.MORPH_CLOSE, rect_element, iterations=2)
        horizontal_sum = np.sum(closing/255, 1)

        thresh = closing.shape[1]*.05
        found = False
        i = 0
        while i < len(horizontal_sum):
            if horizontal_sum[i] <= thresh:
                horizontal_sum[i] = 0
                i += 1
            else:
                break
        i = len(horizontal_sum) - 1
        while i >= 0:
            if horizontal_sum[i] <= thresh:
                horizontal_sum[i] = 0
                i -= 1
            else:
                break
        Z = self.fy * 0.2032 / (np.sum(horizontal_sum>0) + 0.001)
        self.observed_distance_to_sign = Z
        return Z
        # cv2.imshow("ROI_filt", closing)
        # cv2.imshow("ROI", M.astype(np.uint8)*255)

        # cv2.waitKey(-1)


