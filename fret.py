import cv2
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from statistics import median
from math import inf

def threshold(img, s):
    I = img
    I[I <= s] = 0
    I[I > s] = 255
    return I

def remove(l):
    gaps = []
    new_l = []
    for i in range(len(l) - 1):
        gaps.append(l[i + 1] - l[i])
    for index, g in enumerate(gaps):
        if g > 15:
            new_l.append(l[index])
    return new_l
class Video:
   
    def __init__(self, path=None, img=None):
        if img is None:
            self.image = cv2.imread(path)
        elif path is None:
            self.image = img
        else:
            print("Incorrect image parameter")
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def __str__(self):
        return self.image

    def set_image(self, img):
        self.image = img

    def set_gray(self, grayscale):
        self.gray = grayscale

    def print_cv2(self, is_gray=True):
        if is_gray:
            cv2.imshow('image', self.gray)
        else:
            cv2.imshow('image', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def print_plt(self, is_gray=True):
        if is_gray:
            plt.imshow(self.gray, cmap='gray')
        else:
            plt.imshow(self.image)
        plt.show()

    def edges_canny(self, min_val=100, max_val=200, aperture=3):
        return cv2.Canny(self.gray, min_val, max_val, aperture)

    def edges_laplacian(self, ksize=3):
        return cv2.Laplacian(self.gray, cv2.CV_8U, ksize)

    def edges_sobelx(self, ksize=3):
        return cv2.Sobel(self.gray, cv2.CV_8U, 1, 0, ksize)

    def edges_sobely(self, ksize=3):
        return cv2.Sobel(self.gray, cv2.CV_8U, 0, 1, ksize)

    def lines_hough_transform(self, edges, min_line_length, max_line_gap, threshold=15):
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, min_line_length, max_line_gap)
        return lines

def rotate(image):
   
    image_to_rotate = image.image

    edges = image.edges_sobely()
    edges = threshold(edges, 127)

    lines = image.lines_hough_transform(edges, 50, 50) 
    slopes = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slopes.append(abs((y2 - y1) / (x2 - x1)))

    median_slope = median(slopes)
    angle = median_slope * 45

    return Video(img=rotate(image_to_rotate, -angle))


def crop(image):
   
    cropImage = image.image

    edges = image.edges_sobely()
    edges = threshold(edges, 127)

    lines = image.lines_hough_transform(edges, 50, 50)  
    y = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            y.append(y1)
            y.append(y2)

    y_sort = list(sorted(y))
    y_differences = [0]

    first_y = 0
    last_y = inf

    for i in range(len(y_sort) - 1):
        y_differences.append(y_sort[i + 1] - y_sort[i])
    for i in range(len(y_differences) - 1):
        if y_differences[i] == 0:
            last_y = y_sort[i]
            if i > 3 and first_y == 0:
                first_y = y_sort[i]

    return Video(img=cropImage[first_y - 10:last_y + 10])

def detectFrets(neck):
    
    height = len(neck.image)
    width = len(neck.image[0])
    neck_with_frets = np.zeros((height, width, 3), np.uint8)

    edges = neck.edges_sobelx()
    edges = threshold(edges, 127)
    edges = cv2.medianBlur(edges, 3)

    lines = neck.lines_hough_transform(edges, 20, 5)  
    size = len(lines)

    for x in range(size):
        for x1, y1, x2, y2 in lines[x]:
            cv2.line(neck_with_frets, (x1, y1), (x2, y2), (255, 255, 255), 2)

    neck_fr = Video(img=neck_with_frets)
    neck_fret_gray = neck_fr.gray

    slices = {}
    nb_slices = int(height / 15)
    for i in range(nb_slices):
        slices[(i + 1) * nb_slices] = []  

    for index_line, line in enumerate(neck_fret_gray):
        for index_pixel, pixel in enumerate(line):
            if pixel == 255 and index_line in slices:
                slices[index_line].append(index_pixel)

    slices_differences = {}  
    for k in slices.keys():
        temp = []
        n = 0
        slices[k] = list(sorted(slices[k]))
        for p in range(len(slices[k]) - 1):
            temp.append(slices[k][p + 1] - slices[k][p])
            if slices[k][p + 1] - slices[k][p] > 1:
                n += 1
        slices_differences[k] = temp

    x_values = defaultdict(int)
    for j in slices_differences.keys():
        for index, gap in enumerate(slices_differences[j]):
            if gap > 1:
                x_values[slices[j][index]] += 1

    potential_frets = []
    x_values = dict(x_values)
    for x, nb in x_values.items():
        if nb > 1:
            potential_frets.append(x)

    potential_frets = list(sorted(potential_frets))
    potential_frets = remove(potential_frets)

    potential_ratio = []
    for i in range(len(potential_frets) - 1):
        potential_ratio.append(round(potential_frets[i + 1] / potential_frets[i], 3))

    ratio = potential_ratio[-1]
    last_x = potential_frets[-1]
    while 1:
        last_x *= ratio
        if last_x >= width:
            break
        else:
            potential_frets.append(int(last_x))

    for x in potential_frets:
        cv2.line(neck.image, (x, 0), (x, height), (127, 0, 255), 3)

    return Video(img=neck.image)
