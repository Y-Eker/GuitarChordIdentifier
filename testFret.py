import os
import time
from matplotlib import pyplot as plt
import fret
import cv2
import pathlib
from fret import *
from math import inf



def fretTest():
    i = 1
    plt.figure(1)
    for filename in os.listdir('videos/GuitarVid4.mp4'):
        start_time = time.time()
        chord_image = Video(path='dir' + filename)
        rotated_image = rotate(chord_image)
        cropped_image = crop(rotated_image)
        neck_fret = detectFrets(cropped_image)
        plt.subplot(int("42" + str(i)))
        i += 1
        plt.imshow(cv2.cvtColor(chord_image.image, cv2.COLOR_BGR2RGB))
        plt.subplot(int("42" + str(i)))
        i += 1
        plt.imshow(cv2.cvtColor(neck_fret.image, cv2.COLOR_BGR2RGB))
        print("Done - Time elapsed: %s seconds" % round(time.time() - start_time, 2))

    plt.show()

if __name__ == "__main__":
    fretTest()
