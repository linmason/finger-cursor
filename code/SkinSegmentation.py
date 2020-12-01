from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt


class ColorDetector:

    histogram = np.zeros((256, 256))

    def __init__(self, colorspace):
        self.colorspace = colorspace

    def TrainImage(self, filenames):
        """
        Trains the color histogram based on training images
        :param filename: file to be train the histogram on
        :return: histogram_out: trained color histogram
        """

        histograms = []

        for filename in filenames:
            print('training: ' + filename)

            # Import files
            path = '../training/'
            img = cv2.imread(path + filename)

            if self.colorspace == 'HSV':
                # Convert BGR color image to HSV
                img_in = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                # add to normalized histogram
                n_histogram = np.zeros((256, 256))
                for i, row in enumerate(img_in):
                    for j, [h, s, v] in enumerate(row):
                        n_histogram[h][s] += 1

            elif self.colorspace == 'RGB':
                # Convert BGR color image to RGB
                img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # add to normalized histogram
                n_histogram = np.zeros((256, 256))
                for i, row in enumerate(img_in):
                    for j, [r, g, b] in enumerate(row):
                        n_histogram[r][g] += 1

            num_px = img_in.shape[0] * img_in.shape[1]
            n_histogram = n_histogram / num_px
            histograms.append(n_histogram)

            print('finished training ' + filename)

        # average the normalized histograms from all files
        sum_histogram = np.zeros_like(histograms[0])
        for histogram in histograms:
            sum_histogram = np.add(sum_histogram, histogram)
        self.histogram = sum_histogram / len(histograms)


    def HistoSegmentation(self, img_in, threshold):
        """
        Decides if each pixel is skin color, return segmented image
        :param img_in: np array image to be process
        :param histogram: np array histogram of segmentation colors
        :return: mask: np array segmented binary image with background as 0
        """
        mask = np.zeros((img_in.shape[0], img_in.shape[1]))
        for i, row in enumerate(img_in):
            for j, [h,s,v] in enumerate(row):
                if self.colorspace == 'HSV':
                    if self.histogram[h][s] < threshold:
                        mask[i][j] = 0
                    else:
                        mask[i][j] = 1


        return mask
