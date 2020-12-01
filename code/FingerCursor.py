from PIL import Image, ImageFilter
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
import os
from SkinSegmentation import ColorDetector
import math


class FingerTracker:

    # Contours of template gestures
    template_contours = []
    drawing_pts = []

    # Tracking information for each image
    topmost = ()
    gesture = 0
    largest_contour = []


    def __init__(self, morph_op, edge_det, track_method, match_method, min_skin_hsv=[0,60,80], max_skin_hsv=[20, 255, 255]):
        self.morph_op = morph_op
        self.edge_det = edge_det
        self.track_method = track_method
        self.match_method = match_method

        # Skin color bounds CHANGE LATER TO MY HISTOGRAM
        self.min_skin_hsv = np.array(min_skin_hsv, np.uint8)
        self.max_skin_hsv = np.array(max_skin_hsv, np.uint8)

        self.color_detector = ColorDetector('HSV')


    def GetContour(self, img_in):

        # Convert from BGR to HSV MIGHT MOVE SOME OUT AS LIST COMPREHENSION FOR EFFICIENCY
        img_hsv = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)

        # Smooth the image
        #img_smoothed = cv2.GaussianBlur(img_hsv, (11, 11), 0)
        img_smoothed = img_hsv

        # Skin Color Segmentation CHANGEABLE MIN/MAX
        img_segm = cv2.inRange(img_smoothed, self.min_skin_hsv, self.max_skin_hsv)
        print(img_segm.shape)

        drawn_img_template = Image.fromarray(img_segm, 'L')
        drawn_img_template.save('../results/intermediate/segmented.png')

        # Morphological operator to enhance segmentation CHANGEABLE MO and SE SIZE
        SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        img_mo = cv2.erode(img_segm, SE, iterations=2)
        img_mo = cv2.dilate(img_mo, SE, iterations=2)
        #img_mo = cv2.morphologyEx(img_segm, cv2.MORPH_OPEN, SE)

        drawn_img_template = Image.fromarray(img_mo, 'L')
        drawn_img_template.save('../results/intermediate/mo.png')

        # Find contours
        contours, hierarchy = cv2.findContours(img_mo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Identify largest contour
        return max(contours, key = cv2.contourArea)


    def TrainHistogram(self, imgs_train):
        self.color_detector.TrainImage(imgs_train)


    def SaveTemplate(self, imgs_template):
        """
        Saves the contour of the template gesture images
        :param imgs_template: list of np array representing the template gesture images
        :return:
        """

        for img_template in imgs_template:
            self.template_contours.append(self.GetContour(img_template))



    def TrackandMatch(self, img_in):
        """
        For each image, preprocess, segment, detect edge, extract contour and find finger and match to gestures
        :param imgs_in: np array representing BGR image
        :return:
        """

        # Get largest skin contour in image
        self.largest_contour = self.GetContour(img_in)

        # Identify highest point in contour
        self.topmost = tuple(self.largest_contour[self.largest_contour[:,:,1].argmin()][0])

        # Compare contour of image to gesture templates CHANGEABLE COMPARE MATCH TYPE
        # !!!! might need more than 1 gesture template
        gesture_scores = np.zeros((len(self.template_contours), ), np.float32)
        for template_i, template_contour in enumerate(self.template_contours):
            gesture_scores[template_i] = cv2.matchShapes(self.largest_contour, template_contour, cv2.CONTOURS_MATCH_I2, 0.0)

        # Find the gesture template with closest contour
        self.gesture = np.argmin(gesture_scores)

    def Draw(self, img_draw):
        """
        Draws on img_draw the based on information of current img_in frame
        :param img_draw: np array representing BGR image to be drawn on
        :return: img_out: np array representing drawn BGR image
        """
        img_out = img_draw

        # draw contour
        img_out = cv2.drawContours(img_out, [self.largest_contour], 0, (0, 80, 255), 3)

        # If gesture is pointer
        if self.gesture == 1:
            img_out = cv2.circle(img_out, self.topmost, 10, (0, 0, 255), 3)
            self.drawing_pts.append(self.topmost)

        # Draw dot at saved drawing positions in img_draw
        for index, drawing_pt in enumerate(self.drawing_pts):
            if index == len(self.drawing_pts) - 1:
                break
            # if points close enough together
            if math.dist(drawing_pt, self.drawing_pts[index + 1]) < img_draw.shape[0]/4:
                img_out = cv2.line(img_out, drawing_pt, self.drawing_pts[index + 1], (0, 0, 255), 3)

        return img_out


    def TrackandDraw(self, imgs_in, imgs_draw, video):
        """
        tracks and draws on a set of imgs
        :param imgs_in: list of np arrays BGR images
        :param imgs_draw: list of np arrays representing BGR images to be drawn on
        :return:
        """
        for i, img_in in enumerate(imgs_in):
            # Run algo
            self.TrackandMatch(img_in)
            img_out = self.Draw(imgs_draw[i])

            # Convert colorspace
            visualize_img = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)

            # Save drawn image
            drawn_img = Image.fromarray(visualize_img, 'RGB')
            drawn_img.save('../results/final/' + self.morph_op + str(self.edge_det) + self.track_method + self.match_method + str(i) + '.png')

            # Write drawn image to video
            video.write(img_out)


def FingerCursor(gaussian_filter, morph_op, edge_det, track_method, match_method):
    """
    Wrapper function for pre- and post- processing of Template-matching based Finger Tracking to convert image
    colorspace and draw results
    :param morph_op: string representing image matching criteria used for similarity
    :param match_method: string representing image matching criteria used for similarity
    :return:
    """
    print('Begin processing')

    # Import template images
    print("Importing template images")
    template_imgs = []
    template_path = '../data/template_images/'
    for i, template_filename in enumerate(sorted(os.listdir(template_path))):
        print(template_filename)
        template_imgs.append(cv2.imread(template_path + template_filename))

    # Import files into list "imgs"
    print("Importing video as images")
    cam = cv2.VideoCapture("../data/video_in.mp4")
    imgs = []

    while(True):
        # Read image from video
        ret, frame = cam.read()

        if ret:
            imgs.append(frame)
        else:
            break

    # Placeholder
    imgs_in = imgs

    # Initialize video composer
    video_path = '../results/final/' + morph_op + str(edge_det) + track_method + match_method + '.mp4'
    height, width, layers = imgs_in[0].shape
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(video_path, fourcc, 24, (width, height))

    # Run Tracking
    print("Running tracking and matching")
    ft = FingerTracker(morph_op, edge_det, track_method, match_method)
    ft.GetContour(template_imgs[1])
    ft.SaveTemplate(template_imgs)
    ft.TrackandDraw(imgs_in, imgs_in, video)

    cv2.destroyAllWindows()
    video.release()

    print('Finished processing')


FingerCursor(True, "closing", True, "box", "KNN")

