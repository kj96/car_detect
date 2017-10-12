import numpy as np
import cv2
from skimage.feature import hog
import random
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob

car_img = glob.glob('D:/Online-compete/interview/train/cars/**/*.png', recursive=True)
ncar_img = glob.glob('D:/Online-compete/interview/train/non-cars/**/*.png', recursive=True)

pix_per_cell = 8
cell_per_block = 2
orient = 9

def get_hog_features(img):
    features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                   visualise=False, feature_vector=True)
    return features

def extract_features(imgs):
    features = []
    for file in imgs:
        image = mpimg.imread(file)
        feature_image = np.copy(image)
        hog_features = get_hog_features(feature_image[:,:,2])
        features.append(hog_features)
    return features

def single_img_features(img):
    img_features = []
    feature_image = np.copy(img)
    hog_features = get_hog_features(feature_image[:,:,2])
    img_features.append(hog_features)
    return np.concatenate(img_features)


car_hog_features = extract_features(car_img)

ncar_hog_features = extract_features(ncar_img)

# createe array of feature vectors
X = np.vstack((car_hog_features, ncar_hog_features)).astype(np.float64)
X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)

# create array of labels
y = np.hstack((np.ones(len(car_hog_features)), np.zeros(len(ncar_hog_features))))
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)


# dfine classifier & fit training data
svc = LinearSVC()
svc.fit(X_train, y_train)


def draw_boxes(img, bboxes):
    draw_img = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(draw_img, bbox[0], bbox[1], (0, 255, 0), 3)
    return draw_img

# Window size (x and y dimensions), and overlap fraction (for both x and y)
def slide_window(img,x_start_stop=[None, None], y_start_stop=[None, None],xy_window=(64, 64)):

    xy_overlap=(0.65, 0.65)
    # calculate area of ROI
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    # calculate number of pixels in each step
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))

    # calculate number of windows
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)


    window_list = []

    # Loop through to find x and y window positions
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            window_list.append(((startx, starty), (endx, endy)))
    return window_list

def search_windows(img, windows):

    on_windows = []

    for window in windows:
        # extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))

        # extract features for that single window
        features = single_img_features(test_img)

        # scale extracted features
        scaler = StandardScaler().fit(features)
        test_features = scaler.transform(np.array(features).reshape(1, -1))

        prediction = svc.predict(test_features)

        # if positive then save the window
        if prediction == 1:
            on_windows.append(window)
    return on_windows

def add_heat(heatmap, bbox_list):
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap

def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap

def draw_labeled_bboxes(img, labels):
    for car_number in range(1, labels[1]+1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    return img

import matplotlib.image as mpimg
from scipy.ndimage.measurements import label

image = mpimg.imread('D:/Online-compete/interview/test_images/q.jpg')

def detect_car(testCar):
    draw_image = np.copy(testCar)

    heat = np.zeros_like(testCar[:,:,0]).astype(np.float)

    windows = slide_window(testCar, x_start_stop=[0, 1280], y_start_stop=[380, 680], xy_window=(128, 128))

    hot_windows = search_windows(testCar, windows)
    add_heat(heat, hot_windows)

    windows = slide_window(testCar, x_start_stop=[0, 1280], y_start_stop=[390, 620], xy_window=(96, 96))


    hot_windows = search_windows(testCar, windows)
    add_heat(heat, hot_windows)

    windows = slide_window(testCar,x_start_stop=[0, 1280], y_start_stop=[390, 560], xy_window=(72, 72))

    hot_windows = search_windows(testCar, windows)
    add_heat(heat, hot_windows)


    windows = slide_window(testCar,x_start_stop=[0, 1280], y_start_stop=[390, 500], xy_window=(64, 64))

    hot_windows = search_windows(testCar, windows)
    add_heat(heat, hot_windows)
    heat = apply_threshold(heat, 2)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    draw_labeled_bboxes(draw_image, labels)

    return draw_image

detected_car_image = detect_car(image)

cv2.imwrite("as.png",detected_car_image)

myclassifier = "D:/Online-compete/interview/mycarclassifier.pkl"
joblib.dump(svc, myclassifier)
print("DONE!")
