import cv2
import sys
import numpy as np
import matplotlib.pyplot as  plt

from scipy import signal
from sklearn.decomposition import FastICA

import warnings
import random

# FILES
CASCADE_PATH = "haarcascade_frontalface_default.xml"
VIDEO_DIR = "/VIDEOS"
DEFAULT_VIDEO = "android-1.mp4"
RESULTS_SAVE_DIR = "/results"

# face_size
MIN_FACE_SIZE = 100

# VIDOE
FPS = 0

# WINDOW
WINDOW_TIME_SEC = 30
WINDOW_SIZE = int(np.ceil(WINDOW_TIME_SEC * FPS))

# ROI
EYE_LOWER_FRAC = 0.25

#
ADD_BOX_ERROR = False
BOX_ERROR_MAX = 0.5

# segmentation
SEGMENTATION_WIDTH_FRACTION = 1.2
SEGMENTATION_HEIGHT_FRACTION = 0.8

# GRABCUT
GRABCUT_ITER = 5

# HEART RATE
MIN_HR_BPM = 45.0
MAX_HR_BPM = 240.0

SEC_PER_MIN = 60


def segment(image, faceBox):
    mask = np.zeros(image.shape[:2], np.uint8)
    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(image, mask, faceBox, bgModel, fgModel, GRABCUT_ITER, cv2.GC_INIT_WITH_RECT)
    backgroundMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), True, False).astype('uint8')
    backgroundMask = np.broadcast_to(backgroundMask[:, :, np.newaxis], np.shape(image))
    return backgroundMask


def getROI(image, faceBox):
    # USE segmentation
    widthFrac = SEGMENTATION_WIDTH_FRACTION
    heightFrac = SEGMENTATION_HEIGHT_FRACTION
    # Bounding Box
    (x, y, w, h) = faceBox
    widthOffset = int((1 - widthFrac) * w / 2)
    heightOffset = int((1 - heightFrac) * h / 2)
    faceBoxAdjusted = (x + widthOffset, y + heightOffset, int(widthFrac * w), int(heightFrac * h))

    backgroundMask = segment(image, faceBoxAdjusted)

    (x, y, w, h) = faceBox


    # grab FOREHEAD only
    #backgroundMask[y + h * EYE_LOWER_FRAC:, :] = True

    roi = np.ma.array(image, mask=backgroundMask)
    return roi


def distance(roi1, roi2):
    return sum((roi1[i] - roi2[i]) ** 2 for i in range(len(roi1)))


def getBestROI(frame, faceCascade, previousFacebox):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1,
                                         minNeighbors=5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),)

    #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    roi = None
    faceBox = None

    if len(faces) == 0:
        faceBox = previousFacebox

    elif len(faces) > 1:
        if previousFacebox is not None:
            minDist = float("inf")
            for face in faces:
                if distance(previousFacebox, face) < minDist:
                    faceBox = face
        else:
            # choose largest facebox
            maxArea = 0
            for face in faces:
                if (face[2] * face[3]) > maxArea:
                    faceBox = face
    else:
        faceBox = faces[0]

    if faceBox is not None:
        if ADD_BOX_ERROR:
            noise = []
            for i in range(4):
                noise.append(random.uniform(-BOX_ERROR_MAX, BOX_ERROR_MAX))
            (x, y, w, h) = faceBox
            x1 = x + int(noise[0] * w)
            y1 = y + int(noise[1] * h)
            x2 = x + w + int(noise[2] * w)
            y2 = y + h + int(noise[3] * h)

            faceBox = (x1, y1, x2 - x1, y2 - y1)

        roi = getROI(frame, faceBox)

    return faceBox, roi


# grab fps from video
def plotSignals(signals, label):
    seconds = np.arange(0, WINDOW_TIME_SEC, 1.0 / FPS)
    colors = ["r", "g", "b"]

    fig = plt.figure()
    fig.patch.set_facecolor('white')
    for i in range(3):
        plt.plot(seconds, signals[:, i], colors[i])
    plt.xlabel('Time (sec)', fontsize=17)
    plt.ylabel(label, fontsize=17)
    plt.tick_params(axis='x', labelsize=17)
    plt.tick_params(axis='y', labelsize=17)
    plt.show()


def plotSpectrum(freqs, powerSpec):
    idx = np.argsort(freqs)
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    for i in range(3):
        plt.plot(freqs[idx], powerSpec[idx, i])
    plt.xlabel("Frequency (Hz)", fontsize=17)
    plt.ylabel("Power", fontsize=17)
    plt.tick_params(axis='x', labelsize=17)
    plt.tick_params(axis='y', labelsize=17)
    plt.xlim([0.75, 4])
    plt.show()


def getHeartRate(window, lastHR):
    # Normalize across the window to have zero-mean and unit variance
    mean = np.mean(window, axis=0)
    std = np.std(window, axis=0)
    normalized = (window - mean) / std

    # Separate into three source signals using ICA
    ica = FastICA()
    srcSig = ica.fit_transform(normalized)

    # Find power spectrum
    powerSpec = np.abs(np.fft.fft(srcSig, axis=0)) ** 2
    freqs = np.fft.fftfreq(WINDOW_SIZE, 1.0 / FPS)

    # Find heart rate
    maxPwrSrc = np.max(powerSpec, axis=1)
    validIdx = np.where((freqs >= MIN_HR_BPM / SEC_PER_MIN) & (freqs <= MAX_HR_BPM / SEC_PER_MIN))
    validPwr = maxPwrSrc[validIdx]
    validFreqs = freqs[validIdx]
    maxPwrIdx = np.argmax(validPwr)
    hr = validFreqs[maxPwrIdx]
    print("heartrate" + str(hr))

    #plotting
    plotSignals(normalized, "Normalized color intensity")
    plotSignals(srcSig, "Source signal strength")
    plotSpectrum(freqs, powerSpec)

    return hr


# try:
#    videoFile = sys.argv[1]
# except:
#    videoFile = DEFAULT_VIDEO some issue  in  Opencv-python ffmpeg windows /ubuntu

#videoFile = 'android-1.mp4'
video = cv2.VideoCapture(r"D:\pythonProject\Heart_rate_fastICA\VIDEOS\deep_fake.mp4")
print(video.isOpened())
faceCascade = cv2.CascadeClassifier(CASCADE_PATH)

FPS = video.get(cv2.CAP_PROP_FPS)
print(FPS)
WINDOW_SIZE = int(np.ceil(WINDOW_TIME_SEC * FPS))
colorSig = []  # Will store the average RGB color values in each frame's ROI
heartRates = []  # Will store the heart rate calculated every 1 second
previousFaceBox = None
while True:
    # Capture frame-by-frame
    ret, frame = video.read()
    if not ret:
        print("Reading Complete")
        break

    previousFaceBox, roi = getBestROI(frame, faceCascade, previousFaceBox)

    if (roi is not None) and (np.size(roi) > 0):
        colorChannels = roi.reshape(-1, roi.shape[-1])
        avgColor = colorChannels.mean(axis=0)
        colorSig.append(avgColor)

    # Calculate heart rate every one second (once have 30-second of data)

    if (len(colorSig) >= WINDOW_SIZE) and (len(colorSig) % np.ceil(FPS) == 0):
        windowStart = len(colorSig) - WINDOW_SIZE
        window = colorSig[windowStart: windowStart + WINDOW_SIZE]
        lastHR = heartRates[-1] if len(heartRates) > 0 else None
        heartRates.append(getHeartRate(window, lastHR))
    if np.ma.is_masked(roi):
        roi = np.where(roi.mask == True, 0, roi)
    cv2.imshow('ROI', roi)
    cv2.waitKey(1)

print(heartRates)
#print(videoFile)
#filename = RESULTS_SAVE_DIR + videoFile[0:-4]

#np.save(filename, heartRates)
video.release()
cv2.destroyAllWindows()
