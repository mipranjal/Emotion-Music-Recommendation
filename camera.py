import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pandastable import Table, TableModel
from tensorflow.keras.preprocessing import image
import datetime
from threading import Thread
# from Spotipy import *  
import time
import pandas as pd

# Load the Haarcascade classifier for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
ds_factor = 0.6  # Downscale factor for the video feed

# Load the pre-trained emotion detection model
emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
# ... (model architecture definition)
emotion_model.load_weights('model.h5')

cv2.ocl.setUseOpenCL(False)  # Disable OpenCL support in OpenCV

# Mapping of emotion indices to emotion labels
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
# Mapping of emotion indices to music recommendation CSV file paths
music_dist = {
    0: "songs/angry.csv",
    1: "songs/disgusted.csv",
    2: "songs/fearful.csv",
    3: "songs/happy.csv",
    4: "songs/neutral.csv",
    5: "songs/sad.csv",
    6: "songs/surprised.csv"
}

# Global variables for the last video frame, webcam capture object, and detected emotion index
global last_frame1
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
show_text = [0]

# Class for calculating Frames Per Second (FPS) during streaming
class FPS:
    def __init__(self):
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        self._end = datetime.datetime.now()

    def update(self):
        self._numFrames += 1

    def elapsed(self):
        return (self._end - self._start).total_seconds()

    def fps(self):
        return self._numFrames / self.elapsed()


# Class for using another thread for video streaming to boost performance
class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


# Class for reading video stream, generating prediction, and recommendations
class VideoCamera(object):
    def get_frame(self):
        global cap1
        global df1
        cap1 = WebcamVideoStream(src=0).start()  # Start the video stream
        image = cap1.read()  # Read a frame from the video stream
        image = cv2.resize(image, (600, 500))  # Resize the frame for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
        # Detect faces in the frame
        face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
        df1 = pd.read_csv(music_dist[show_text[0]])  # Read music recommendations based on the detected emotion
        df1 = df1[['Name', 'Album', 'Artist']]
        df1 = df1.head(15)

        for (x, y, w, h) in face_rects:
            # Draw a rectangle around the detected face
            cv2.rectangle(image, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 2)
            roi_gray_frame = gray[y:y + h, x:x + w]  # Extract the region of interest (face) from the frame
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            # Preprocess the face image for emotion prediction
            prediction = emotion_model.predict(cropped_img)

            maxindex = int(np.argmax(prediction))  # Get the index of the predicted emotion
            show_text[0] = maxindex  # Update the detected emotion index
            cv2.putText(image, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                        2, cv2.LINE_AA)  # Display the detected emotion label on the frame
            df1 = music_rec()  # Fetch music recommendations based on the detected emotion

        global last_frame1
        last_frame1 = image.copy()  # Save the processed frame
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(last_frame1)
        img = np.array(img)
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes(), df1  # Return the processed frame and music recommendations


# Function for fetching music recommendations based on the detected emotion
def music_rec():
    df = pd.read_csv(music_dist[show_text[0]])
    df = df[['Name', 'Album', 'Artist']]
    df = df.head(15)
    return df

'''This script combines elements of computer vision, emotion detection, and music recommendations using TensorFlow/Keras, OpenCV, and Pandas. It leverages a threaded approach for video streaming to enhance performance.'''