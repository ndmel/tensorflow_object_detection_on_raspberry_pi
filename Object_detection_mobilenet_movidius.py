######## Picamera Object Detection Using Tensorflow Classifier #########
#
# Author: Evan Juras
# Date: 4/15/18
# Description:
# This program uses a TensorFlow classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a Picamera feed.
# It draws boxes and scores around the objects of interest in each frame from
# the Picamera. It also can be used with a webcam by adding "--usbcam"
# when executing this script from the terminal.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.


# Import packages
import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
import argparse
import sys
import ast
import time
import mvnc.mvncapi as mvnc

def open_ncs_device():

    # Look for enumerated NCS device(s); quit program if none found.
    devices = mvnc.enumerate_devices()
    if len( devices ) == 0:
        print( "No devices found" )
        quit()

    # Get a handle to the first enumerated device and open it
    device = mvnc.Device( devices[0] )
    device.open()

    return device

# ---- Step 2: Load a graph file onto the NCS device -------------------------

def load_graph( device ):

    # Read the graph file into a buffer
    with open( PATH_TO_CKPT, mode='rb' ) as f:
        blob = f.read()

    # Load the graph buffer into the NCS
    graph = mvnc.Graph( PATH_TO_CKPT )
        # Set up fifos
    fifo_in, fifo_out = graph.allocate_with_fifos(device, blob)

    return graph, fifo_in, fifo_out

# Set up camera constants
#IM_WIDTH = 1280
#IM_HEIGHT = 720
#IM_WIDTH = 640   # Use smaller resolution for
#IM_HEIGHT = 480  # slightly faster framerate
MOD_F = "1.0"
MOD_RES = "224"
DES_RES = IM_WIDTH = IM_HEIGHT = int(MOD_RES)

# Select camera type (if user enters --usbcam when calling this script,
# a USB webcam will be used)
camera_type = 'picamera'
parser = argparse.ArgumentParser()
parser.add_argument('--usbcam', help='Use a USB webcam instead of picamera',
                    action='store_true')
args = parser.parse_args()
if args.usbcam:
    camera_type = 'usb'

# This is needed since the working directory is the object_detection folder.
sys.path.append('..')

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'mobilenet_v1_' + MOD_F + "_" + MOD_RES

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'graph')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','imagenet_1000_labels.txt')

# Number of classes the object detector can identify
NUM_CLASSES = 1000

## Load the label map.
# Label maps map indices to category names, so that when the convolution
# network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_str = ""
with open(PATH_TO_LABELS) as f:
    for line in f:
        label_str += line
label_map = ast.literal_eval(label_str)

#categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
#category_index = label_map_util.create_category_index(categories)

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
#detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
#detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
#num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize camera and perform object detection.
# The camera has to be set up and used differently depending on if it's a
# Picamera or USB webcam.

# I know this is ugly, but I basically copy+pasted the code for the object
# detection loop twice, and made one work for Picamera and the other work
# for USB.

device = open_ncs_device()
graph, fifo_in, fifo_out = load_graph(device)


### Picamera ###
if camera_type == 'usb':
    # Initialize USB webcam feed

    camera = cv2.VideoCapture(0)
    ret = camera.set(3,IM_WIDTH)
    ret = camera.set(4,IM_HEIGHT)

    while(True):

        t1 = cv2.getTickCount()

        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        ret, frame = camera.read()
        height, width, channels = frame.shape

        img = cv2.resize(frame, (DES_RES, DES_RES))

#        frame_expanded = np.reshape(frame, (IM_WIDTH, IM_HEIGHT, 3))

        # Perform the actual detection by running the model with the image as input
        start_t = time.time()

        graph.queue_inference_with_fifo_elem( fifo_in, fifo_out, img.astype(np.float32), None )
        output, userobj = fifo_out.read_elem()
        top_pred = output.argmax()

        duration = time.time() - start_t
        print("duration: ", duration)


        cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,0.3,(255,255,0),2,cv2.LINE_AA)
        print("pred max index: ", top_pred)
        print("pred max label: ", label_map.get(top_pred))
        print('fps: ' , frame_rate_calc)
        print()

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    camera.release()

cv2.destroyAllWindows()

