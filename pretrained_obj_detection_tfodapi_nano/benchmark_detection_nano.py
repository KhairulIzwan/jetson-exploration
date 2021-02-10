# USAGE
# python benchmark_detection_nano.py \
#	--trt-graph model/ssd_mobilenet_v1_coco_trt.pb \
# 	--image sample_data/images/huskies.jpg

# import the necessary packages
from tensorflow.python.util import deprecation
import tensorflow.contrib.tensorrt as trt
import tensorflow as tf
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def loadTRTGraph(graphFile):
	# open the graph file
	with tf.gfile.GFile(graphFile, "rb") as f:
		# instantiate the GraphDef class and read the graph
		graphDef = tf.GraphDef()
		graphDef.ParseFromString(f.read())

	# return the graph    
	return graphDef

# turn off the deprecation warnings and logs to 
# keep the console clean for convenience
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--trt-graph", required=True,
	help="path for the trt graph of the detector model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability used to filter weak detections")
args = vars(ap.parse_args())

# load the TRT graph
print("[INFO] loading TRT graph...")
trtGraph = loadTRTGraph(args["trt_graph"])

# instantiate the ConfigProto class, enable GPU usage growth, create
# TensorFlow session, and import the TRT graph into the session
print("[INFO] initializing TensorFlow session...")
tfConfig = tf.ConfigProto()
tfConfig.gpu_options.allow_growth = True
tfSess = tf.Session(config=tfConfig)
tf.import_graph_def(trtGraph, name="")

# grab a reference to both the image and boxes tensors
imageTensor = tfSess.graph.get_tensor_by_name("image_tensor:0")
boxesTensor = tfSess.graph.get_tensor_by_name("detection_boxes:0")

# for each bounding box we would like to know the score
# (i.e., probability) and class label
scoresTensor = tfSess.graph.get_tensor_by_name("detection_scores:0")
classesTensor = tfSess.graph.get_tensor_by_name("detection_classes:0")
numDetections = tfSess.graph.get_tensor_by_name("num_detections:0")

# load the image from disk and resize it to have a maximum width of
# 500 pixels
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)

# prepare the image for detection
(H, W) = image.shape[:2]
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = np.expand_dims(image, axis=0)

# warm up the nano by performing inference for 1000 times
print("[INFO] warming up the nano...")
for _ in range(1000):
	_ = tfSess.run(
		[boxesTensor, scoresTensor, classesTensor, numDetections],
		feed_dict={imageTensor: image})

# initialize an empty list to store the inference times
times = []

# now that the nano is warmed up perform inference for another 20 
# times and record the FPS information
for _ in range(20):
	# initialize the timer and perform inference
	start = time.time()
	_ = tfSess.run(
		[boxesTensor, scoresTensor, classesTensor, numDetections],
		feed_dict={imageTensor: image})

	# measure the time it took to perform inference and append
	# the time to the master list
	delta = (time.time() - start)
	times.append(delta)

# take the mean over all the different inference times and
# calculate the fps
timingMean = np.array(times).mean()
fps = 1 / timingMean
print("[INFO] approx. FPS: {:.4f}".format(fps))