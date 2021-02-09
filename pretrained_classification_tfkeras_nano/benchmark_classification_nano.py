# USAGE
# python benchmark_classification_nano.py --trt-graph model/mobilenet_v2_imagenet_trt_graph.pb \
#	--image sample_data/tiger.jpg

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow.contrib.tensorrt as trt
import tensorflow as tf
import numpy as np
import argparse
import time
import cv2

def loadTRTGraph(graphFile):
	# open the graph file
	with tf.gfile.GFile(graphFile, "rb") as f:
		# instantiate the GraphDef class and read the graph
		graphDef = tf.GraphDef()
		graphDef.ParseFromString(f.read())

	# return the graph    
	return graphDef

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--trt-graph", required=True, 
	help="path to the TRT graph")
ap.add_argument("-i", "--image", required=True,
	help="path to our input image")
args = vars(ap.parse_args())

# set the names of the input and output tensors
inputTensorName = "input_1:0"
outputTensorName = "Logits/Softmax:0"

# load the TRT graph
print("[INFO] loading TRT graph...")
trtGraph = loadTRTGraph(args["trt_graph"])

# instantiate the configuration, enable GPU usage growth, create
# TensorFlow session, and import the TRT graph into the session
print("[INFO] initializing TensorFlow session...")
tfConfig = tf.ConfigProto()
tfConfig.gpu_options.allow_growth = True
tfSess = tf.Session(config=tfConfig)
tf.import_graph_def(trtGraph, name="")

# grab the output tensors from the TensorFlow session
outputTensor = tfSess.graph.get_tensor_by_name(outputTensorName)

# load the input image
image = cv2.imread(args["image"])
 
# the model was trained on RGB ordered images but OpenCV represents
# images in BGR order, so swap the channels, and then resize to
# the image size we determined
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))

# add batch dimension to the image to make it compatible
# for prediction and preprocess the image
image = np.expand_dims(image, axis=0)
image = preprocess_input(image)

# run inference for 1000 times to warm up the nano
print("[INFO] warming up the nano...")
for _ in range(1000):
	_ = tfSess.run(outputTensor, 
		feed_dict={inputTensorName: image})

# initialize an empty list to store the inference times
times = []

# after the nano is warmed up start measuring the fps by 
# running inference for 20 times
for _ in range(20):
	# initialize the timer and perform inference
	start = time.time()
	_ = tfSess.run(outputTensor, 
		feed_dict={inputTensorName: image})

	# measure the time it took to perform inference and append
	# the time to the master list
	delta = (time.time() - start)
	times.append(delta)

# take the mean over all the different inference times and
# calculate the fps
timingMean = np.array(times).mean()
fps = 1 / timingMean
print("[INFO] approx. FPS: {:.4f}".format(fps))