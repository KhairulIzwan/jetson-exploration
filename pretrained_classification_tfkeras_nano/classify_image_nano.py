# USAGE
# python classify_image_nano.py --trt-graph model/mobilenet_v2_imagenet_trt_graph.pb --image sample_data/tiger.jpg

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
	with tf.io.gfile.GFile(graphFile, "rb") as f:
		# instantiate the GraphDef class and read the graph
		graphDef = tf.compat.v1.GraphDef()
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

# load the input image and clone it for annotation purposes
image = cv2.imread(args["image"])
output = image.copy()
 
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

# pass the image through the TensorFlow session to 
# obtain our predictions
start = time.time()
predictions = tfSess.run(outputTensor, 
	feed_dict={inputTensorName: image})
end = time.time()
print("[INFO] classification took {:.4f} seconds...".format(
	end - start))

# extract the top-5 predictions
topPredictions = decode_predictions(predictions, top=5)

# loop over the results
for (i, prediction) in enumerate(topPredictions[0]):
	# check to see if this is the top result, and if so, draw the
	# label on the image
	if i==0:
		# format the label and draw the prediction on the 
		# output image
		label = prediction[1].upper()
		text = "Label: {}, {:.2f}%".format(label, 
			prediction[2] * 100)
		cv2.putText(output, text, (10, 30),  
			cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
	
	# display the classification result to the terminal
	print("{}. {}: {:.2f}%".format(i + 1, prediction[1].upper(),
		prediction[2] * 100))

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)
