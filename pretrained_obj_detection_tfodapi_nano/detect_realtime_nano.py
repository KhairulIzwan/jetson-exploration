# USAGE
# python detect_realtime_nano.py --trt-graph model/ssd_mobilenet_v1_coco_trt.pb \
# --labels model/mscoco_label_map.pbtxt 	

# python detect_realtime_nano.py --trt-graph model/ssd_mobilenet_v1_coco_trt.pb \
# --labels model/mscoco_label_map.pbtxt --input sample_data/videos/sample_video.mp4

# import the necessary packages
from object_detection.utils.label_map_util import create_category_index_from_labelmap
from tensorflow.python.util import deprecation
from imutils.video import VideoStream
from imutils.video import FPS
import tensorflow.contrib.tensorrt as trt
import tensorflow as tf
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# load the Protobuffer network graph
def loadTRTGraph(graphFile):
	# open the graph file
	with tf.gfile.GFile(graphFile, "rb") as f:
		# instantiate the GraphDef class and read the graph
		graphDef = tf.GraphDef()
		graphDef.ParseFromString(f.read())

	# return the graph    
	return graphDef

def postProcess(H, W, boxes, scores, classes, threshold):
	# compute the (x, y)-coordinates of the bounding box 
	# for the object, convert them to integers, 
	# extract the confidence and class index
	boxes = boxes[0] * np.array([H, W, H, W])
	boxes = boxes.astype(np.int32)
	scores = scores[0]
	classes = classes[0].astype(np.int32)

	# create a mask to return boxes with confidence score 
	# above the threshold
	mask = np.where(scores >= threshold)

	# return a tuple of box co-ordinates, confidence and category
	return (boxes[mask], scores[mask], classes[mask])

def labelMap(labelPath):
	# grab the indices for the labels
	catIndex = create_category_index_from_labelmap(
		labelPath)

	# make a dictionary to map the indices to the class names
	classDict = {int(x['id']): x['name'] \
		for x in catIndex.values()}

	# calculate the total number of classes
	numClasses = max(c for c in classDict.keys()) + 1

	# add missing classes as, say,'CLS12' if any and return
	return {i: classDict.get(i, 'CLS{}'.format(i)) \
		for i in range(numClasses)}

# turn off the deprecation warnings and logs to keep
# the console clean for convenience
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--trt-graph", required=True, 
	help="path where the trt graph will be saved")
ap.add_argument("-l", "--labels", required=True,
	help="path to the labels")
ap.add_argument("-i", "--input",
	help="path to input video file")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# read the classes and then generate a set of bounding box 
# colors for each class
CLASSES_DICT = labelMap(args["labels"])
COLORS = np.random.uniform(0, 255, size=(len(CLASSES_DICT), 3))

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

# get the input and output (confidence scores, bounding boxes, 
# class labels, and number of objects detected) tensors 
# from the TensorFlow session
tfInput = tfSess.graph.get_tensor_by_name("image_tensor" + ":0")
tfScores = tfSess.graph.get_tensor_by_name("detection_scores:0")
tfBoxes = tfSess.graph.get_tensor_by_name("detection_boxes:0")
tfClasses = tfSess.graph.get_tensor_by_name("detection_classes:0")
tfNumDetections = tfSess.graph.get_tensor_by_name("num_detections:0")

# initialize variables to store frame dimensions
H = None
W = None

# if a video path was not supplied, grab a reference to the webcam
if not args.get("input", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	# vs = VideoStream(src="nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080,format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink").start()
	time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
	print("[INFO] opening video file...")
	vs = cv2.VideoCapture(args["input"])

# create a dummy image and run inference on it to warm up the nano
print("[INFO] warming up the nano...")
dummyImage = np.random.random((300, 300, 3))
dummyImage = np.expand_dims(dummyImage, axis=0)
for _ in range(1000):
	_ = tfSess.run(
		[tfBoxes, tfScores, tfClasses, tfNumDetections],
		feed_dict={tfInput: dummyImage})

# start the frames per second throughput estimator
fps = FPS().start()

# loop over the frames from the video stream
while True:
	# grab the next frame and handle if we are reading from either
	# VideoCapture or VideoStream
	frame = vs.read()
	frame = frame[1] if args.get("input", False) else frame

	# if we are viewing a video and we did not grab a frame then we
	# have reached the end of the video
	if args["input"] is not None and frame is None:
		break

	# resize the frame to have a maximum width of 500 pixels
	frame = imutils.resize(frame, width=500)

	# check to see if the frame dimensions are not set
	if W is None or H is None:
		# set the frame dimensions
		(H, W) = frame.shape[:2]

	# resize the frame to have dimensions 300x300
	blob = cv2.resize(frame, (300, 300))
	
	# run the frame through the TensorFlow session to get prediction
	(scores, boxes, classes, numDetections) = tfSess.run([tfScores, 
		tfBoxes, tfClasses, tfNumDetections], 
		feed_dict={tfInput: blob[None, ...]})

	# postprocess the predictions
	(boxes, scores, classes) = postProcess(H, W, boxes, scores, 
			classes, args["confidence"])
	
	# loop over the detections
	for (box, score, idx) in zip(boxes, scores, classes):
		# get the class name
		className = CLASSES_DICT.get(idx, 'CLS{}'.format(idx))

		# grab the coordinates of the bounding box
		(startY, startX, endY, endX) = box[0], box[1], \
			box[2], box[3]

		# draw the prediction on the frame
		cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
		y = startY - 15 if startY - 15 > 15 else startY + 15
		info = "{}: {:.2f}%".format(className,
			score * 100)
		cv2.putText(frame, info, (startX, y),\
			cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[idx], 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# if we are not using a video file, stop the camera video stream
if not args.get("input", False):
	vs.stop()

# otherwise, release the video file pointer
else:
	vs.release()

# close any open windows
cv2.destroyAllWindows()
