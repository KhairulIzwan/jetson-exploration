# USAGE
# python prepare_trt_graph.py --output-path model --model ssd_mobilenet_v1_coco \
# 	--trt-graph model/ssd_mobilenet_v1_coco_trt.pb

# import the necessary packages
from tf_trt_models.detection import download_detection_model
from tf_trt_models.detection import build_detection_graph
from tensorflow.python.util import deprecation
import tensorflow.contrib.tensorrt as trt 
import tensorflow as tf
import argparse
import os

# turn off the deprecation warnings and logs to keep
# the console clean for convenience
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output-path", required=True, 
    help="path to where the downloaded detection model will be saved")
ap.add_argument("-m", "--model", required=True, 
    help="name of the model to download and build graph for")
ap.add_argument("-c", "--confidence", type=float, default=0.3,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--trt-graph", required=True, 
    help="path where the trt graph will be saved")
args = vars(ap.parse_args())

# download the specified model
print("[INFO] downloading model...")
(configPath, checkpointPath)  = download_detection_model(args["model"],
	args["output_path"])

# prepare the TensorFlow graph from the model configuration and
# checkpoint
print("[INFO] preparing the TensorFlow graph...")
(frozenGraph, inputNames, outputNames) = build_detection_graph(
    config=configPath,
    checkpoint=checkpointPath,
    score_threshold=args["confidence"],
    batch_size=1)

# create the optimized TRT graph from 
# the frozen TensorFlow graph
print("[INFO] creating TRT graph...")
trtGraph = trt.create_inference_graph(
    input_graph_def=frozenGraph,
    outputs=outputNames,
    max_batch_size=1,
    max_workspace_size_bytes=1 << 25,
    precision_mode="FP16",
    minimum_segment_size=50)

# serialize the TRT graph
print("[INFO] serializing TRT graph...")
with open(args["trt_graph"], "wb") as f:
    f.write(trtGraph.SerializeToString())