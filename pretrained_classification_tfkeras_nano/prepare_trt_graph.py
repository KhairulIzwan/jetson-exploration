# USAGE
# python prepare_trt_graph.py --weights ../pretrained_models/mobilenet_v2_imagenet.h5 \
# 	--trt-graph model/mobilenet_v2_imagenet_trt_graph.pb

# import the necessary packages
from tensorflow.graph_util import convert_variables_to_constants
from tensorflow.graph_util import remove_training_nodes
from tensorflow.keras.models import load_model
import tensorflow.contrib.tensorrt as trt
import tensorflow as tf
import argparse
import os

def freezeGraph(graph, session, outputNames):
	# begin with graph's default context
	with graph.as_default():
		# remove the training nodes from the graph and convert
		# the variables in the graph to constants
		graphDefInf = remove_training_nodes(graph.as_graph_def())
		graphDefFrozen = convert_variables_to_constants(session, 
			graphDefInf, outputNames)

		# return the frozen graph
		return graphDefFrozen

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True, 
	help="path to the pretrained MobileNet V2 weights")
ap.add_argument("-t", "--trt-graph", required=True, 
	help="path to the TRT Graph file")
args = vars(ap.parse_args())

# ensure network model is trained further, load the model and
# extract the underlying TensorFlow session
tf.keras.backend.set_learning_phase(0)
model = load_model(args["weights"])
session = tf.keras.backend.get_session()

# grab the output operation names from model
outputNames = [t.op.name for t in model.outputs]

# freeze the session graph
print("[INFO] freezing network...")
frozenGraph = freezeGraph(session.graph, session, outputNames)

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
