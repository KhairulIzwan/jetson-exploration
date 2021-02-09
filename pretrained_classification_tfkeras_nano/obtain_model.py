#!/usr/bin/env python

# import MobileNet V2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

# grab the pretrained MobileNet V2 weights
model = MobileNetV2(weights="imagenet")

# serialize MobileNet V2 pretrained on ImageNet to disk
model.save("pretrained_models/mobilenet_v2_imagenet.h5")
