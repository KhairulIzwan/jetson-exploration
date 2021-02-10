# jetson-exploration

## Table of Contents
1. Jetson
	1. Nano

## Jetson Nano SBC
### Jetson Nano Specifications and Horsepower
- 128-core NVIDIA Maxwell GPU
- Quad-Core ARM Cortex-A57 MPCore CPU
- 472 GFLOPs
- 4GB 64-bit LPDDR4 25.6GB/s (not expandable)
- 1x microSD memory card slot
- 4x USB 3.0
- 1 HDMI
- 1 display port
- Gigabit Ethernet

*While the NVIDIA Jetson Nano has a CUDA-capable GPU, the GPU is not capable of*
*being used for training models. You must use a fully capable machine with*
*enough memory and ideally a full-scale deep learning GPU for training your*
models.*

*Instead, the Jetson shines for performing inference without consuming as much* 
*power as a full scale deep learning rig would.*

### Pre-trained Image Classifiers with Jetson Nano
- The Jetson Nano can work with models built with other deep learning frameworks
such as mxnet, PyTorch, and Caffe, but models must be optimized with a tool 
known as TensorRT (TRT).

![TENSORRT](https://github.com/KhairulIzwan/jetson-exploration/blob/main/img/TensorRT.png)

![KERASTOTENSORRT](https://github.com/KhairulIzwan/jetson-exploration/blob/main/img/KerasToTRT.png)

#### TensoRT
- TensorRT is a framework developed by NVIDIA to optimize deep learning models
for NVIDIA devices.
- Optimized models will run faster and more efficiently on these devices.
- TensorRT optimizes a model from the following perspectives:
	- Eliminates unused layers
	- Combines and merge layers including convolution, bias, and ReLU
	- Combines operations with the same source tensor and similar parameters
	- Eliminates no-op equivalent operations
	- Tunes and calibrates weights
-  The result is a specialized TensorRT Graph that is both memory and 
computationally efficient for the target device.

**Can TensorFlow/Keras models be used with TensorRT?**

Keras models cannot be used directly with TensorRT. First the model must be 
converted into a Frozen Graph. At that point we can use TensorRT to optimize the
model. For more refer [TensorRT-Developer-Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)

**Preparing Keras/TensorFlow Models for the Jetson Nano**
- Download the h5 files

**Converting a Keras/TensorFlow model to TRT graph**
- Convert h5 file to pb file

**Performing Inference using the TRT Graph**

**Improving Jetson Classification Speed By Multiple Orders of Magnitude**

## References
1. [Remote Access to Jetson Nano](https://forums.developer.nvidia.com/t/remote-access-to-jetson-nano/74142)
2. [Jetson Nano -Remote VNC Access](https://medium.com/@bharathsudharsan023/jetson-nano-remote-vnc-access-d1e71c82492b)
3. [Getting Started with the NVIDIA Jetson Nano Developer Kit](https://www.hackster.io/news/getting-started-with-the-nvidia-jetson-nano-developer-kit-43aa7c298797)
4. [How to Use SCP Command to Securely Transfer Files](https://linuxize.com/post/how-to-use-scp-command-to-securely-transfer-files/)
5. [Running samples on the jetson nano](https://forums.developer.nvidia.com/t/running-samples-on-the-jetson-nano/73461)

**Not Working**

1. [How to install Teamviewer on a Jetson Nano](https://medium.com/@hmurari/how-to-install-teamviewer-on-a-jetson-nano-38080f87f039)
2. [The problem about Skipping acquire of configured file](https://forums.developer.nvidia.com/t/the-problem-about-skipping-acquire-of-configured-file/122395)
