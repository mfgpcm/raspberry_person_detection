# Person Detection on a Raspberry Pi

This script detects persons in a RTSP video stream and sends an email if one is detected.
We use OpenCV2 for the video processing and Tensorflow Lite for the DNN detecting persons.
It runs on a Raspberry Pi Mini 2 wifi with Debian Bullseye.
The script is based on the [TensorFlow Lite Python object detection example with Raspberry Pi](https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/raspberry_pi/detect.py.)

# Setup

1. Clone on the raspberry
1. Download the pretrained [Mobilenetv1](https://www.kaggle.com/models/tensorflow/ssd-mobilenet-v1/tfLite/metadata/1?lite-format=tflite) model or [Yolov5](https://www.kaggle.com/models/kaggle/yolo-v5/tfLite/tflite-tflite-model) model from Kaggle and unzip them in ```./model```
1. Run the ```setup_raspberry.sh```
1. ```cp detect_person_sample_config.json detect_person_config.json``` and modify content as you wish
1. Run the script via ```python3 detect_person_dnn.py -d -s```
1. *Optional* Follow the steps in ```detect_person.service``` to setup and execute the script as a service, so it runs automatically after a reset
