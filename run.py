from Detector import *

#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz"  #fastest for video obj
#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz" #best for image obj
modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_640x640_coco17_tpu-8.tar.gz" #best for video obj #fps < 1
#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz" #best for image obj #taking long time to load model

#Sumary: ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz is the best model for video object detection (fastest)
#Sumary: efficientdet_d4_coco17_tpu-32.tar.gz is the best model for image object detection (best accuracy)
#Sumary: faster_rcnn_resnet101_v1_640x640_coco17_tpu-8.tar.gz is the best model for video object detection (best accuracy) but fps < 1
#Sumary: mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz is the best model for image object detection (best accuracy) but taking long time to load model

imagePath = "object/icebear.jpg"
videoPath = "object/data12.mp4"
threshold = 0.25
classFile = 'coco.names'
detector = Detector()
detector.readClasses(classFile)
detector.downloadModel(modelURL)
detector.loadModel()
detector.predictImage(imagePath,threshold)
#detector.predictVideo(videoPath,threshold)
#detector.predictWebcam(threshold)