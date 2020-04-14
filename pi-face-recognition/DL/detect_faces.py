# USAGE
# python detect_faces.py --image rooster.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
import numpy as np
import argparse
import face_recognition
import cv2
import pickle
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
ap.add_argument("-p", "--prototxt", default="deploy.prototxt.txt",
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default = "res10_300x300_ssd_iter_140000.caffemodel",
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument("-e", "--encodings", required=True,
help="path to serialized db of facial encodings")
args = vars(ap.parse_args())

net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
start = time.time()

data = pickle.loads(open(args["encodings"], "rb").read())
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
    (300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()

for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > args["confidence"]:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        boxEnc = [(startX, startY, endX, endY)]
        encodings = face_recognition.face_encodings(rgb,boxEnc)
        for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"],
            encoding)
            if True in matches:
                esta=True
end = time.time()
print(end - start)
