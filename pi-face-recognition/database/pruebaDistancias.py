import face_recognition
import imutils
import time
import cv2
import math

detector = cv2.CascadeClassifier("../haarcascade_frontalface_default.xml")


frame1 = cv2.imread("jose/00000.png")
frame2 = cv2.imread("jose/00001.png")


frame1 = imutils.resize(frame1, width=500)
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
rgb1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
rects1 = detector.detectMultiScale(gray1, scaleFactor=1.1,minNeighbors=5, minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE)
if(len(rects1) != 1):
    print("no unica cara")
box1 = [(y, x + w, y + h, x) for (x, y, w, h) in rects1]
encoding1 = face_recognition.face_encodings(rgb1, box1, num_jitters=1)[0] 


frame2 = imutils.resize(frame2, width=500)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
rects2 = detector.detectMultiScale(gray2, scaleFactor=1.1,minNeighbors=5, minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE)
if(len(rects2) != 1):
    print("no unica cara")
box2 = [(y, x + w, y + h, x) for (x, y, w, h) in rects2]
encoding2 = face_recognition.face_encodings(rgb2, box2, num_jitters=1)[0] 

distance = face_recognition.face_distance([encoding1,], encoding2)
print(distance)

sum = 0
for i in range(0,len(encoding1)):
    sum+=(encoding1[i]-encoding2[i])**2
print(math.sqrt(sum))
    
