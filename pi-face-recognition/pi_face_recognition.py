# USAGE
# python pi_face_recognition.py --cascade haarcascade_frontalface_default.xml --encodings encodings.pickle

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import sqlite3
import os
import numpySQLite

maxTries = 5
distanceV = 0.5

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
db = sqlite3.connect('encodings.db', detect_types=sqlite3.PARSE_DECLTYPES)
cursor = db.cursor()

reconociendo = False
while True:
    if not reconociendo:
        print("Ingresar nombre:")
        nameIn = input()
        count = 0
    reconociendo = True
    
    cursor.execute('''SELECT encoding  FROM faces WHERE name = ?''',(nameIn,))
    faces_databaseSQL = cursor.fetchall()
    faces_database = [i[0] for i in faces_databaseSQL]
    if len(faces_database) < 1:
        print("No se encuentra en la base de datos")
        reconociendo = False
        continue
    print("Esperando detectar una cara...")

    while reconociendo:
        startT = time.time()
        
        frame = vs.read()
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rects = detector.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=5, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)
        if(len(rects) != 1):
            #print("no unica cara")
            continue
        box = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
        start = time.time()
        encoding = face_recognition.face_encodings(rgb, box, num_jitters=1)[0] 
        end = time.time()
        print("{}: {}".format("Tiempo codificando",end - start))
        matches = face_recognition.compare_faces(faces_database, encoding, tolerance=distanceV)
        esta=False
        count += 1
        coinc=0
        print(matches)
        print("{}: {}".format("Tiempo total",end - startT))
        for match in matches:
            if(match):
                coinc+=1
        if coinc/len(matches) > 0.5:
            esta=True
            print("\nSí es la persona\n")
            reconociendo = False
        elif count >= maxTries:
            print("\nNo es la persona\n")
            reconociendo = False
        else:
            print("\nNo se reconoció, itento {}/{}\n".format(count,maxTries))
            print("Esperando reconocer de nuevo...")

              
              
              
              
              
              
          
