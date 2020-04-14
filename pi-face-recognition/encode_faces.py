from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os
import time
import sqlite3
import numpySQLite
import datetime


db = sqlite3.connect('encodings.db', detect_types=sqlite3.PARSE_DECLTYPES)
cursor = db.cursor()

maxDistance = 0.3

cursor.execute('''
    CREATE TABLE if not exists faces(id INTEGER PRIMARY KEY,
               name TEXT, encoding ARRAY, date TIMESTAMP)
''')

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required=True,
    help="Name of person")
args = vars(ap.parse_args())
name = args["name"]
pics = []
encodings = []

cursor.execute('''SELECT encoding  FROM faces WHERE name = ?''',(name,))
faces_databaseSQL = cursor.fetchall()

if len(faces_databaseSQL) > 0:
    print("Ya existe una entrada para esta persona, ¿desea borrarla? (Si o No)")
    answer = input()
    if answer == "Si":
        cursor.execute('''DELETE FROM faces WHERE name =?''',(name,))
        db.commit()
        quit()
    else:
        quit()

imagePaths = list(paths.list_images('database/'+name))

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))

    #start = time.time()
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    box = face_recognition.face_locations(rgb,model="hog")
    if(len(box)!=1):
        print("no hay una cara (se detectó ninguna o múltiples caras)")
        continue  
    encodings.append(face_recognition.face_encodings(rgb, box)[0])
    pics.append(imagePath)
    #end = time.time()
    #print("{}: {}".format("Tiempo total",end - start))
cambiarFotos = False
i=0
for encoding1 in encodings:
    j=0
    for encoding2 in encodings:
        distance=face_recognition.face_distance([encoding1,],encoding2)
        if (distance>maxDistance):
            print("Distancia:{} No coinciden {} y {}".format(distance,pics[i],pics[j]))
            cambiarFotos = True
        j+=1
    i+=1
if(not cambiarFotos):
    encodingProm = encodings[0]
    for encoding in encodings[1:]:
        encodingProm += encoding
    encodingProm/=len(encodings)
    
    cursor.execute('''INSERT INTO faces(name, encoding, date) VALUES(?,?,?)''',
                   (name, encodingProm, datetime.datetime.now().strftime("%y-%m-%d")))
    db.commit()

    #print(face_recognition.face_distance(encodings,encodingProm))

        

