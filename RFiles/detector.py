import cv2
import numpy as np


imagePath = 'd.jpg'
#cascPath = sys.argv[2]
# Create the haar cascade
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

rec = cv2.createLBPHFaceRecognizer();
rec.load("recognizer\\trainingData.yml")
id=0 
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)


print "Found {0} faces!".format(len(faces))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    id,conf = rec.predict(gray[y:y+h, x:x+w])
    cv2.cv.PutText(cv2.cv.fromarray(image), str(id), (x,y+h), font, 255);



cv2.imshow("Faces found" ,image)
cv2.waitKey(0)
