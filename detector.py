import cv2
import numpy as np

subjects = ["", "random1", "random2", "Barack Obama" , "", ""]

imagePath = 'd.jpg'
#cascPath = sys.argv[2]
# Create the haar cascade
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

rec = cv2.face.LBPHFaceRecognizer_create();
rec.read("C:\\Users\\Bilal\\Desktop\\Python (Face Recognition)\\Face Recognition\\trained-data\\trainingData.yml")
id=0 
#font = cv2.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Detect faces in the image
face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_default.xml')

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
   


print "Found {0} faces!".format(len(faces))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    face, rect = gray[y:y+w, x:x+h], (x, y, w, h)
    label= rec.predict(face)
    print label
    #id,conf = rec.predict(gray[y:y+w, x:x+h])
    #cv2.cv.PutText(cv2.cv.fromarray(image), str(id), (x,y+h), font, 255);
    if (label[1]<70):
        cv2.putText(image, str(subjects[label[0]]), (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        

cv2.imshow("Faces found" ,image)
cv2.waitKey(0)
