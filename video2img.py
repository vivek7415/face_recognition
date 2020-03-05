import cv2
import numpy as np
import os

count = 0

# for path, subdirnames, filenames in os.walk(C:/Users/vivek/Desktop/Face_recognition/training_img):
#     for filename in filenames:

cap = cv2.VideoCapture(0)

count = 0

while True:
    ret, test_img = cap.read()
    if not ret:
        continue
    cv2.imwrite('frame%d.jpg'%count, test_img)
    count += 1
    cv2.imshow('face capturing', test_img)
    if cv2.waitKey(10) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
