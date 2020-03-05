import cv2
import numpy as np
import faceRecognition as fr
import os

# =============================================================================
# test_img = cv2.imread('C:/Users/vivek/Desktop/Face_recognition/test_img/Vivek.jpg')
# faces_detected, gray_img = fr.faceDetection(test_img)
# print("faces detected: ", faces_detected)
# =============================================================================

# =============================================================================
# for face in faces_detected:
#     x,y,w,h = face
#     cv2.rectangle(test_img, (x,y), (x+w, y+h), (255,0,0), thickness=5)
#     cv2.putText(test_img,'vivek', (x,y), cv2.FONT_HERSHEY_DUPLEX, 2, (255,0,0), 4)
#  
# =============================================================================

# =============================================================================
# faces, faceID = fr.labels_for_training_data('C:/Users/vivek/Desktop/Face_recognition/training_imgs')
# face_recognizer = fr.train_classifier(faces, faceID)
# face_recognizer.save('trainingData.yml')
# =============================================================================

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('C:/Users/vivek/Desktop/Face_recognition/trainingData.yml')
name = {0:'Unknown', 1:'Vivek'}

cap = cv2.VideoCapture(0)
while True:
    ret, test_img = cap.read()
    faces_detected, gray_img = fr.faceDetection(test_img)
    
# =============================================================================
#     for (x,y,w,h) in faces_detected:
#         cv2.rectangle(test_img, (x,y), (x+w, y+h), (0,255,0), 2)
#     
#     resized_img = cv2.resize(test_img, (1000,700))
#     cv2.imshow('face detection ', resized_img)
#     cv2.waitKey(10)
# =============================================================================
    for face in faces_detected:
        x,y,w,h = face
        roi_gray = gray_img[y:y+w, x:x+h]
        label, confidence = face_recognizer.predict(roi_gray)
        print('confidence: ', confidence)
        print('label: ', label)
        if confidence<100 :
            fr.draw_rect(test_img, face)
            predicted_name = name[label]
            fr.put_text(test_img, predicted_name, x, y)
    
    resized_img = cv2.resize(test_img, (1000,700))
    cv2.imshow('face detection:' , resized_img)
    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

