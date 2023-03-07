import cv2
import os
cwd = os.getcwd()
input_path = cwd+"/datasets/input"

fourcc = cv2.VideoWriter_fourcc(*'H264')
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    #store the frame 
    cv2.imwrite('input'+count+'.png',frame)
    