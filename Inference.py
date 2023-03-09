from retinaface import RetinaFace
from deepface import DeepFace
import os
import cv2
import glob
import shutil
import time
# it is a simple inference from one image that shows each face detected 
# and tries to find the similar images from the dataset directory 
# the dataset directory contains one folder for each face recognized and validated plus one unknown folder
cwd = os.getcwd()
img_path = "img.jpg"
unknown_path = cwd+"/datasets/unknown"
tmpImage = 'tmp.jpg'
datasetFold = cwd+"/datasets/"

def image_is_unknown(facial_img_path, facename):
    for faceFold in os.listdir(datasetFold):
        if faceFold == 'unknown' or faceFold == 'input' or faceFold == 'output':
            continue
        for imagePath in glob.glob(datasetFold+faceFold+"/*.jpg"):    
            obj = DeepFace.verify(facial_img_path, imagePath, model_name = 'ArcFace', detector_backend = 'retinaface',enforce_detection = False)
            if(obj['distance'] < 0.5):
                img = cv2.imread(facial_img_path)
                count = len(os.listdir(datasetFold+faceFold))
                facename = faceFold
                cv2.imwrite(datasetFold+faceFold+"/"+facename+'_'+str(count+1)+".jpg",img)
                return False,facename
            #print(obj['distance'])
    return True,facename

# read from input folder the frames stored
input_path = cwd+"/datasets/input/"
avgTime = 0
count = 0
while len(os.listdir(input_path)) > 0:
    
    frame_name = os.listdir(input_path)[0]
    shutil.move(input_path + os.listdir(input_path)[0], cwd +'/datasets/tmp/'+ frame_name)
    frame = cv2.imread(cwd +'/datasets/tmp/'+frame_name)
    times = time.time()
    faces = RetinaFace.detect_faces(frame)
    timee = time.time()
    print("the time to detect faces :", timee - times)
    count+=1
    avgTime += timee - times
    img = frame 
    for face in faces: 
        if len(face) == 0:
             continue 
        identity = faces[face]
        facial_area = identity["facial_area"]
        landmarks = identity["landmarks"] 
        facial_img = img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]
        cv2.imwrite(tmpImage,facial_img)
        facename = 'unknown'
        is_unknown, facename = image_is_unknown(tmpImage,facename)
        if(is_unknown):
            count = len(os.listdir(unknown_path))
            cv2.imwrite(unknown_path+'/'+frame_name +'.jpg',facial_img)
        cv2.rectangle(img, (facial_area[2], facial_area[3])
        , (facial_area[0], facial_area[1]), (255, 255, 255), 1)
        cv2.putText(img, facename, (facial_area[2], facial_area[3]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        cv2.circle(img, (int(landmarks["left_eye"][0]),int(landmarks["left_eye"][1])), 1, (0, 0, 255), -1)
        cv2.circle(img, (int(landmarks["right_eye"][0]),int(landmarks["right_eye"][1])), 1, (0, 0, 255), -1)
        cv2.circle(img, (int(landmarks["nose"][0]),int(landmarks["nose"][1])), 1, (0, 0, 255), -1)
        cv2.circle(img, (int(landmarks["mouth_left"][0]),int(landmarks["mouth_left"][1])), 1, (0, 0, 255), -1)
        cv2.circle(img, (int(landmarks["mouth_right"][0]),int(landmarks["mouth_right"][1])), 1, (0, 0, 255), -1)
        cv2.imwrite(cwd +'/datasets/output/'+frame_name,img)
    #     cv2.imshow('image preview', img)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print("the avg time to detect faces :", avgTime/count)
