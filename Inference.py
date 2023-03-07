from retinaface import RetinaFace
from deepface import DeepFace
import os
import cv2
import glob
# it is a simple inference from one image that shows each face detected 
# and tries to find the similar images from the dataset directory 
# the dataset directory contains one folder for each face recognized and validated plus one unknown folder
cwd = os.getcwd()
img_path = "img.jpg"
unknown_path = cwd+"/datasets/unknown"
tmpImage = 'tmp.jpg'
datasetFold = cwd+"/datasets/"

def image_is_unknown(facial_img_path):
    for faceFold in os.listdir(datasetFold):
        if faceFold == 'unknown':
            continue
        for imagePath in glob.glob(datasetFold+faceFold+"/*.jpg"):    
            obj = DeepFace.verify(facial_img_path, imagePath, model_name = 'ArcFace', detector_backend = 'retinaface',enforce_detection = False)
            if(obj['distance'] < 0.5):
                img = cv2.imread(facial_img_path)
                count = len(os.listdir(datasetFold+faceFold))
                cv2.imwrite(datasetFold+faceFold+"/"+faceFold+str(count+1)+".jpg",img)
                return False
            print(obj['distance'])
    return True

fourcc = cv2.VideoWriter_fourcc(*'H264')
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    faces = RetinaFace.detect_faces(frame)
    img = frame 
    for face in faces: 
        if len(face) == 0:
             continue 
        identity = faces[face]
        facial_area = identity["facial_area"]
        landmarks = identity["landmarks"] 
        cv2.rectangle(img, (facial_area[2], facial_area[3])
        , (facial_area[0], facial_area[1]), (255, 255, 255), 1)
        
        facial_img = img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]
        cv2.imwrite(tmpImage,facial_img)
        if(image_is_unknown(tmpImage)):
            count = len(os.listdir(unknown_path))
            cv2.imwrite(unknown_path+'/unknown'+str(count+1)+'.jpg',facial_img)

        cv2.circle(img, (int(landmarks["left_eye"][0]),int(landmarks["left_eye"][1])), 1, (0, 0, 255), -1)
        cv2.circle(img, (int(landmarks["right_eye"][0]),int(landmarks["right_eye"][1])), 1, (0, 0, 255), -1)
        cv2.circle(img, (int(landmarks["nose"][0]),int(landmarks["nose"][1])), 1, (0, 0, 255), -1)
        cv2.circle(img, (int(landmarks["mouth_left"][0]),int(landmarks["mouth_left"][1])), 1, (0, 0, 255), -1)
        cv2.circle(img, (int(landmarks["mouth_right"][0]),int(landmarks["mouth_right"][1])), 1, (0, 0, 255), -1)
        cv2.imshow('image preview', img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
