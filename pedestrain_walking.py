#Single shot object detection
import cv2
import numpy as np

model = cv2.dnn.readNetFromCaffe('C:/Users/ADMIN/Downloads/Data Sets_SDC/opencv_config_files/Day 5/MobileNetSSD_deploy.prototxt',
                                 'C:/Users/ADMIN/Downloads/Data Sets_SDC/opencv_config_files/Day 5/MobileNetSSD_deploy.caffemodel')

CONF_THR = 0.3

LABELS = {1: 'aerplane', 2: 'bicycle', 3: 'bird', 
          4: 'boat', 5: 'bottle', 6:'bus', 7: 'car', 9: 'chair',
          10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
          14: 'motorbike', 15: 'person', 16: 'pottedplant',
          17: 'sheep', 18: 'sofa', 19: 'train', 20: ' tvmonitor'}

video = cv2.VideoCapture('C:/Users/ADMIN/Downloads/180301_06_B_CityRoam_01_preview.mp4')

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    h, w = frame.shape[0:2]
    blob = cv2.dnn.blobFromImage(frame, 1/127.5, (300*w//h, 300), (127.5,127.5,127.5), False)
    model.setInput(blob)
    output = model.forward()
    
    for i in range(output.shape[2]):
        conf = output[0,0,i,2]
        if conf > CONF_THR:
            label = output[0,0,i,1]
            x0, y0, x1, y1 = (output[0,0,i,3:7]*[w,h,w,h]).astype(int)
            cv2.rectangle(frame, (x0,y0), (x1,y1), (0,255,0), 2)
            cv2.putText(frame, '{}: {:.2f}'.format(LABELS[label], conf), (x0,y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
    cv2.imshow('frame', frame)
    key = cv2.waitKey(3)
    if key == 27:
        break

cv2.destroyAllWindows()