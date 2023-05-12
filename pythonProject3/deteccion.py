import torch
import cv2
import numpy as np
import pandas
#modelo
model = torch.hub.load('ultralytics/yolov5','custom',path='C:/Users/cacyp/OneDrive/Documents/Vision Arti/model/frutas.pt') #modificar path absoluto del archivo .pt de la IA en YOLOv5


cap = cv2.VideoCapture(0)

#deteccion
while True:
    #lectura videocapt
    ret, frame = cap.read()

    #deteccion
    detect = model(frame)
    info = detect.pandas().xyxy[0]
    print(info)
    if 0 in info['class'].values:
        print("Se detectó una manzana verde.")

    cv2.imshow('Detector de Frutas', np.squeeze(detect.render()))



    t=cv2.waitKey(5)
    if t==27:
        break

cap.release()
cv2.destroyAllWindows()
