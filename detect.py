import numpy as np
import cv2
import datetime
from openpyxl import Workbook
import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="123",
  database="detectedVideosDatabase"
)
def insertData(cursor, videoName, videoPath):
    sql = "insert into detectedVideos (name, path) values(%s, %s)"
    val = (videoName, videoPath)
    cursor.execute(sql, val)
    mydb.commit()
mycursor = mydb.cursor()

confidenceThreshold = 0.5
NMSThreshold = 0.3
modelConfiguration = "./yolo-coco/yolov3.cfg"
modelWeights = "./yolo-coco/yolov3.weights"
labelsPath = "./yolo-coco/coco.names"
labels = open(labelsPath).read().strip().split('\n')
np.random.seed(10)
COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
outputLayer = net.getLayerNames()
outputLayer = [outputLayer[i[0] - 1] for i in net.getUnconnectedOutLayers()]
date = datetime.datetime.now()
video_capture = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter(str(date.strftime("%Y-%m-%d")) + '.avi', fourcc, 5.0, (640,480))
videoPath = "/home/utku/" + str(date) + ".avi"
out = cv2.VideoWriter(videoPath, fourcc, 5.0, (640,480))
insertData(mycursor, str(date), videoPath)

(W, H) = (None, None)
time = 10
while True:
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)

    if W is None or H is None:
        (H,W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416,416), swapRB= True, crop= False)
    net.setInput(blob)
    layersOutputs = net.forward(outputLayer)
    

    boxes = []
    confidences = []
    classIDs = []

    for output in layersOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidenceThreshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype('int')
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    detectionNMS = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, NMSThreshold)
    if (len(detectionNMS) > 0 ):
        for i in detectionNMS.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            if (labels[classIDs[i]] == "person"):
                
               
                print(x,y)
                print("Tespit edilen İnsan Sayısı Anlık : ")
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, str(datetime.datetime.now()), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                out.write(frame)
            if (labels[classIDs[i]] == "bear"):
                
                print(x,y)
                print("Tespit edilen Ayı Sayısı Anlık : ")
                
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, str(datetime.datetime.now()), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                out.write(frame)
            if (labels[classIDs[i]] == "bird"):
                
                print(x,y)
                print("Tespit edilen Kuş Sayısı Anlık : ")
                
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, str(datetime.datetime.now()), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                out.write(frame)
            if (labels[classIDs[i]] == "cat"):
                
                print(x,y)
                print("Tespit edilen Kedi Sayısı Anlık : ")
                
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, str(datetime.datetime.now()), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                out.write(frame)
            if (labels[classIDs[i]] == "dog"):
                
                print(x,y)
                print("Tespit edilen Köpek Sayısı Anlık : ")
               
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, str(datetime.datetime.now()), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                out.write(frame)
            if (labels[classIDs[i]] == "horse"):
                
                print(x,y)
                print("Tespit edilen At Sayısı Anlık : ")
                
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, str(datetime.datetime.now()), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                out.write(frame)
            if (labels[classIDs[i]] == "sheep"):
                
                print(x,y)
                print("Tespit edilen Kuzu Sayısı Anlık : ")
                
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, str(datetime.datetime.now()), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                out.write(frame)
            if (labels[classIDs[i]] == "cow"):
                
                print(x,y)
                print("Tespit edilen İnek Sayısı Anlık : ")
                
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, str(datetime.datetime.now()), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                out.write(frame)
            if (labels[classIDs[i]] == "elephant"):
                
                print(x,y)
                print("Tespit edilen Fil Sayısı Anlık : ")
                
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, str(datetime.datetime.now()), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                out.write(frame)
            if (labels[classIDs[i]] == "zebra"):
                
                print(x,y)
                print("Tespit edilen Zebra Sayısı Anlık : ")
                
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, str(datetime.datetime.now()), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                out.write(frame)
            if (labels[classIDs[i]] == "giraffe"):
                
                print(x,y)
                print("Tespit edilen Zürafa Sayısı Anlık : ")
                
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, str(datetime.datetime.now()), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                out.write(frame)
    #cv2.imshow('sonuc', frame)
    k = cv2.waitKey(1)&0xFF
    if(k == ord('q')):
        break
    #elif(k == ord('r')):
    #   out.release()
    #   out = cv2.VideoWriter('output2.avi', fourcc, 5.0, (640,480))
    if(date.hour != datetime.datetime.now().hour or date.minute != datetime.datetime.now().minute):
        date = datetime.datetime.now()
        videoPath = "/home/utku/" + str(date) + ".avi"
        out.release()
        out = cv2.VideoWriter(videoPath, fourcc, 5.0, (640,480))
        insertData(mycursor, str(date), videoPath)
        
        
kitap.close()
video_capture.release()
cv2.destroyAllWindows()
