import numpy as np
import cv2
import datetime
from openpyxl import Workbook

kitap = Workbook()
kitap.create_sheet("veriler")
yaz = kitap.get_sheet_by_name("veriler")
yaz.append(['Label', 'Count', 'Xcoordinate', 'Ycoordinate', 'Date-time'])
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
video_capture = cv2.VideoCapture("http://192.168.1.39:8080/hls/stream.m3u8")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter(str(date.strftime("%Y-%m-%d")) + '.avi', fourcc, 5.0, (640,480))
out = cv2.VideoWriter(str(date) + '.avi', fourcc, 5.0, (640,360))

(W, H) = (None, None)
time = 10
while True:
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    hebele = 0
    bear = 0
    giraffe = 0
    bird = 0
    cat = 0
    dog = 0
    horse = 0
    sheep = 0
    cow = 0
    elephant = 0
    zebra = 0
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
                hebele += 1
                yaz.append(['person', hebele, x, y, datetime.datetime.now()])
                print(x,y)
                print("Tespit edilen Ayı Sayısı Anlık : ", str(hebele))
                kitap.save("veriler.xlsx")
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, str(datetime.datetime.now()), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                out.write(frame)
            if (labels[classIDs[i]] == "bear"):
                bear += 1
                yaz.append(['bear', bear, x, y, datetime.datetime.now()])
                print(x,y)
                print("Tespit edilen Ayı Sayısı Anlık : ", str(hebele))
                kitap.save("veriler.xlsx")
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, str(datetime.datetime.now()), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                out.write(frame)
            if (labels[classIDs[i]] == "bird"):
                bird += 1
                yaz.append(['bird', bird, x, y, datetime.datetime.now()])
                print(x,y)
                print("Tespit edilen Kuş Sayısı Anlık : ", str(bird))
                kitap.save("veriler.xlsx")
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, str(datetime.datetime.now()), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                out.write(frame)
            if (labels[classIDs[i]] == "cat"):
                cat += 1
                yaz.append(['cat', cat, x, y, datetime.datetime.now()])
                print(x,y)
                print("Tespit edilen Kedi Sayısı Anlık : ", str(cat))
                kitap.save("veriler.xlsx")
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, str(datetime.datetime.now()), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                out.write(frame)
            if (labels[classIDs[i]] == "dog"):
                dog += 1
                yaz.append(['dog', dog, x, y, datetime.datetime.now()])
                print(x,y)
                print("Tespit edilen Köpek Sayısı Anlık : ", str(dog))
                kitap.save("veriler.xlsx")
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, str(datetime.datetime.now()), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                out.write(frame)
            if (labels[classIDs[i]] == "horse"):
                horse += 1
                yaz.append(['horse', horse, x, y, datetime.datetime.now()])
                print(x,y)
                print("Tespit edilen At Sayısı Anlık : ", str(horse))
                kitap.save("veriler.xlsx")
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, str(datetime.datetime.now()), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                out.write(frame)
            if (labels[classIDs[i]] == "sheep"):
                sheep += 1
                yaz.append(['sheep', sheep, x, y, datetime.datetime.now()])
                print(x,y)
                print("Tespit edilen Kuzu Sayısı Anlık : ", str(sheep))
                kitap.save("veriler.xlsx")
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, str(datetime.datetime.now()), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                out.write(frame)
            if (labels[classIDs[i]] == "cow"):
                cow += 1
                yaz.append(['cow', cow, x, y, datetime.datetime.now()])
                print(x,y)
                print("Tespit edilen İnek Sayısı Anlık : ", str(cow))
                kitap.save("veriler.xlsx")
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, str(datetime.datetime.now()), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                out.write(frame)
            if (labels[classIDs[i]] == "elephant"):
                elephant += 1
                yaz.append(['elephant', elephant, x, y, datetime.datetime.now()])
                print(x,y)
                print("Tespit edilen Fil Sayısı Anlık : ", str(elephant))
                kitap.save("veriler.xlsx")
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, str(datetime.datetime.now()), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                out.write(frame)
            if (labels[classIDs[i]] == "zebra"):
                zebra += 1
                yaz.append(['zebra', zebra, x, y, datetime.datetime.now()])
                print(x,y)
                print("Tespit edilen Zebra Sayısı Anlık : ", str(zebra))
                kitap.save("veriler.xlsx")
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, str(datetime.datetime.now()), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                out.write(frame)
            if (labels[classIDs[i]] == "giraffe"):
                giraffe += 1
                yaz.append(['giraffe', giraffe, x, y, datetime.datetime.now()])
                print(x,y)
                print("Tespit edilen Zürafa Sayısı Anlık : ", str(giraffe))
                kitap.save("veriler.xlsx")
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
    #	out.release()
    #	out = cv2.VideoWriter('output2.avi', fourcc, 5.0, (640,480))
    if(date.hour != datetime.datetime.now().hour or date.minute != datetime.datetime.now().minute):
    	date = datetime.datetime.now()
    	out.release()
    	out = cv2.VideoWriter(str(date) + '.avi', fourcc, 5.0, (640,360))
    	
    	
kitap.close()
video_capture.release()
cv2.destroyAllWindows()
