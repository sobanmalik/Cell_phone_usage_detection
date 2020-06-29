
def classify(roi):
    import numpy as np
    import cv2

    args = {
            "config": './yolo_files/yolov3-tiny.cfg',
            "weights": './yolo_files/yolov3-tiny_best_14k.weights',
            "classes": './yolo_files/object.names'
            }
    # Load names classes
    classes = None
    with open(args['classes'], 'r') as f:
        classes = [line.strip() for line in f.readlines()]


    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    
    net_1 = cv2.dnn.readNet(args['weights'],args['config'])
    print('1')
    # Get names of output layers, output for YOLOv3 is ['yolo_16', 'yolo_23']
    def getOutputsNames(net_1):
        layersNames = net_1.getLayerNames()
        return [layersNames[i[0] - 1] for i in net_1.getUnconnectedOutLayers()]


#     # Darw a rectangle surrounding the object and its class name 
#     def draw_pred(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

#         label = str(classes[class_id])
#         color = COLORS[class_id]

#         cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

#         cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#     image = ROI
    blob_1 = cv2.dnn.blobFromImage(roi, 1.0/255.0, (416,416), [0,0,0], True, crop=False)
    Width = roi.shape[1]
    Height = roi.shape[0]
    net_1.setInput(blob_1)

    outs_1 = net_1.forward(getOutputsNames(net_1))
#     print(outs_1)
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.4
    nms_threshold = 0.3

    for out in outs_1: 
        #print(out.shape)
        for detection in out:
            #print(detection)

        #each detection  has the form like this [center_x center_y width height obj_score class_1_score class_2_score ..]
            scores = detection[5:]#classes scores starts from index 5
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # apply  non-maximum suppression algorithm on the bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    print(indices)
    if indices == ():
        return 0,0,0,0,'no phone',0
        
    else:
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            print(class_ids[i])
    #         count += 1
            label = str(classes[class_ids[i]])
#             color = (255,255,255)
            return x,y,w,h, label, confidence