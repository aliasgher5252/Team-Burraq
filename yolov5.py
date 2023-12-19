import torch
import numpy as np
import cv2
COCO_CLASSES = {"X":0}
class YoloDetector():

    def __init__(self,model_path=None):
        if model_path!=None:
            self.model = torch.hub.load('.', 'custom', path=model_path, source='local')
            print("Model Successfully Loaded...")
        else:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m',pretrained=True)

    
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using Device: {self.device}")

        self.model=self.model.to(self.device)
    
    def prediction(self,frame):
        #resized_frame=cv2.resize(frame,input_shape)
        results=self.model(frame)
        print("The result is {}".format(results))
        return frame,results.xyxy[0]
    
    def get_classIDs(self,labels):
        return [COCO_CLASSES[label] for label in labels]

    def get_bbox(self,frame,confidence,labels):
        resized_frame, predictions=self.prediction(frame)
        class_ids=self.get_classIDs(labels)
        detections=[]
        for prediction in predictions:
            x1, y1, x2, y2, conf, class_id = prediction.cpu().numpy()
            
            x1, y1, x2, y2, conf, class_id=int(x1), int(y1), int(x2),int(y2), float(conf), int(class_id)
            if (class_id in class_ids) and conf>=confidence:
                detections.append([x1, y1, x2, y2])
        return resized_frame,detections
