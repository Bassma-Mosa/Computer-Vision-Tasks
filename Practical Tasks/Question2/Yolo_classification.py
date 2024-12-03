
from ultralytics import YOLO

model= YOLO('yolov8n-cls.pt')

model.train(data='D:/FCI_Material/First/Computer Vision/sec5/Alzheimer_s Dataset', epochs=5)

metrics = model.val()
metrics.top1
metrics.top5

results = model.predict("D:/FCI_Material/First/Computer Vision/sec5/Alzheimer_s Dataset/val/NonDemented/26 (77).jpg")
probs = results.probs
print(probs.data)
