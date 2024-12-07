from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO("yolov8n-seg.pt")

results=model(source="image.jpg",show=True,conf=0.4,save=True)
# results=model(source="C:/Users/asd432/Desktop/videoo.mp4",show=True,conf=0.4,save=True)

result_img = results[0].plot()
plt.imshow(result_img)
plt.axis('off')
plt.show()
