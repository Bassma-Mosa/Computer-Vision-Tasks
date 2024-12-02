from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO("yolov8n-seg.pt")

# results=model(source="Video.mp4",conf=0.4,save=True)
results=model(source="image.jpg",conf=0.4,save=True)

result_img = results[0].plot()
plt.imshow(result_img)
plt.axis('off')
plt.show()