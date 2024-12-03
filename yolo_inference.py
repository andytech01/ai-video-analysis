from ultralytics import YOLO

model = YOLO("models/best.pt")  # Load model

# Inference
results = model.predict("assets/08fd33_4.mp4", save=True)
print(results[0])

print("=" * 30)

for box in results[0].boxes:
    print(box)
