from ultralytics import YOLO

# Load a model
model = YOLO("./best.pt")

# Train the model
train_results = model.train(
    data="mydata.yaml",
    imgsz=240,
    epochs=1000,
    device=0,  
    lr0 = 0.001 
)
