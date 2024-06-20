from ultralytics import YOLO

def train():
    model = YOLO("yolov8n.pt")


    results = model.train(
        data="data.yaml",
        imgsz = 640,
        epochs = 2,
        batch = 8)

    # Evaluate the model's performance on the validation set
    results = model.val()

    model.export()

if __name__ == "__main__":
    train()
