from ultralytics import YOLO

def train():
    model = YOLO("best.pt")


    results = model.train(
        data="data.yaml",
        imgsz = 640,
        epochs = 1,
        batch = 1)

    # Evaluate the model's performance on the validation set
    results = model.val()

    model.export()

if __name__ == "__main__":
    train()
