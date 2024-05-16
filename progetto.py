from ultralytics import YOLO
import numpy as np
import cv2

# Canny edge detection
# def canny_edge(img, sigma=0.33):
#     v = np.median(img)
#     lower = int(max(0, (1.0 - sigma) * v))
#     upper = int(min(255, (1.0 + sigma) * v))
#     edges = cv2.Canny(img, lower, upper)
#     return edges



# img = cv2.imread('5.jpg')
# img = cv2.resize(img, (500,500))
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# canny = canny_edge(gray)


# lines = cv2.HoughLinesP(canny, 1, np.pi/180, 90, minLineLength=100, maxLineGap=30)
# for line in lines:
#     x1,y1,x2,y2 = line[0]
#     cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

# cv2.imshow('Immagine', img)
# cv2.waitKey(0)














model = YOLO("yolov8n-seg.pt")


results = model.train(
	data="data.yaml",
	imgsz = 640,
	epochs = 3,
	batch = 8)

# Evaluate the model's performance on the validation set
results = model.val()

print(results)

# from ndarray
img = cv2.imread("1.jpg")
results = model.predict(source=img, save=True, save_txt=True)  # save predictions as labels

# model_trained = YOLO("runs/detect/yolov8n_corners/weights/best.pt")


# for corner in findCorners(image):
# 	corner = np.array(corner).astype(int)
# 	image = cv2.circle(image, (*corner,), 10, 100, -1)

# image = cv2.imread("3.jpg")

# image = cv2.resize(image, (0,0), fx = 0.5, fy=0.5)

# cv2.imshow("Immagine", image)
# cv2.waitKey(0)


# def findCorners(image):
# 	result = model.predict(source = image, line_width = 1, conf = 0.25, save_txt = True, save = True)

# 	print(result[0].boxes)

# 	boxes = result[0].boxes
# 	arr = boxes.xywh.numpy()
# 	points = arr[:, 0:2]

# 	# corners = order_points(points)

# 	return points
