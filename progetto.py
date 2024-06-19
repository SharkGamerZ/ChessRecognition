from ultralytics import YOLO
import torch
import numpy as np
import cv2

# --------------------------------------------------------------------------------------------------
#									RECOGNITION WITH STRAIGHT LINES
# --------------------------------------------------------------------------------------------------

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




# --------------------------------------------------------------------------------------------------
#									TRAIN WITH AI
# --------------------------------------------------------------------------------------------------


# model = YOLO("best.pt")


# results = model.train(
# 	data="data.yaml",
# 	imgsz = 640,
# 	epochs = 2,
# 	batch = 8)

# # Evaluate the model's performance on the validation set
# results = model.val()

# model.export()

# --------------------------------------------------------------------------------------------------
#								INFERENCE WITH AI
# --------------------------------------------------------------------------------------------------


# Chessboard Recognition
# --------------------------------------------------------------------------------------------------

# Load the model
model = YOLO("best.pt")

# Load the file and do inferencec
filename = "2.jpg"
img = cv2.imread(filename)
img = cv2.resize(img, (640, 640))
results = model(filename, save=True, save_conf=True, conf=0.5)


# Extract binary mask for the chessboard
# --------------------------------------------------------------------------------------------------
for result in results:
    # get array results
    masks = result.masks.data
    boxes = result.boxes.data
    # extract classes
    clss = boxes[:, 5]
    # get indices of results where class is 0 (chessboard has class 0)
    chessboard_indices = torch.where(clss == 0)
    # use these indices to extract the relevant masks
    chessboard_masks = masks[chessboard_indices]
    # scale for visualizing results
    chessboard_mask = torch.any(chessboard_masks, dim=0).int() * 255
    # save to file
    cv2.imwrite(str(model.predictor.save_dir / 'merged_segs.jpg'), chessboard_mask.cpu().numpy())

mask = cv2.imread(str(model.predictor.save_dir / 'merged_segs.jpg'))
mask = cv2.resize(mask, (640, 640))


# Extract contours
# --------------------------------------------------------------------------------------------------
imgray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour = contours[0]

# Draw approximated polygon for the chessboard
# --------------------------------------------------------------------------------------------------
perimeter = cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)
cv2.drawContours(img, [approx], -1, (0,0,255), 3)

cv2.imshow('Contours', img)
cv2.waitKey(0)



