from ultralytics import YOLO
import torch
import numpy as np
import cv2

def getMask(model, result):
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

    return mask

def getCorners(filename=None):
    # Load the model
    model = YOLO("best.pt")

    # Load the file and do inference
    results = model(filename, save=True, save_conf=True, conf=0.5)

    # Extract binary mask for the chessboard
    # --------------------------------------------------------------------------------------------------
    mask = getMask(model, results[0])


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

    return approx

if __name__ == "__main__":
    filename = "test.jpg"
    img = cv2.imread(filename)
    img = cv2.resize(img, (640, 640))

    approx = getCorners(filename)
    
    cv2.drawContours(img, [approx], -1, (0,0,255), 3)

    cv2.imshow('Contours', img)
    cv2.waitKey(0)

