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

def sortPoints(approx):
    somme = []

    # Find the UpperLeft and DownRight points
    for i in range(len(approx)):
        num = approx[i][0]
        somme.append(num[0] + num[1])

    # Put the UpperLeft point in first position and DownRight point in third position
    ul = np.argmin(somme)
    dr = np.argmax(somme)
    approx[[ul, 0]] = approx[[0, ul]]
    approx[[dr, 2]] = approx[[2, dr]]


    if approx[1][0][0] > approx[3][0][0]:
        approx[[1, 3]] = approx[[3, 1]]


    return approx



def getChessboardCorners(filename):
    # Preprocess the image
    img = cv2.imread(filename)
    img = cv2.resize(img, (640, 640))
    # get the minimum bounding box for the chip image
    image = img[10:-10, 10:-10]
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[..., 0]
    ret, thresh = cv2.threshold(imgray, 20, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    mask = 255 - thresh
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    max_area = 0
    best = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            best = contour

    rect = cv2.minAreaRect(best)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # crop image inside bounding box
    scale = 1  # cropping margin, 1 == no margin
    W = rect[1][0]
    H = rect[1][1]

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)

    angle = rect[2]
    rotated = False
    if angle < -45:
        angle += 90
        rotated = True

    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    size = (int(scale * (x2 - x1)), int(scale * (y2 - y1)))

    M = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)

    cropped = cv2.getRectSubPix(image, size, center)
    cropped = cv2.warpAffine(cropped, M, size)

    croppedW = W if not rotated else H
    croppedH = H if not rotated else W

    image = cv2.getRectSubPix(
        cropped, (int(croppedW * scale), int(croppedH * scale)), (size[0] / 2, size[1] / 2))
    cv2.imshow("Vediamo", image)
    cv2.waitKey(0)

    exit()



    # Load the file and do inference
    results = model(img, save=True, save_conf=True, conf=0.5)

    # Extract binary mask for the chessboard
    # --------------------------------------------------------------------------------------------------
    mask = getMask(model, results[0])


    # Extract contours
    # --------------------------------------------------------------------------------------------------
    imgray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

    # Take only the largest contour (to avoid errors)
    contour = max(contours, key = cv2.contourArea)
    
    # Draw approximated polygon for the chessboard
    # --------------------------------------------------------------------------------------------------
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)

    points = sortPoints(approx)

    return approx

if __name__ == "__main__":
    filename = "test7.jpg"
    img = cv2.imread(filename)
    img = cv2.resize(img, (640, 640))

    approx = getChessboardCorners(filename)
    
    cv2.drawContours(img, [approx], -1, (0,0,255), 3)

    cv2.imshow('Contours', img)
    cv2.waitKey(0)

