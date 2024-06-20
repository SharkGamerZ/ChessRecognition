import cv2

from chessboard.chessboard import getChessboardCorners
from chess_pieces.chess_pieces import getPiecesList


if __name__ == '__main__':
	filename = 'test.jpg'
	img = cv2.imread(filename)
	img = cv2.resize(img, (640, 640))


	chessboard = getChessboardCorners(filename)
	pieces = getPiecesList(filename)

	cv2.drawContours(img, [chessboard], -1, (0,0,255), 3)
	[cv2.rectangle(img, p[1], p[2], (0,255,0), 2) for p in pieces]

	cv2.imshow("Immagine", img)
	cv2.waitKey()