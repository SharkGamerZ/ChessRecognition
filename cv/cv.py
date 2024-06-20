import cv2

from chessboard.chessboard import getChessboardCorners
from chess_pieces.chess_pieces import getPiecesList


if __name__ == '__main__':
	filename = 'test.jpg'
	img = cv2.imread(filename)
	img = cv2.resize(img, (640, 640))


	chessboard = getChessboardCorners(filename)
	pieces = getPiecesList(filename)

	ul, dl, dr, ur = chessboard 

	print(ul, dl, dr, ur)

	l = ((ul[0][0] + dl[0][0])/2, (ul[0][1] + dl[0][1])/2)

	print(l)

	# Devo dividire ogni lato per 8, andando a prendere:
	#	- per le x le 8 coppie (x1, x2) che delimitano ogni colonna
	#	- per le y le 8 coppie (y1, y2) che delimitano ogni riga

	# Calcolo poi il punto centrale di ogni pezzo
	# Controllo il punto centrale in che range di riga/colonna sia 

	cv2.drawContours(img, [chessboard], -1, (0,0,255), 3)
	[cv2.rectangle(img, p[1], p[2], (0,255,0), 2) for p in pieces]

	cv2.imshow("Immagine", img)
	cv2.waitKey()
