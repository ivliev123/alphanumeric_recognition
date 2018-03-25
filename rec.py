import numpy as np
import argparse
import cv2


#image = cv2.imread('barcode_01.jpg')
image = cv2.imread('test.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray,3)

#gradX = cv2.Sobel(gray, cv2.CV_64F,1,0,ksize=-1)
#gradY = cv2.Sobel(gray, cv2.CV_64F,0,1,ksize=-1)
#gradient = cv2.subtract(gradX, gradY)
#gradient = cv2.convertScaleAbs(gradient)

#blurred = cv2.blur(gradient, (9, 9))
#(_,thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
#closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


#closed = cv2.erode(closed, None, iterations = 4)
#closed = cv2.dilate(closed, None, iterations = 4)


th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY_INV,11,7)
th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,11,7)
#edge = cv2.Canny (gray, 200,200,11)
#cv2.imwrite("Canny.jpg", edge)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
closed = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)


closed = cv2.erode(closed, None, iterations = 4)
closed = cv2.dilate(closed, None, iterations = 4)
cv2.imwrite("сlosed.jpg", closed)

cv2.imwrite("th2.jpg", th2)
cv2.imwrite("th3.jpg", th3)

im_big, cnts_big, hierarchy_big  = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

im2, cnts, hierarchy  = cv2.findContours(th3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

#rect = cv2.minAreaRect(c)
#box = np.int0(cv2.boxPoints(rect))
#cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

b=0

for array_big in cnts_big:
	if b<4:
		c = sorted(cnts_big, key = cv2.contourArea, reverse = True)[b]
		x1,y1,w1,h1 = cv2.boundingRect(c)
		cv2.rectangle(image,(x1,y1),(x1+w1,y1+h1),(0,0,255),2)
	b = b+ 1

b=0
for array in cnts:
	#if b<10:
		#c = sorted(cnts, key = cv2.contourArea, reverse = True)[b]
		#x1,y1,w1,h1 = cv2.boundingRect(c)
		#cv2.rectangle(image,(x1,y1),(x1+w1,y1+h1),(0,0,255),2)
	

	cnt = cnts[b]
	x,y,w,h = cv2.boundingRect(cnt)
	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
	#cv2.drawContours(image, [cnt], 0, (0,255,0), 3)
	b = b+ 1
print(b)
#cnt = cnts[1]
#cv2.drawContours(image, [cnt], 0, (0,255,0), 3)

cv2.imwrite("Image.jpg", image)
cv2.waitKey(0)
