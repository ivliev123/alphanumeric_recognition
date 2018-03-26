import numpy as np
import argparse
import cv2


#image = cv2.imread('barcode_01.jpg')
image = cv2.imread('baseno/k.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret,th1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)

th3_base = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,7)


cv2.imwrite("th3_base.jpg", th1)

im2, cnts, hierarchy  = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

b=0
for array in cnts:

	cnt = cnts[b]
	x,y,w,h = cv2.boundingRect(cnt)
	get= th1[y:y+h, x:x+w]
	get_resize=cv2.resize(get, (20,20))
	cv2.imwrite("base/"+"k."+str(b)+".jpg", get_resize)
	b=b+1
