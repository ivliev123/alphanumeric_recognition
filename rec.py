import numpy as np
import argparse
import cv2
import base

def index_min(array, n): #массив и номер столбца
    array_new=[]
    for i in range(len(array)):
        array_new.append(array[i][n-1])
    minimym = min(array_new)
    index=array_new.index(minimym)
    return minimym, index

#image = cv2.imread('barcode_01.jpg')
image = cv2.imread('test2.jpg')


final_wide = 600
r = float(final_wide) / image.shape[1]
dim = (final_wide, int(image.shape[0] * r))
image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
 
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


th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY_INV,11,7)
th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,11,7)
#edge = cv2.Canny (gray, 200,200,11)
#cv2.imwrite("Canny.jpg", edge)

#th3 = cv2.erode(th3, None, iterations = 1)
#th3 = cv2.dilate(th3, None, iterations = 2)



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
		get_x1=x1
		get_y1=y1-h1
		get_h1=int(1.3*h1)
		get_w1=w1
		if get_x1<0:
			get_w1=get_w1+get_x1
			get_x1=0
			
		if get_y1<0:
			get_h1=get_h1+get_y1 
			get_y1=0
		get_im_ar = image[get_y1:get_y1+get_h1, get_x1:get_x1+get_w1]
		cv2.imwrite("info/"+str(b) +".jpg", get_im_ar)
		
		gray_info = cv2.cvtColor(get_im_ar, cv2.COLOR_BGR2GRAY)
		#gray_info = cv2.medianBlur(gray_info,3)
		gray_info = cv2.bilateralFilter(gray_info,7,5,5)
		cv2.imwrite("info/gray_info.jpg", gray_info)
		th3_info = cv2.adaptiveThreshold(gray_info,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,9,4)
		cv2.imwrite("info/th3."+str(b)+".jpg",th3_info)
		im_info, cnts_info, hierarchy_info  = cv2.findContours(th3_info, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		k=0
		for array_info in cnts_info:
			
			cnt_info = cnts_info[k]
			x_info,y_info,w_info,h_info = cv2.boundingRect(cnt_info)
			if w_info>0.2*h_info and w_info<0.9*h_info and w_info>4 and h_info>4:
				im_digit_letter = th3_info[y_info:y_info+h_info, x_info:x_info+w_info]   # переменная (фото которое нужно распознавать
				im_digit_letter_res=cv2.resize(im_digit_letter,(20,20))
				# тут в будущем будет функция для прогонк через нейронку
				

				cv2.rectangle(get_im_ar,(x_info,y_info),(x_info+w_info,y_info+h_info),(255,255,255),2)
				cv2.imwrite("info/"+str(b) +".jpg", get_im_ar)
			k=k+1

		get_x2=x1
		get_h2=int(1.3*h1)
		get_y2=y1+h1-int(0.3*h1)
		get_w2=w1

		get_im_ar = image[get_y2:get_y2+get_h2, get_x2:get_x2+get_w2]
		cv2.imwrite("info/"+str(b) +".002.jpg", get_im_ar)

	b = b+ 1

b=0
#for array in cnts:
	#if b<10:
		#c = sorted(cnts, key = cv2.contourArea, reverse = True)[b]
		#x1,y1,w1,h1 = cv2.boundingRect(c)
		#cv2.rectangle(image,(x1,y1),(x1+w1,y1+h1),(0,0,255),2)
	

	#cnt = cnts[b]
	#x,y,w,h = cv2.boundingRect(cnt)
	#if w>0.2*h and w<0.9*h and w>4 and h>4:
		#cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),0)
	#b = b+ 1
#print(b)
#cnt = cnts[1]
#cv2.drawContours(image, [cnt], 0, (0,255,0), 3)

cv2.imwrite("Image.jpg", image)
cv2.waitKey(0)
