# USAGE
# python detect_barcode.py --image images/barcode_01.jpg

# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2



# load the image and convert it to grayscale
image = cv2.imread("images/test2.jpg")
if image.shape[1] > 500:
    image = imutils.resize(image, width=500)

image2=image.copy()
clone = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("b_gray.png",gray)
cv2.imwrite("a_Image.png", image)

# compute the Scharr gradient magnitude representation of the images
# in both the x and y direction using OpenCV 2.4
ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)
cv2.imwrite("c_gradX.png",gradX)
cv2.imwrite("d_gradY.png",gradY)

# subtract the y-gradient from the x-gradient
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

# blur and threshold the image
blurred = cv2.blur(gradient, (9, 9))
cv2.imwrite("e_blurred.png",blurred)
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
cv2.imwrite("f_thresh.png",thresh)

# construct a closing kernel and apply it to the thresholded image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
cv2.imwrite("g_closed.png",closed)

# perform a series of erosions and dilations
closed = cv2.erode(closed, None, iterations = 4)
cv2.imwrite("h_closed1.png",closed)
closed = cv2.dilate(closed, None, iterations = 4)
cv2.imwrite("l_closed2.png",closed)

# find the contours in the thresholded image, then sort the contours
# by their area, keeping only the largest one
cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]


for i in range(3):
	c = sorted(cnts, key = cv2.contourArea, reverse = True)[i]

	# compute the rotated bounding box of the largest contour
	rect = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
	box = np.int0(box)

	# draw a bounding box arounded the detected barcode and display the
# image
	cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
	x,y,w,h = cv2.boundingRect(c)
	cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)


	coords = np.column_stack(np.where(image > 0))
	#angle = cv2.minAreaRect(coords)[-1]
	angle = cv2.minAreaRect(c)[-1]
	if angle < -45:
		angle = (90 + angle)

	else:
		angle = -angle
	print('angle',angle)

	(h, w) = closed.shape[:2]
	center = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D(center, angle, 1.0)
	rotated = cv2.warpAffine(image2, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
	cv2.imwrite("info"+str(i)+".jpg", rotated)

cv2.imwrite("m_Image_cnt.png", image)
cv2.waitKey(0)
