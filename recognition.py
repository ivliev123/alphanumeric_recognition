import numpy as np
import argparse
import cv2
import base
import imutils
import tensorflow as tf



base_digite_s_letter=base.base_digite_s_letter
all_array_text=[]

array_for_bace = ["B", "C", "D", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "X", "Y", "Z",  "A", "E", "I", "O", "U", "Y", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]

new_model = tf.keras.models.load_model('/home/ivliev/tensorflow/alphanumer.model')



def index_min(array, n): #массив и номер столбца
    array_new=[]
    for i in range(len(array)):
        array_new.append(array[i][n])
    minimym = min(array_new)
    index=array_new.index(minimym)
    return minimym, index

def index_max(array, n):
    array_new=[]
    for i in range(len(array)):
        array_new.append(array[i][n])
    maximym = max(array_new)
    indexmax=array_new.index(maximym)
    return maximym, indexmax



####################################################обработка баркодов
# load the image and convert it to grayscale
image = cv2.imread("test2.jpg")
if image.shape[1] > 600:
    image = imutils.resize(image, width=600)

image2=image.copy()
clone = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imwrite("b_gray.png",gray)
#cv2.imwrite("a_Image.png", image)


ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)

# subtract the y-gradient from the x-gradient
gradient = cv2.subtract(gradX, gradY)
cv2.imwrite("gradient.png",gradient)
gradient = cv2.convertScaleAbs(gradient)
cv2.imwrite("gradient2.png",gradient)

# blur and threshold the image
blurred = cv2.blur(gradient, (9, 9))
#cv2.imwrite("e_blurred.png",blurred)
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
#cv2.imwrite("f_thresh.png",thresh)

# construct a closing kernel and apply it to the thresholded image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#cv2.imwrite("g_closed.png",closed)

# perform a series of erosions and dilations
closed = cv2.erode(closed, None, iterations = 4)
#cv2.imwrite("h_closed1.png",closed)
closed = cv2.dilate(closed, None, iterations = 4)

cnts, histogram = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if imutils.is_cv2() else cnts[1]



for i in range(1):
    c = sorted(cnts, key = cv2.contourArea, reverse = True)[i]

    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
    box = np.int0(box)

    # draw a bounding box arounded the detected barcode and display the
# image
    cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
    x,y,w,h = cv2.boundingRect(c)
    #cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)


    coords = np.column_stack(np.where(image > 0))
    #angle = cv2.minAreaRect(coords)[-1]
    angle = cv2.minAreaRect(c)[-1]
    if angle < -45:
        angle = (90 + angle)

    else:
        angle = -angle
    #print('angle',angle)

    (h, w) = closed.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image2, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite("rotated.jpg", rotated)
print('angle',angle)
####################################################обработка баркодов

##################################################################################################################
image = rotated
if image.shape[1] > 600:
    image = imutils.resize(image, width=600)

image2=image.copy()
clone = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)

gradient = cv2.subtract(gradX, gradY)
cv2.imwrite("gradient.png",gradient)
gradient = cv2.convertScaleAbs(gradient)
cv2.imwrite("gradient2.png",gradient)

# blur and threshold the image
blurred = cv2.blur(gradient, (9, 9))
#cv2.imwrite("e_blurred.png",blurred)
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
#cv2.imwrite("f_thresh.png",thresh)

# construct a closing kernel and apply it to the thresholded image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#cv2.imwrite("g_closed.png",closed)

closed = cv2.erode(closed, None, iterations = 4)
closed = cv2.dilate(closed, None, iterations = 4)
#cv2.imwrite("l_closed2.png",closed)

# find the contours in the thresholded image, then sort the contours
# by their area, keeping only the largest one
cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]


##########обработка перевернутого изображения
gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray,3)

th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY_INV,11,7)
th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,11,7)
#cv2.imwrite("th2.jpg", th2)
#cv2.imwrite("th3.jpg", th3)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
closed1 = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)
closed1 = cv2.erode(closed1, None, iterations = 4)
closed1 = cv2.dilate(closed1, None, iterations = 4)
#cv2.imwrite("сlosed1.jpg", closed)

# im_big, cnts_big, hierarchy_big  = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts_big, hierarchy_big  = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



b=0
for array_big in cnts_big:
    if b<2:
        c = sorted(cnts_big, key = cv2.contourArea, reverse = True)[b]
        x1,y1,w1,h1 = cv2.boundingRect(c)
        #cv2.rectangle(rotated,(x1,y1),(x1+w1,y1+h1),(0,0,255),2)
        vn="v"
        if vn=="v":
            get_x1=x1
            get_y1=y1-h1
            get_h1=int(1*h1)
            get_w1=w1
            if get_x1<0:
                get_w1=get_w1+get_x1
                get_x1=0

            if get_y1<0:
                get_h1=get_h1+get_y1
                get_y1=0
            get_im_ar = rotated[get_y1:get_y1+get_h1, get_x1:get_x1+get_w1]
        if vn=="n":
            get_x2=x1-5
            get_h2=int(1.9*h1)
            get_y2=y1+h1-int(0.8*h1)
            get_w2=w1

            get_im_ar = rotated[get_y2:get_y2+get_h2, get_x2:get_x2+get_w2]



        cv2.imwrite("info/"+str(b) +".jpg", get_im_ar)

        gray_info = cv2.cvtColor(get_im_ar, cv2.COLOR_BGR2GRAY)
        #gray_info = cv2.medianBlur(gray_info,3)
        gray_info = cv2.bilateralFilter(gray_info,7,5,5)
        cv2.imwrite("info/gray_info.jpg", gray_info)
        th3_info = cv2.adaptiveThreshold(gray_info,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,9,4)
        cv2.imwrite("info/th3."+str(b)+".jpg",th3_info)
        # im_info, cnts_info, hierarchy_info  = cv2.findContours(th3_info, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_info, hierarchy_info  = cv2.findContours(th3_info, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        k=0
        all_digit_letter = []

        for array_info in cnts_info:

            cnt_info = cnts_info[k]
            x_info,y_info,w_info,h_info = cv2.boundingRect(cnt_info)
            if w_info>0.3*h_info and w_info<0.9*h_info and w_info>=4 and h_info>9:
                #добавление координат и размеров в массив для дальнейшей обработки на мин и маx
                digit_letter = [x_info,y_info,x_info+w_info,y_info+h_info]
                all_digit_letter.append(digit_letter)
                im_digit_letter = th3_info[y_info:y_info+h_info, x_info:x_info+w_info]   # переменная (фото которое нужно распознавать
                im_digit_letter_res=cv2.resize(im_digit_letter,(20,20))
                im_digit_letter_res_ar=np.array(im_digit_letter_res)  #рисунок который нужно сравнивать переведен в массив

                # тут в будущем будет функция для прогонк через нейронку
                test_array = []
                img_for_nn = cv2.resize(im_digit_letter,(28,28))
                ret, img_for_nn = cv2.threshold(img_for_nn,127,255,cv2.THRESH_BINARY_INV)
                test_array.append(img_for_nn)
                test_array = np.array(test_array)

                predictions = new_model.predict(test_array)
                # print(np.argmax(predictions[0]))
                print(array_for_bace[np.argmax(predictions[0])])

                font = cv2.FONT_HERSHEY_PLAIN
                cv2.putText(get_im_ar,array_for_bace[np.argmax(predictions[0])],(x_info,y_info), font, 1,(255,0,0.,),1,cv2.LINE_AA)

                #minimym, index = index_min(base_digite_s_letter,3)
                #font = cv2.FONT_HERSHEY_SIMPLEX
                #cv2.putText(get_im_ar,base_digite_s_letter[index][0],(x_info,y_info), font, 1,(255,0,0),1,cv2.LINE_AA)

                cv2.imwrite("kontur/"+str(k)+".jpg",im_digit_letter_res)
                for l in range(len(base_digite_s_letter)):

                    im_from_base=cv2.imread(base_digite_s_letter[l][1])


                    im_from_base = cv2.cvtColor(im_from_base, cv2.COLOR_BGR2GRAY)
                    im_from_base_ar=np.array(im_from_base)
                    #cv2.imwrite("testtest.jpg",im_from_base)
                    razn=0
                    for y in range(20):
                        for x in range(20):
                            pixl1 = im_digit_letter_res_ar[y][x]
                            pixl2 = im_from_base_ar[y][x]
                            razn= razn + abs(int(pixl1) - (pixl2))

                    #print(razn)
                    base_digite_s_letter[l][2]=razn
                    #print(razn)
                #print(base_digite_s_letter)
                minimym, index = index_min(base_digite_s_letter,2)
                font = cv2.FONT_HERSHEY_PLAIN
                array_text = [base_digite_s_letter[index][0],x_info,y_info,"None"]
                # last element- delta y (for find all digits and letter on one leval)
                all_array_text.append(array_text)

                cv2.rectangle(get_im_ar,(x_info,y_info),(x_info+w_info,y_info+h_info),(255,255,255,0.1),2)


                # cv2.putText(get_im_ar,base_digite_s_letter[index][0],(x_info,y_info), font, 1,(0,0,255,),1,cv2.LINE_AA)
                cv2.imwrite("info/"+str(b) +".jpg", get_im_ar)

            k=k+1
        if len(all_digit_letter)>0:
            minimymx, indexminx = index_min(all_digit_letter,0)
            minimymy, indexminY = index_min(all_digit_letter,1)
            maximymxw, indexmaxxw = index_max(all_digit_letter,2)
            maximymyh, indexmaxyh = index_max(all_digit_letter,3)


            image_for_tesseract = gray_info[minimymy:maximymyh,minimymx:maximymxw]
            image_for_tesseract = cv2.adaptiveThreshold(image_for_tesseract,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,13,5)

            #image_for_tesseract=Image.fromarray(image_for_tesseract)
            #text = pytesseract.image_to_string(image_for_tesseract)
            #print(text)
            cv2.imwrite("info/image_for_tesseract"+str(b)+".png",image_for_tesseract)
        #get_x2=x1
        #get_h2=int(1*h1)
        #get_y2=y1+h1-int(0.55*h1)
        #get_w2=w1

        #get_im_ar = rotated[get_y2:get_y2+get_h2, get_x2:get_x2+get_w2]
        #cv2.imwrite("info/"+str(b) +".002.jpg", get_im_ar)






    b = b+ 1

b=0
array_for_finish=[]
n_str=0
metka=True
#формирование строки
all_array_text.sort(key=lambda i: i[2])

for dl1 in range(len(all_array_text)):
    #print(all_array_text[dl1][0])
    if metka==True:
        array_for_finish.append([])

    array_for_finish[n_str].append([all_array_text[dl1][0],all_array_text[dl1][1]])
    if dl1==len(all_array_text)-1:
        delta_y = abs(all_array_text[dl1][2] - all_array_text[dl1-1][2])
    else:
        delta_y = abs(all_array_text[dl1][2] - all_array_text[dl1+1][2])
    if delta_y > 5:
        #print()
        metka=True
        n_str+=1
    else:
        metka=False

#print(array_for_finish)

stroka=""
#првильное размещение символов в строке
for d in range(len(array_for_finish)):
    array_for_finish[d].sort(key=lambda i: i[1])
    stroka=""
    for l in range(len(array_for_finish[d])):
        stroka=stroka+array_for_finish[d][l][0]
    print()
    print(stroka)

cv2.imwrite("Image.jpg", clone)
cv2.waitKey(0)
