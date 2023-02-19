import cv2
import cvzone
import numpy as np
from cvzone.ColorModule import ColorFinder

cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

totalMoney = 0
myColorFinder = ColorFinder(False)

#customColorDetection
hsvVals = {'hmin':0,'smin':119,'vmin':60,'hmax':179,'smax':242,'vmax':255}
imgCount = np.zeros((480,640,3),np.uint8)

# configuration
def throwaway(a):
    pass

cv2.namedWindow("Settings")
cv2.resizeWindow("Settings",640,240)
cv2.createTrackbar("Threshold1","Settings",2,255,throwaway)
cv2.createTrackbar("Threshold2","Settings",100,255,throwaway)

def preProcessing(img):
    imgPre = cv2.GaussianBlur(img,(5,5),3)
    thresh1 = cv2.getTrackbarPos("Threshold1","Settings")
    thresh2 = cv2.getTrackbarPos("Threshold2","Settings")
    imgPre = cv2.Canny(imgPre, thresh1, thresh2)
    kernel = np.ones((3,3),np.uint8)

    # makes the outlines thicker and can do multiple iterations
    imgPre = cv2.dilate(imgPre,kernel,iterations=1)

    # closes any semi open contours
    imgPre = cv2.morphologyEx(imgPre, cv2.MORPH_CLOSE, kernel)

    return imgPre


while True:
    success, img = cap.read()
    imgPre = preProcessing(img)
    #cvzone.findContours(img,)
    imgContours, conFound = cvzone.findContours(img,imgPre,minArea=20)

    totalMoney = 0
    imgCount = np.zeros((480, 640, 3), np.uint8)

    # work only if we detect a contour

    if conFound:
        for contour in conFound:
            peri = cv2.arcLength(contour['cnt'],True)
            approx = cv2.approxPolyDP(contour['cnt'],0.02*peri,True)

            # circle is any contour with more than 5 edges
            if len(approx) > 5:

                area = contour['area']
                #print(area)
                x,y,w,h = contour['bbox']
                imgCrop = img[y:y+h,x:x+w]
                imgColor, mask = myColorFinder.update(imgCrop,hsvVals)
                whitePixelCount = cv2.countNonZero(mask)
                #print(whitePixelCount)


                if whitePixelCount > 100:
                    totalMoney += 5
                elif area < 4850:
                    totalMoney += 1
                else:
                    totalMoney += 2

    print(totalMoney)
    cvzone.putTextRect(imgCount, f'Rs.{totalMoney}', (100, 200), scale=10,offset=10,thickness=7,)
    imgStacked = cvzone.stackImages([img,imgPre,imgContours,imgCount],2,1)
    cvzone.putTextRect(imgStacked, f'Rs.{totalMoney}', (50, 50))
    cv2.imshow("Image",imgStacked)
    #cv2.imshow("ImgColor",imgColor)
    cv2.waitKey(1)


