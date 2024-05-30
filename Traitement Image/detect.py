import  cv2
import copy
import os
from matplotlib import pyplot as plt
import numpy as np
from utils import stackImages

def empty(a):
    pass

def canny_cv(originalImage):
    image =  copy.deepcopy(originalImage)
    gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    #gray = cv2.medianBlur(gray, 6)
    med_val = np.median(gray)
    #test_multiple_thresholds(image)

    upper = int(min(255,1.3*med_val))#formule pour le seuil haut
    lower = int(min(0,0.7*med_val))#formule pour le seuil bas

    
    thresh1 = cv2.getTrackbarPos("Seuil bas", "Settings")
    
    thresh2 = cv2.getTrackbarPos("Seuil haut", "Settings")
    edge = cv2.Canny(gray, lower, upper+60)
    kernel = np.ones((1,1), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilate = cv2.dilate(edge, kernel, iterations=1)
    dilate= cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)  
        
    cv2Contours, _ =cv2.findContours(dilate, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    filteredCv2Image, bboxCoordinates = filter_cv_contours(image, cv2Contours, minArea=300)
    return  dilate, filteredCv2Image, bboxCoordinates

        
        

def hough_detection(originalImage):
    image = copy.deepcopy(originalImage)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 9)
    imageOpen = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    rows = imageOpen.shape[0]
    circles = cv2.HoughCircles(imageOpen, cv2.HOUGH_GRADIENT, 1, rows / 8,param1=100, param2=30, minRadius=0, maxRadius=0)
    boundingBoxCoordinates = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        for (x, y, r) in circles:
            
            if r>300:
                continue
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
            x1, y1 = x - r, y - r
            x2, y2 = x + r, y + r
            boundingBoxCoordinates.append([x1, y1, x2, y2])
            cv2.rectangle(image, (x - r, y - r), (x + r, y + r), (0, 255, 0), 2) 
    return image, boundingBoxCoordinates 


   
    

def filter_cv_contours(_image, contours, minArea) :
    image= copy.deepcopy(_image)
    boundingBoxCoordinates = []
    for c in contours:
            area = cv2.contourArea(c)
            peri = cv2.arcLength(c, True)
            #

            approx = cv2.approxPolyDP(c, 0.02*peri, True)
            #
            if area > minArea and len(approx) >5:
           
                cv2.drawContours(image,[c], 0, (0,255,0), 3)
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
                boundingBoxCoordinates.append([x, y, x+w, y+h]) 

                #text = f'({x}, {y})'
                text = f'Width: {w}'
                cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    return image, boundingBoxCoordinates

def test_multiple_thresholds(image):

    cv2.namedWindow("Settings")
    cv2.resizeWindow("Settings", 640, 240)
    cv2.createTrackbar("Seuil bas", "Settings", 50, 255, empty)
    cv2.createTrackbar("Seuil haut", "Settings", 182, 255, empty)
    while (1):
        dilate, canny, cannyBbox = canny_cv(image)
        hough, houghBbox= hough_detection(image)
        #
        #
        images = [image, dilate, canny, hough]
        winNames = ["Original", "Dilate", "Canny", "Hough"]
        imgStack = stackImages(images, 2, 0.5, winNames)
        cv2.imshow('image',imgStack)
        #detectColor(cannyBbox, image)
        #detectColor(houghBbox, image)
    
        k = cv2.waitKey(500) 
        if k == 27:
            break

def main_detect() : 
    path = "input/tests/" 
    for filename in os.listdir(path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".JPG"):
            
            image_path = os.path.join(path,filename)
            image= cv2.imread(image_path)
            #dilate, canny, cannyBbox = canny_cv(image)
            dilate= np.zeros_like(image)
            canny = np.zeros_like(image) 
            hough, houghBbox= hough_detection(image)
            #
            
            images = [image, dilate, canny, hough]
            winNames = ["Original", "Dilate", "Canny", "Hough"]
            imgStack = stackImages(images, 2, 0.5, winNames)
            #annotations = assign_coin_class(houghBbox , image)
            #cv2.imshow('image',imgStack)
            #if annotations:
                #display_annotations(image, annotations, filename)
            #cv2.waitKey(0)
            #detectColor(cannyBbox, image)
            #detectColor(houghBbox, image)
            test_multiple_thresholds(image)
        else:
            continue
        
#main_detect()