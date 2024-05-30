
import  cv2
import os
from matplotlib import pyplot as plt
import numpy as np
from utils import stackImages, display_annotations, sorted_alphanumeric
from detect import canny_cv, hough_detection
from annotate import assign_coin_class
from evaluate import get_gt_array, calculate_map

def main() : 
    path = "input/tests/" 
    annotations_array = []  
    gt_array = get_gt_array()
    print('gt_array {}'.format(gt_array))
    filenames = sorted_alphanumeric(os.listdir(path))
    print(filenames)
    for filename in filenames:
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".JPG"):
            print(filename)
            image_path = os.path.join(path,filename)
            image= cv2.imread(image_path)
            #dilate, canny, cannyBbox = canny_cv(image)
            hough, houghBbox= hough_detection(image)
            #print("canny", cannyBbox)
            #print("hough", houghBbox)    
            annotations = assign_coin_class(houghBbox, image)
            #cv2.imshow('image',imgStack)
            if annotations:
                annotations_array.append(annotations)
                display_annotations(image, annotations, filename)
        else:
            continue 
    print('annotations_array {}'.format(annotations_array))
    map = calculate_map(annotations_array, gt_array, 0.5,'interp')
    print('map {}'.format(map))
    
main()