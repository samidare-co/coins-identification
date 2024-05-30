import copy
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt

def images_stack(images,win_names):
    """
    Afficher plusieurs images soit horizontalement ou verticalement
    :param images: liste des images a afficher 
    :param win_names: liste des noms des fenetres
    """
    formattedImages = []
    for image in images:
        if len(image.shape) == 2:  
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  
        formattedImages.append(image)
        
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Horizontal
    img_stack = np.hstack(formattedImages)
    for index, name in enumerate(win_names):
        image = cv2.putText(img_stack, f'{index + 1}. {name}', (5 + formattedImages[0].shape[1] * index, 30),
                            font, 1, (205, 0, 255), 2, cv2.LINE_AA) 

    # Vertical
    """ img_stack = np.vstack(formattedImages)
    for index, name in enumerate(win_names):
        image = cv2.putText(img_stack, f'{index + 1}. {name}', (5, 30 + formattedImages[0].shape[1] * index),
                            font, 1, (205, 0, 255), 2, cv2.LINE_AA) """

    cv2.imshow("Image Processing", img_stack)

def stackImages(_imgList, cols, scale, winNames):
    """
    Afficher plusieurs images horizontalement ET/OU verticalement
    :param _imgList: liste des images a afficher 
    :param cols: nombre d'images dans une ligne  
    :param scale: echelle de l'image 
    :return: image stack 
    """
    imgList = copy.deepcopy(_imgList)

    width1, height1 = imgList[0].shape[1], imgList[0].shape[0]

    totalImages = len(imgList)
    rows = totalImages // cols if totalImages // cols * cols == totalImages else totalImages // cols + 1
    blankImages = cols * rows - totalImages

   
    imgBlank = np.zeros((height1, width1, 3), np.uint8)
    imgList.extend([imgBlank] * blankImages)

 
    for i in range(cols * rows):
        imgList[i] = cv2.resize(imgList[i], (width1, height1), interpolation=cv2.INTER_AREA)
        imgList[i] = cv2.resize(imgList[i], (0, 0), None, scale, scale)

        if len(imgList[i].shape) == 2:  
            imgList[i] = cv2.cvtColor(imgList[i], cv2.COLOR_GRAY2BGR)

    hor = [imgBlank] * rows
    for y in range(rows):
        line = []
        for x in range(cols):
            line.append(imgList[y * cols + x])
        hor[y] = np.hstack(line)
    ver = np.vstack(hor)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for index, name in enumerate(winNames):
        image = cv2.putText(ver, f'{index + 1}. {name}', (5 + imgList[0].shape[1] * (index%cols), (imgList[0].shape[0] * (index // cols))+60),
                            font, 2, (205, 0, 255), 2, cv2.LINE_AA) 
    return ver

    
def display_annotations(image, bboxes, filename): 
    for key, boxes in bboxes.items():
        for box in boxes:
            x1, y1, x2, y2, score = box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, key, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.imsave('output/'+filename, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

def read_csv_annotations() : 
    with open('input/tests/_annotations.csv') as annotations : 
        lines = annotations.readlines()
        return lines

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)