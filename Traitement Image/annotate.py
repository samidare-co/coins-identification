import  cv2
import copy
import os
import argparse
from matplotlib import pyplot as plt
import numpy as np
from utils import stackImages, display_annotations
from detect import canny_cv, hough_detection


""" ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())


image_path = args['image'] """




def getCoinColor(coin, img, plot):
    x1, y1, x2, y2 = coin
    # Centre de la pièce 
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    # Rayon de la piece 
    r = (x2 - x1) // 2
    # Rayon de la piece sans la bordure 
    small_r = int(r * 0.85)

    width,height = img.shape[1], img.shape[0]
    # Créer un masque qui recouvre la pièce entière 
    mask = np.zeros_like(img)
    
    cv2.circle(mask, (cx, cy), int(r*0.95), (255, 255, 255), -1)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # Créer un masque plus petit qui recouvre le centre de la pièce 
    small_mask = np.zeros_like(img)
    
    cv2.circle(small_mask, (cx, cy), small_r, (255, 255, 255), -1)
    small_mask = cv2.cvtColor(small_mask, cv2.COLOR_BGR2GRAY)
    imageWithBorder = img.copy()

    border_colors = []
    center_colors = []
    for i in range(cx - r, cx + r):
        for j in range(cy - r, cy + r):
            if (j>=img.shape[0] or i>=img.shape[1]):
                continue
            # Couleurs entre le masque original et le masque plus petit

            if mask[j, i] == 255 and small_mask[j, i] == 0:
                hsv_color = cv2.cvtColor(np.uint8([[img[j, i]]]), cv2.COLOR_BGR2HSV)
                #
                border_colors.append(hsv_color[0, 0])
                imageWithBorder[j, i] = [0, 0, 255]
                  
            # Couleurs dans le masque plus petit
            if mask[j, i] == 255 and small_mask[j, i] == 255:
                hsv_color = cv2.cvtColor(np.uint8([[img[j, i]]]), cv2.COLOR_BGR2HSV)
                #
                center_colors.append(hsv_color[0, 0])

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_sat = np.mean(hsv_img[:, 1,:])
    
    border_hue_avg = np.mean([color[0] for color in border_colors])
    center_hue_avg = np.mean([color[0] for color in center_colors])
        
    border_sat_avg = np.mean([color[1] for color in border_colors])
    center_sat_avg = np.mean([color[1] for color in center_colors])
    if plot == True:
        fig, axs = plt.subplots(1, 4, figsize=(10, 5))
        axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[0].set_title('Original Image')
        axs[1].imshow(mask, cmap='gray')
        axs[1].set_title('Mask')
        axs[2].imshow(small_mask, cmap='gray')
        axs[2].set_title('Small Mask')
        axs[3].imshow(cv2.cvtColor(imageWithBorder, cv2.COLOR_RGB2BGR))
        plt.show()
    return border_hue_avg, center_hue_avg, border_sat_avg, center_sat_avg
        
    
def detectColor(coins  , image, plot):
    img = copy.deepcopy(image)
    detect_dict = {'2 euro' : [], '1 euro': [], 'bronze': [], 'gold': [], 'unknown': []}
    diff_sat =[]
    border_saturations= []
    center_saturations = []
    center_hues = []
    for coin in coins:
        # Calculer la moyenne des couleurs de la bordure et du centre
        border_hue_avg, center_hue_avg, border_sat_avg, center_sat_avg = getCoinColor(coin, img, plot)
        center_hues.append(center_hue_avg)
        
        border_saturations.append(border_sat_avg)
        center_saturations.append(center_sat_avg)
        
        diff_sat.append(abs(border_sat_avg - center_sat_avg))
        

        
        
        
        
        

     
    possible_euros = mean_outliers(diff_sat)
    
    
    bronze_hue = 13
    for possible_euros_i in possible_euros:
        if center_hues[possible_euros_i] > bronze_hue:
            if center_saturations[possible_euros_i] > border_saturations[possible_euros_i]:
                detect_dict['2 euro'].append(coins[possible_euros_i])
            else :
                detect_dict['1 euro'].append(coins[possible_euros_i])
    for i in range(len(coins)):
        if i not in possible_euros or i in possible_euros and center_hues[i] < bronze_hue:
            #detect_dict['unknown'].append(coins[i])
            if center_hues[i] > bronze_hue:
                detect_dict['gold'].append(coins[i])
            else:
                detect_dict['bronze'].append(coins[i])

    
    return detect_dict


def mean_outliers(data):
    mean = np.mean(data)
    return [i for i, x in enumerate(data) if x > mean]
    
def std_outliers(data):
    std = np.std(data)
    mean = np.mean(data)
    
    z_score = [(x - mean) / std for x in data]
    
    #return [i for i, x in enumerate(data) if x > 2*std]
    return [z_score.index(x) for x in z_score if x > 2]

def percentile_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1 * IQR
    upper_bound = Q3 + 1 * IQR
    outliers_indices = [i for i, x in enumerate(data) if x <lower_bound or x>upper_bound]
    
    return outliers_indices


def calculate_diameter_ratio(box1, box2):
    firstBoxDiameter= box1[3] - box1[1]
    secondBoxDiameter= box2[3] - box2[1]
    ratio = max(firstBoxDiameter, secondBoxDiameter) / min(firstBoxDiameter, secondBoxDiameter)
    return firstBoxDiameter, secondBoxDiameter, ratio
    


def assign_coin_class(boxes, image, plot=False):
    
    print('boxes {}'.format(boxes))
    
    detected_classes= {"2 euro": [], "1 euro": [], "50 centimes": [], "20 centimes": [], "10 centimes": [], "5 centimes": [], "2 centimes": [], "1 centime": []}
    detectedColors = detectColor(boxes, image, plot)

    if len(boxes) < 2:
        # Ignorer les images avec 1 seul piece
        return
        #return detectedColors
    ratios = {
        "2 euro": {"color": "2 euro", "values": [("2 euro", 1, "2 euro"),("1 euro", 1.1075, "1 euro"), ("50 centimes", 1.0619, "gold"), ("20 centimes", 1.1573, "gold"), ("5 centimes", 1.2118, "bronze"), ("10 centimes", 1.3038, "gold"), ("2 centimes", 1.3733, "bronze"), ("1 centime", 1.5846, "bronze")]},
        "50 centimes": {"color": "gold", "values": [("50 centimes", 1, "gold"),("1 euro", 1.0430, "1 euro"), ("20 centimes", 1.0899, "gold"), ("5 centimes", 1.1412, "bronze"), ("10 centimes", 1.2278, "gold"), ("2 centimes", 1.2933, "bronze"), ("1 centime", 1.4923, "bronze")]},
        "1 euro": {"color": "1 euro", "values": [("1 euro", 1, "1 euro"), ("20 centimes", 1.0450, "gold"), ("5 centimes", 1.0941, "bronze"), ("10 centimes", 1.1772, "gold"), ("2 centimes", 1.24, "bronze"), ("1 centime", 1.4308, "bronze")]},
        "20 centimes": {"color": "gold", "values": [("20 centimes", 1, "gold"),("5 centimes", 1.0471, "bronze"), ("10 centimes", 1.1266, "gold"), ("2 centimes", 1.1867, "bronze"), ("1 centime", 1.3692, "bronze")]},
        "5 centimes": {"color": "bronze", "values": [("5 centimes", 1, "bronze"), ("10 centimes", 1.0759, "gold"), ("2 centimes", 1.133, "bronze"), ("1 centime", 1.3077, "bronze")]},
        "10 centimes": {"color": "gold", "values": [("10 centimes", 1, 'gold'), ("2 centimes", 1.0533, "bronze"), ("1 centime", 1.2154, "bronze")]},
        "2 centimes": {"color": "bronze", "values": [("2 centimes", 1, "bronze"), ("1 centime", 1.1538, "bronze")]}
    }
    first_box = {}
    for key, value in detectedColors.items():
        if value and isinstance(value[0], list):
            first_box[key] = value.pop(0) + [1]
            break
   
    first_box_coordinates = list(first_box.values())[0]
    first_box_color = list(first_box.keys())[0]
    first_box_is_assigned = False
    for detColor, boxes in detectedColors.items(): 
        for box in boxes:
            
            
            
            first_box_diameter, second_box_diameter, ratio = calculate_diameter_ratio(first_box_coordinates,box)
            nearest_ratio = 100
            first_class = ''
            second_class = ''
        
            for coin, data in ratios.items():
                coin_color = data["color"]
                
                if (coin_color != first_box_color):
                    continue
                classes = data["values"]
                
                for c, r, color in classes:
                    if color != detColor:
                        continue
                    
                    if abs(ratio - r) < nearest_ratio:
                        
                        nearest_ratio = abs(ratio - r)
                        first_class = coin
                        second_class = c
            if first_box_diameter > second_box_diameter:
                if (not(first_box_is_assigned)):
                    first_box_is_assigned = True
                    detected_classes[first_class].append(first_box_coordinates)
                    
                confidence_score_box = box + [1]
                
                detected_classes[second_class].append(confidence_score_box)
            else:
                if (not(first_box_is_assigned)):
                    first_box_is_assigned = True
                    detected_classes[second_class].append(first_box_coordinates)
                confidence_score_box = box + [1]
                
                detected_classes[first_class].append(confidence_score_box)

    
    # TODO : ajouter une propiete circle en plus des boundinx box pour eviter de les recalculer dans assign_coin_class et getCoinColor

    return detected_classes






def annotate_main() : 
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
            annotations = assign_coin_class(houghBbox , image, plot = True)
            #cv2.imshow('image',imgStack)
            if annotations:
                display_annotations(image, annotations, filename)
        else:
            continue
        
#annotate_main()

