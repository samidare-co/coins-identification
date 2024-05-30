from utils import read_csv_annotations
import numpy as np
def get_gt_array():
    lines= read_csv_annotations()
    gt_array = []
    images_done= []
    for line_num in range(1, len(lines)):
        line = lines[line_num]

        
        image ={"2 euro": [], "1 euro": [], "50 centimes": [], "20 centimes": [], "10 centimes": [], "5 centimes": [], "2 centimes": [], "1 centime": []}
        data = line.split(',') 
        
        image_name = data[0]
        # si l'image suivante n'est pas la meme que l'image courante et que l'image courante n'a pas deja ete traitee = enlever les images qui n'ont qu'une seule classe / piece
        if (line_num==len(lines)-1 ):
            if image_name not in images_done:
                continue
        elif (lines[line_num+1].split(',')[0] != image_name and image_name not in images_done):
            continue
        class_name = data[3]
        if class_name == 'verso-dore' or class_name == 'verso-bronze':
            # cas ou il n y'a que des classes verso
            if image_name not in images_done:
                gt_array.append(image)
                images_done.append(image_name)
            continue
        x1 = int(data[4])
        y1 = int(data[5])
        x2 = int(data[6])
        y2 = int(data[7])
        if image_name in images_done:
            gt_array[-1][class_name].append([x1, y1, x2, y2])
        else :
            image[class_name].append([x1, y1, x2, y2])
            gt_array.append(image)
            images_done.append(image_name)
    return gt_array

def get_iou(det, gt):
    
    
    det_x1, det_y1, det_x2, det_y2= det
    gt_x1, gt_y1, gt_x2, gt_y2 = gt

    x_left = max(det_x1, gt_x1)
    y_top = max(det_y1, gt_y1)
    x_right = min(det_x2, gt_x2)
    y_bottom = min(det_y2, gt_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    area_intersection = (x_right - x_left) * (y_bottom - y_top)
    det_area = (det_x2 - det_x1) * (det_y2 - det_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    area_union = det_area + gt_area - area_intersection + 1e-9
    iou = area_intersection / area_union
    return iou
   

def calculate_map(det_boxes, gt_boxes, iou_threshold = 0.5, method='area') : 
    print('det_boxes {}'.format(det_boxes))
    """
        det_boxes = [
            {
                '1 euro': [[x1,y1,x2,y2, score], [x1,y1,x2,y2, score], ...],
                '20 centimes' : [[x1,y1,x2,y2, score], [x1,y1,x2,y2, score], ...],
                ...
            }, 
            {det_boxes_img_2}, 
            ...
            {det_boxes_img_N},
        ]

        gt_boxes = [
            {
                '1 euro': [[x1,y1,x2,y2], [x1,y1,x2,y2], ...],
                '20 centimes' : [[x1,y1,x2,y2], [x1,y1,x2,y2], ...],
            }, 
            {gt_boxes_img_2},
            ...
            {gt_boxes_img_N},
        ]
    """
    gt_labels = {cls_key for im_gt in gt_boxes for cls_key in im_gt.keys()} 

    
    
    # average precision pour toutes les classes
    aps = []
    for idx, label in enumerate(gt_labels):
        
        
        # recuperer les detections pour la classe label 
        cls_dets = [[im_idx,im_dets_label] for im_idx, im_dets in enumerate(det_boxes) for im_dets_label in im_dets[label]]
        if len(cls_dets) == 0:
            continue

        """
            Pour N images :
            cls_dets = [
                [0, [x1,y1,x2,y2, score]],
                [0, [x1,y1,x2,y2, score]],
                ...
                [N, [x1,y1,x2,y2, score]],
            ]
        """

        # Trier par score de confiance dans l'ordre decroissant
        cls_dets = sorted(cls_dets, key = lambda k :-k[1][-1])
        
        # Quelle boxe de la verite terrain a deja ete associee
        gt_matched = [[False for _ in im_gts[label]] for im_gts in gt_boxes]

        # Nombre de boxes verite terrain pour la classe label pour calculer le recall
        num_gts = sum([len(im_gts[label]) for im_gts in gt_boxes])

        tp = [0] * len(cls_dets)
        fp = [0] * len(cls_dets)

        # Pour chaque detection
        for det_idx, (im_idx,det_pred) in enumerate(cls_dets): 
            # Recuperer les boxes verite terrain pour l'image im_idx 
            im_gts = gt_boxes[im_idx][label]
            max_iou_found = -1 
            max_iou_gt_idx = -1 
            
            # Meilleur boxe verite terrain pour la detection courante
            for gt_box_idx, gt_box in enumerate(im_gts):
                gt_box_iou = get_iou(det_pred[:-1], gt_box)
                if gt_box_iou > max_iou_found:
                    max_iou_found = gt_box_iou
                    max_iou_gt_idx = gt_box_idx
            if max_iou_found < iou_threshold or gt_matched[im_idx][max_iou_gt_idx]:
                fp[det_idx] = 1
            else :
                tp[det_idx] = 1
                gt_matched[im_idx][max_iou_gt_idx] = True
                
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        eps = np.finfo(np.float32).eps
        recalls = tp /np.maximum(num_gts, eps)
        precisions = tp / np.maximum(tp + fp, eps)
        
        if method == 'area':
            recalls = np.concatenate(([0,0], recalls, [1,0]))
            precisions = np.concatenate(([0,0], precisions, [0,0]))
            
            # Rempalcement des valeurs de precision pour les valeurs de recall superieures a r 
            for i in range(precisions.size -1, 0,-1):
                precisions[i-1] = np.maximum(precisions[i-1], precisions[i])
            
            i = np.where(recalls[1:] != recalls[:-1])[0]
            ap = np.sum((recalls[i+1] - recalls[i]) * precisions[i+1])

        elif method =='interp':
            ap = 0.0 
            for interp_pt in np.arange(0, 1+1E-3,0.1):
                prec_interp_pt = precisions[recalls >= interp_pt]
                
                prec_interp_pt = prec_interp_pt.max() if prec_interp_pt.size > 0 else 0
                ap+= prec_interp_pt
            ap /= 11
        
        else : 
            raise ValueError('Method doit etre interp ou area') 
        
        aps.append(ap)
    mean_ap = sum(aps) / len(aps)
    return mean_ap

