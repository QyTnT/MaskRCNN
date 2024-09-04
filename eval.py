import os
import os.path as osp

import tensorflow as tf
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from mask_rcnn import MASK_RCNN
from utils.utils import get_classes, get_coco_label_map
from utils.utils_map import Make_json, prep_metrics

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import numpy as np

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def calculate_miou(test_coco, mask_dets):
    """
    计算mIoU指标
    """
    iou_sum = 0
    num_images = len(mask_dets.getImgIds())
    for img_id in mask_dets.getImgIds():
        img_anns = test_coco.loadAnns(test_coco.getAnnIds(imgIds=img_id))
        det_anns = mask_dets.loadAnns(mask_dets.getAnnIds(imgIds=img_id))

        for det_ann in det_anns:
            det_mask = test_coco.annToMask(det_ann)
            max_iou = 0
            for img_ann in img_anns:
                img_mask = test_coco.annToMask(img_ann)
                intersection = np.logical_and(det_mask, img_mask).sum()
                union = np.logical_or(det_mask, img_mask).sum()
                if union == 0:
                    continue
                iou = intersection / union
                max_iou = max(max_iou, iou)

            iou_sum += max_iou

    miou = iou_sum / num_images
    return miou

def calculate_precision_recall(test_coco, mask_dets,iou_thr=0.5):
    all_true_labels = []
    all_scores = []

    for img_id in mask_dets.getImgIds():
        img_anns = test_coco.loadAnns(test_coco.getAnnIds(imgIds=img_id))
        det_anns = mask_dets.loadAnns(mask_dets.getAnnIds(imgIds=img_id))

        for det_ann in det_anns:
            det_mask = test_coco.annToMask(det_ann)
            for img_ann in img_anns:
                img_mask = test_coco.annToMask(img_ann)
                intersection = np.logical_and(det_mask, img_mask).sum()
                union = np.logical_or(det_mask, img_mask).sum()
                if union == 0:
                    continue
                iou = intersection / union
                all_scores.append(det_ann['score'])
                all_true_labels.append(1 if iou > iou_thr else 0)

    precision, recall, _ = precision_recall_curve(all_true_labels, all_scores)
    pr_auc = auc(recall, precision)
    return precision, recall, pr_auc

if __name__ == '__main__':
    #------------------------------------------------------------------------------------------------------------------#
    #   map_mode用于指定该文件运行时计算的内容
    #   map_mode为0代表整个map计算流程，包括获得预测结果、计算指标。
    #   map_mode为1代表仅仅获得预测结果。
    #   map_mode为2代表仅仅计算指标。
    #-------------------------------------------------------------------------------------------------------------------#
    map_mode        = 0
    #-------------------------------------------------------#
    #   评估自己的数据集必须要修改
    #   所需要区分的类别对应的txt文件
    #-------------------------------------------------------#
    classes_path    = 'model_data/crack_classes .txt'
    #-------------------------------------------------------#
    #   获得测试用的图片路径和标签
    #   默认指向根目录下面的datasets/coco文件夹
    #-------------------------------------------------------#
    Image_dir       = "./datasets/coco/JPEGImages"
    Json_path       = "./datasets/coco/Jsons/test_annotations.json"
    #-------------------------------------------------------#
    #   结果输出的文件夹，默认为map_out
    #   里面存放了一些json文件，主要是检测结果。
    #-------------------------------------------------------#
    map_out_path    = 'map_out'
    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#
    test_coco       = COCO(Json_path)
    class_names, _  = get_classes(classes_path)
    COCO_LABEL_MAP  = get_coco_label_map(test_coco, class_names)
    
    ids         = list(test_coco.imgToAnns.keys())

    #------------------------------------#
    #   创建文件夹
    #------------------------------------#
    if not osp.exists(map_out_path):
        os.makedirs(map_out_path)
        
    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        yolact      = MASK_RCNN(confidence = 0.05, nms_iou = 0.5)
        print("Load model done.")
        
        print("Get predict result.")
        make_json   = Make_json(map_out_path, COCO_LABEL_MAP)
        for i, id in enumerate(tqdm(ids)):
            image_path  = osp.join(Image_dir, test_coco.loadImgs(id)[0]['file_name'])
            image       = Image.open(image_path)
            mask_path = osp.join("mask",test_coco.loadImgs(id)[0]['file_name'])

            box_thre, class_thre, class_ids, masks_arg, masks_sigmoid = yolact.get_map_out(image)

            converted_mask = np.where(masks_sigmoid, 255, 0).astype(np.uint8)
            converted_mask = np.squeeze(converted_mask, axis=2)
            from PIL import Image
            img = Image.fromarray(converted_mask)
            img.save(mask_path)
            if box_thre is None:
                continue
            prep_metrics(box_thre, class_thre, class_ids, masks_sigmoid, id, make_json)
        make_json.dump()
        print(f'\nJson files dumped, saved in: \'eval_results/\', start evaluting.')

    if map_mode == 0 or map_mode == 2:
        bbox_dets = test_coco.loadRes(osp.join(map_out_path, "bbox_detections.json"))
        mask_dets = test_coco.loadRes(osp.join(map_out_path, "mask_detections.json"))

        print('\nEvaluating BBoxes:')
        bbox_eval = COCOeval(test_coco, bbox_dets, 'bbox')
        bbox_eval.evaluate()
        bbox_eval.accumulate()
        bbox_eval.summarize()

        print('\nEvaluating Masks:')
        bbox_eval = COCOeval(test_coco, mask_dets, 'segm')
        bbox_eval.evaluate()
        bbox_eval.accumulate()
        bbox_eval.summarize()

        print('\nCalculating mIoU:')
        miou = calculate_miou(test_coco, mask_dets)
        print(f'\nmIoU: {miou}')

        print('\nCalculating Precision-Recall Curve:')
        iou_thrs = [0.5,0.6,0.7,0.8,0.9]
        plt.figure()
        for iou_thr in iou_thrs:
            precision, recall, pr_auc = calculate_precision_recall(test_coco, mask_dets,iou_thr)
            print(f'PR AUC: {pr_auc}')
            plt.plot(recall, precision,label=f'IoU={iou_thr:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.savefig(osp.join(map_out_path, 'pr_curve.png'))
        plt.show()
