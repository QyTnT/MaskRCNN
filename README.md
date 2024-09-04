## Mask-Rcnn-实例分割模型在Tensorflow2当中的实现
---

## 所需环境
tensorflow-gpu==2.2.0

## 文件下载
这个训练好的权重是基于coco数据集的，可以直接运行用于coco数据集的实例分割。  
链接: https://pan.baidu.com/s/1ES_j_ilclWneT4GbY7VQdg    
提取码: gcgy    
   

## 训练步骤 

### b、训练自己的数据集
1. 数据集的准备  
**本文使用labelme工具进行标注，标注好的文件有图片文件和json文件，二者均放在before文件夹里，具体格式可参考shapes数据集。**    

2. 数据集的处理  
修改coco_annotation.py里面的参数。第一次训练可以仅修改classes_path，classes_path用于指向检测类别所对应的txt。    
训练自己的数据集时，可以自己建立一个cls_classes.txt，里面写自己所需要区分的类别。    
 
修改coco_annotation.py中的classes_path，使其对应cls_classes.txt，并运行coco_annotation.py。    

3. 开始网络训练  
**训练的参数较多，均在train.py中，大家可以在下载库后仔细看注释，其中最重要的部分依然是train.py里的classes_path。**   
**classes_path用于指向检测类别所对应的txt，这个txt和coco_annotation.py里面的txt一样！训练自己的数据集必须要修改！**    
修改完classes_path后就可以运行train.py开始训练了，在训练多个epoch后，权值会生成在logs文件夹中。   

4. 训练结果预测  
训练结果预测需要用到两个文件，分别是mask_rcnn.py和predict.py。
首先需要去mask_rcnn.py里面修改model_path以及classes_path，这两个参数必须要修改。    
**model_path指向训练好的权值文件，在logs文件夹里。   
classes_path指向检测类别所对应的txt。**     
完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。     

### c、训练coco数据集
1. 数据集的准备  
coco训练集 http://images.cocodataset.org/zips/train2017.zip   
coco验证集 http://images.cocodataset.org/zips/val2017.zip   
coco训练集和验证集的标签 http://images.cocodataset.org/annotations/annotations_trainval2017.zip   

2. 开始网络训练  
解压训练集、验证集及其标签后。打开train.py文件，修改其中的classes_path指向model_data/coco_classes.txt。   
修改train_image_path为训练图片的路径，train_annotation_path为训练图片的标签文件，val_image_path为验证图片的路径，val_annotation_path为验证图片的标签文件。


## 预测步骤
### a、使用预训练权重
1. 将预训练模型权重放入model_data，运行predict.py，输入   
```python
img/street.jpg
```
2. 在predict.py里面进行设置可以进行fps测试和video视频检测。   

3. 运行predict.py，输入    
```python
img/street.jpg
```
4. 在predict.py里面进行设置可以进行fps测试和video视频检测。    

## 评估步骤 
### a、评估自己的数据集
1. 本文使用coco格式进行评估。    
2. 如果在训练前已经运行过coco_annotation.py文件，代码会自动将数据集划分成训练集、验证集和测试集。
3. 如果想要修改测试集的比例，可以修改coco_annotation.py文件下的trainval_percent。trainval_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1。train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1。
4. 在mask_rcnn.py里面修改model_path以及classes_path。**model_path指向训练好的权值文件，在logs文件夹里。classes_path指向检测类别所对应的txt。**    
5. 前往eval.py文件修改classes_path，classes_path用于指向检测类别所对应的txt，这个txt和训练时的txt一样。评估自己的数据集必须要修改。运行eval.py即可获得评估结果。  

### b、评估coco的数据集
1. 下载好coco数据集。  
2. 在mask_rcnn.py里面修改model_path以及classes_path。**model_path指向coco数据集的权重，在logs文件夹里。classes_path指向model_data/coco_classes.txt。**    
3. 前往eval.py设置classes_path，指向model_data/coco_classes.txt。修改Image_dir为评估图片的路径，Json_path为评估图片的标签文件。 运行eval.py即可获得评估结果。  
  
## Reference
https://github.com/matterport/Mask_RCNN     
https://github.com/feiyuhuahuo/Yolact_minimal     
