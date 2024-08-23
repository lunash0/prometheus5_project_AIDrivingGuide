# AI Driving Guide project - Traffic Lights Detection

<br>[

> This README file provides a description of one of the project's many tasks: the traffic lights detection task. For detailed information about overall information about the project or other detection task, please refer to the README files below.  
>   
> - [main README.md](.README.md)
> - [Pedestrian Detection README.md](./Pedestrian-Detection/README.md)
> - [Road Lane Detection README.md](./Lane-Detection/README.md)
> - [Road Sign Detection README.md](./RoadSign-Detection/README.md)


<br><br>
## 1. Introduction
- We use retinanet_resnet50_fpn_v2 model model to detect traffic lights.
- Not only can it distinguish red/green/yellow lights, but it can also distinguish information about left turns and right turns.

  <br><br><br>

## Model

- Using the pretrained retinanet_resnet50_fpn_v2 model provided by torchvision, we performed additional training to fit custom dataset.
- Constructs an improved RetinaNet model with a ResNet-50-FPN backbone.

- The parameters are as follows.
  - weights (RetinaNet_ResNet50_FPN_V2_Weights, optional) ? The pretrained weights to use. See RetinaNet_ResNet50_FPN_V2_Weights below for more details, and possible values. By default, no pre-trained weights are used.
  - progress (bool) ? If True, displays a progress bar of the download to stderr. Default is True.
  - num_classes (int, optional) ? number of output classes of the model (including the background)
  - weights_backbone (ResNet50_Weights, optional) ? The pretrained weights for the backbone.
  - trainable_backbone_layers (int, optional) ? number of trainable (not frozen) layers starting from final block. Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable. If None is passed (the default) this value is set to 3.
  - **kwargs ? parameters passed to the torchvision.models.detection.RetinaNet base class. Please refer to the source code for more details about this class.

  <br><br><br>

## 3. Training

### Dataset
- We used a custom dataset, which was downloaded [AIhub's traffic light](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71579) dataset and refined to suit our purposes.
  - Data containing vehicle signal light information was selected from various signal information (vehicle signal lights, crosswalk signal lights, etc.).
  - Only the necessary content was extracted and used from the annotation file.

- You can use your own dataset.  
  (You need to modify the config file contents to suit the dataset you are using.)

<br/>

### Training command
- We can execute the following command to start the training. 
- We will monitor the mAP metric for saving the best model.
```commandline
python train.py
```
<br><br><br>

## 4. Result
The graphs for mAP@0.50:095 and mAP@0.50 are as follows.(in 20 epochs)
<div align="center">
  <img src="./IMG/mAP_images.jpeg" style="width:1000px;">
</div>
