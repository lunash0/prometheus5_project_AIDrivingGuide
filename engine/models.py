from torchvision.models.detection.retinanet import retinanet_resnet50_fpn, RetinaNetClassificationHead
import torch 
import torchvision
from functools import partial
# sys.path.append('/home/yoojinoh/Others/PR/prometheus5_project_AIDrivingGuide/')
from models.Lane_Detection.model.lanenet import LaneNet

"""1. Pedestrian Detection Model"""
def build_ped_model(num_classes):
    model = retinanet_resnet50_fpn(pretrained=True)
    
    in_features = model.head.classification_head.conv[0][0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(in_features, num_anchors, num_classes)
    
    return model

def load_ped_model(checkpoint_path, num_classes, device):
    model = build_ped_model(num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
    model.to(device)
    model.eval()
    return model


"""2. Traffic Light Detection Model"""
def build_tl_model(num_classes=91):
    model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
        weights='DEFAULT'
    )
    num_anchors = model.head.classification_head.num_anchors

    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=partial(torch.nn.GroupNorm, 32)
    )
    return model

def load_tl_model(checkpoint_path, num_classes, device):
    model = build_tl_model(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model 


"""3. Lane Detection Model"""

def build_lane_model(device, model_type='ENet'):
    model = LaneNet(DEVICE=device, arch=model_type)
    return model
def load_lane_model(checkpoint_path, model_type, device):
    model = build_lane_model(device, model_type)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model