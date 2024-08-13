from torchvision.models.detection.retinanet import retinanet_resnet50_fpn, RetinaNetClassificationHead
import torch 

def build_model(num_classes):
    model = retinanet_resnet50_fpn(pretrained=True)
    in_features = model.head.classification_head.conv[0][0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(in_features, num_anchors, num_classes)
    
    return model

def load_model(checkpoint_path, num_classes, device):
    model = build_model(num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
    model.to(device)
    model.eval()
    return model