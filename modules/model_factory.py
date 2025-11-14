import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def _replace_head(model, num_classes: int):
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def get_model(name: str, num_classes: int, pretrained: bool = True):
    if name == "fasterrcnn_resnet50_fpn_v2":
        if pretrained:
            weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            # não passe num_classes com weights; substitua o head depois
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
                weights=weights, weights_backbone=None
            )
            if num_classes is not None and num_classes != 91:
                model = _replace_head(model, num_classes)
            return model
        else:
            return torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
                weights=None, weights_backbone=None, num_classes=num_classes
            )

    if name == "fasterrcnn_resnet50_fpn":
        if pretrained:
            weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights=weights, weights_backbone=None
            )
            if num_classes is not None and num_classes != 91:
                model = _replace_head(model, num_classes)
            return model
        else:
            return torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights=None, weights_backbone=None, num_classes=num_classes
            )

    raise ValueError(f"Unknown model name: {name}")