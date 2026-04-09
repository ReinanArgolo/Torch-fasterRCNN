import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator

def _replace_head(model, num_classes: int):
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def _maybe_set_rpn_topn(model, pre_nms_train=None, pre_nms_test=None, post_nms_train=None, post_nms_test=None):
    rpn = getattr(model, "rpn", None)
    if rpn is None:
        return

    def _set_topn(base_attr: str, train_val, test_val):
        if train_val is None and test_val is None:
            return
        # TorchVision stores the values in private dicts and exposes callable methods.
        # We must update the private dicts: _pre_nms_top_n / _post_nms_top_n
        private_attr = f"_{base_attr}"
        current = getattr(rpn, private_attr, None)
        if isinstance(current, dict):
            new_dict = dict(current)
            if train_val is not None:
                new_dict["training"] = int(train_val)
            if test_val is not None:
                new_dict["testing"] = int(test_val)
            setattr(rpn, private_attr, new_dict)
        else:
            # Construct the expected dict if it's missing or not a dict
            training_val = int(train_val) if train_val is not None else (
                current.get("training") if isinstance(current, dict) else 2000
            )
            testing_val = int(test_val) if test_val is not None else (
                current.get("testing") if isinstance(current, dict) else 1000
            )
            setattr(rpn, private_attr, {"training": training_val, "testing": testing_val})

    _set_topn("pre_nms_top_n", pre_nms_train, pre_nms_test)
    _set_topn("post_nms_top_n", post_nms_train, post_nms_test)


def _maybe_set_roi_params(model, box_score_thresh=None, box_nms_thresh=None, box_detections_per_img=None):
    roi = getattr(model, "roi_heads", None)
    if roi is None:
        return
    if box_score_thresh is not None and hasattr(roi, "score_thresh"):
        roi.score_thresh = float(box_score_thresh)
    if box_nms_thresh is not None and hasattr(roi, "nms_thresh"):
        roi.nms_thresh = float(box_nms_thresh)
    if box_detections_per_img is not None and hasattr(roi, "detections_per_img"):
        roi.detections_per_img = int(box_detections_per_img)


def get_model(name: str, num_classes: int, pretrained: bool = True, **kwargs):
    """Factory for detection models.

    Optional kwargs (all optional):
      - anchor_sizes: list[list[int]] or list[tuple[int,...]] sizes per FPN level
      - aspect_ratios: list[list[float]] per FPN level
      - rpn_pre_nms_top_n_train / rpn_pre_nms_top_n_test
      - rpn_post_nms_top_n_train / rpn_post_nms_top_n_test
      - box_score_thresh / box_nms_thresh / box_detections_per_img
    """
    anchor_sizes = kwargs.get("anchor_sizes")
    aspect_ratios = kwargs.get("aspect_ratios")

    rpn_anchor_generator = None
    if anchor_sizes is not None or aspect_ratios is not None:
        # Defaults aligned with torchvision for FPN (5 levels)
        sizes = anchor_sizes or [[32], [64], [128], [256], [512]]
        ratios = aspect_ratios or [[0.5, 1.0, 2.0]] * len(sizes)
        rpn_anchor_generator = AnchorGenerator(sizes=tuple(tuple(s) for s in sizes), aspect_ratios=tuple(tuple(r) for r in ratios))
    if name == "fasterrcnn_resnet50_fpn_v2":
        if pretrained:
            weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            # não passe num_classes com weights; substitua o head depois
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
                weights=weights, weights_backbone=None
            )
            # Evitar conflito de kwargs duplicados: ajuste o anchor generator pós-criação
            if rpn_anchor_generator is not None and hasattr(model, "rpn"):
                model.rpn.anchor_generator = rpn_anchor_generator
            if num_classes is not None and num_classes != 91:
                model = _replace_head(model, num_classes)
            _maybe_set_rpn_topn(
                model,
                pre_nms_train=kwargs.get("rpn_pre_nms_top_n_train"),
                pre_nms_test=kwargs.get("rpn_pre_nms_top_n_test"),
                post_nms_train=kwargs.get("rpn_post_nms_top_n_train"),
                post_nms_test=kwargs.get("rpn_post_nms_top_n_test"),
            )
            _maybe_set_roi_params(
                model,
                box_score_thresh=kwargs.get("box_score_thresh"),
                box_nms_thresh=kwargs.get("box_nms_thresh"),
                box_detections_per_img=kwargs.get("box_detections_per_img"),
            )
            return model
        else:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
                weights=None, weights_backbone=None, num_classes=num_classes
            )
            if rpn_anchor_generator is not None and hasattr(model, "rpn"):
                model.rpn.anchor_generator = rpn_anchor_generator
            return model

    if name == "fasterrcnn_resnet50_fpn":
        if pretrained:
            weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights=weights, weights_backbone=None
            )
            if rpn_anchor_generator is not None and hasattr(model, "rpn"):
                model.rpn.anchor_generator = rpn_anchor_generator
            if num_classes is not None and num_classes != 91:
                model = _replace_head(model, num_classes)
            _maybe_set_rpn_topn(
                model,
                pre_nms_train=kwargs.get("rpn_pre_nms_top_n_train"),
                pre_nms_test=kwargs.get("rpn_pre_nms_top_n_test"),
                post_nms_train=kwargs.get("rpn_post_nms_top_n_train"),
                post_nms_test=kwargs.get("rpn_post_nms_top_n_test"),
            )
            _maybe_set_roi_params(
                model,
                box_score_thresh=kwargs.get("box_score_thresh"),
                box_nms_thresh=kwargs.get("box_nms_thresh"),
                box_detections_per_img=kwargs.get("box_detections_per_img"),
            )
            return model
        else:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights=None, weights_backbone=None, num_classes=num_classes
            )
            if rpn_anchor_generator is not None and hasattr(model, "rpn"):
                model.rpn.anchor_generator = rpn_anchor_generator
            return model

    if name == "fasterrcnn_resnet101_fpn":
        backbone_weights = torchvision.models.ResNet101_Weights.DEFAULT if pretrained else None
        backbone = resnet_fpn_backbone("resnet101", weights=backbone_weights)
        model = FasterRCNN(backbone, num_classes=num_classes)
        if rpn_anchor_generator is not None and hasattr(model, "rpn"):
            model.rpn.anchor_generator = rpn_anchor_generator
        _maybe_set_rpn_topn(
            model,
            pre_nms_train=kwargs.get("rpn_pre_nms_top_n_train"),
            pre_nms_test=kwargs.get("rpn_pre_nms_top_n_test"),
            post_nms_train=kwargs.get("rpn_post_nms_top_n_train"),
            post_nms_test=kwargs.get("rpn_post_nms_top_n_test"),
        )
        _maybe_set_roi_params(
            model,
            box_score_thresh=kwargs.get("box_score_thresh"),
            box_nms_thresh=kwargs.get("box_nms_thresh"),
            box_detections_per_img=kwargs.get("box_detections_per_img"),
        )
        return model

    raise ValueError(f"Unknown model name: {name}")
