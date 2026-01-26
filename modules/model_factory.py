import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

def _replace_head(model, num_classes: int):
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def _maybe_set_rpn_topn(model, pre_nms_train=None, pre_nms_test=None, post_nms_train=None, post_nms_test=None):
    rpn = getattr(model, "rpn", None)
    if rpn is None:
        return

    def _set_topn(attr: str, train_val, test_val):
        if train_val is None and test_val is None:
            return
        current = getattr(rpn, attr, None)
        if isinstance(current, dict):
            if train_val is not None:
                current["training"] = int(train_val)
            if test_val is not None:
                current["testing"] = int(test_val)
            setattr(rpn, attr, current)
        else:
            # Fallback: some versions may store as int; set to testing value when provided.
            if test_val is not None:
                setattr(rpn, attr, int(test_val))

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
                weights=weights, weights_backbone=None, rpn_anchor_generator=rpn_anchor_generator
            )
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
            return torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
                weights=None, weights_backbone=None, num_classes=num_classes, rpn_anchor_generator=rpn_anchor_generator
            )

    if name == "fasterrcnn_resnet50_fpn":
        if pretrained:
            weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights=weights, weights_backbone=None, rpn_anchor_generator=rpn_anchor_generator
            )
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
            return torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights=None, weights_backbone=None, num_classes=num_classes, rpn_anchor_generator=rpn_anchor_generator
            )

    raise ValueError(f"Unknown model name: {name}")