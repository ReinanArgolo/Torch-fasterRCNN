# scripts/quick_infer.py
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from modules import get_model

model = get_model("fasterrcnn_resnet50_fpn_v2", num_classes=2, pretrained=True).eval().to("cuda" if torch.cuda.is_available() else "cpu")
img = Image.open("/home/reinanlinux/Documentos/TrainR-cnn/new_whales_rcnn/images/Validation/val/ALGUMA.jpg").convert("RGB")
x = to_tensor(img).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
with torch.no_grad():
    out = model(x)[0]
print("boxes:", out["boxes"].shape, "max score:", float(out["scores"].max().item()) if len(out["scores"]) else None)