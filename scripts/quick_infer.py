import argparse
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from modules import get_model


def main():
    parser = argparse.ArgumentParser(description="Quick single-image inference")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--model", default="fasterrcnn_resnet50_fpn_v2")
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(args.model, num_classes=args.num_classes, pretrained=args.pretrained).eval().to(device)
    img = Image.open(args.image).convert("RGB")
    x = to_tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)[0]
    print("boxes:", out["boxes"].shape, "max score:", float(out["scores"].max().item()) if len(out["scores"]) else None)


if __name__ == "__main__":
    main()
