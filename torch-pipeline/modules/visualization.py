import os
import torch

def visualize_samples(model, dataset, indices, out_dir: str, device, score_thresh: float = 0.1):
    """
    Salva imagens com predições do modelo. Retorna um dict {idx: num_boxes_desenhadas}.
    """
    from PIL import ImageDraw
    from torchvision.transforms.functional import to_tensor, to_pil_image
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    drawn_counts = {}
    for idx in indices:
        try:
            img, _ = dataset[idx]
        except Exception:
            continue
        if isinstance(img, torch.Tensor):
            img_tensor = img.to(device)
        else:
            img_tensor = to_tensor(img).to(device)
        with torch.no_grad():
            pred = model([img_tensor])[0]
        pil_img = to_pil_image(img_tensor.cpu())
        draw = ImageDraw.Draw(pil_img)
        boxes = pred.get("boxes", [])
        scores = pred.get("scores", [])
        labels = pred.get("labels", [])

        kept = 0
        for b, s, lab in zip(boxes, scores, labels):
            s_val = float(s.item())
            if s_val < score_thresh:
                continue
            x1, y1, x2, y2 = b.tolist()
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1 + 2, y1 + 2), f"{lab}:{s_val:.2f}", fill="yellow")
            kept += 1

        out_path = os.path.join(out_dir, f"sample_{idx}.jpg")
        pil_img.save(out_path)  # sempre salva, mesmo sem caixas
        drawn_counts[idx] = kept
    return drawn_counts