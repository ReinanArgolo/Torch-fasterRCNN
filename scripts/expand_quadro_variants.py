from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Tuple


def _index_images(images_dir: str) -> Dict[str, str]:
    """Map relative path (posix, lower) -> relative path (posix, original)."""
    out: Dict[str, str] = {}
    for root, _dirs, files in os.walk(images_dir):
        for f in files:
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            full = os.path.join(root, f)
            if not os.path.isfile(full):
                continue
            rel = os.path.relpath(full, images_dir).replace(os.sep, "/")
            out[rel.lower()] = rel
    return out


def _find_variants(all_rel: List[str], base_rel: str) -> List[str]:
    """Find files like quadro_0000_*.jpg for a base file quadro_0000.jpg.

    Returns relative paths (posix) excluding the base_rel itself.
    """
    base_name = os.path.basename(base_rel)
    stem, ext = os.path.splitext(base_name)
    # Only expand when stem matches quadro_####
    m = re.match(r"^(quadro_\d{4})$", stem)
    if not m:
        return []
    prefix = m.group(1) + "_"

    candidates = []
    for rel in all_rel:
        bn = os.path.basename(rel)
        if not bn.lower().endswith(ext.lower()):
            continue
        if bn.startswith(prefix):
            candidates.append(rel)
    # Remove self if present
    candidates = [c for c in candidates if c.lower() != base_rel.lower()]
    return sorted(set(candidates))


def expand_variants(coco: dict, images_dir: str) -> Tuple[dict, dict]:
    img_index = _index_images(images_dir)
    all_rel = list(img_index.values())

    images = list(coco.get("images", []))
    annotations = list(coco.get("annotations", []))

    img_id_to_image = {int(im["id"]): im for im in images}
    img_id_to_anns: Dict[int, List[dict]] = {}
    for an in annotations:
        img_id_to_anns.setdefault(int(an["image_id"]), []).append(an)

    existing_file_names_lower = {str(im.get("file_name", "")).replace("\\", "/").lower() for im in images}

    next_img_id = max(img_id_to_image.keys()) + 1 if img_id_to_image else 1
    next_ann_id = max([int(a.get("id", 0)) for a in annotations], default=0) + 1

    created_images = 0
    created_anns = 0

    for im in list(images):
        file_name = str(im.get("file_name", "")).replace("\\", "/")
        # Only expand if this exact file exists or can be resolved by basename
        rel_key = file_name.lower()
        if rel_key not in img_index:
            # Try basename-only resolution
            base = os.path.basename(file_name).lower()
            matches = [rel for rel in all_rel if os.path.basename(rel).lower() == base]
            if len(matches) == 1:
                file_name = matches[0]
                rel_key = file_name.lower()
            else:
                continue

        variants = _find_variants(all_rel, img_index[rel_key])
        if not variants:
            continue

        src_img_id = int(im["id"])
        src_anns = img_id_to_anns.get(src_img_id, [])

        for v_rel in variants:
            if v_rel.lower() in existing_file_names_lower:
                continue

            new_im = dict(im)
            new_im["id"] = next_img_id
            new_im["file_name"] = v_rel
            images.append(new_im)
            existing_file_names_lower.add(v_rel.lower())
            created_images += 1

            for an in src_anns:
                new_an = dict(an)
                new_an["id"] = next_ann_id
                new_an["image_id"] = next_img_id
                annotations.append(new_an)
                next_ann_id += 1
                created_anns += 1

            next_img_id += 1

    out = dict(coco)
    out["images"] = images
    out["annotations"] = annotations

    summary = {
        "created_images": created_images,
        "created_annotations": created_anns,
        "images_total": len(images),
        "annotations_total": len(annotations),
    }
    return out, summary


def main():
    p = argparse.ArgumentParser(description="Expand COCO images to include quadro_####_*.jpg variants")
    p.add_argument("--ann", required=True, help="Input COCO annotation JSON")
    p.add_argument("--images-dir", required=True, help="Root directory containing images (will be walked recursively)")
    p.add_argument("--out", required=True, help="Output COCO annotation JSON")
    args = p.parse_args()

    with open(args.ann, "r") as f:
        coco = json.load(f)

    expanded, summary = expand_variants(coco, images_dir=args.images_dir)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(expanded, f)

    print(summary)


if __name__ == "__main__":
    main()
