import requests
from PIL import Image
from io import BytesIO
import torch 
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import json
from collections import defaultdict
from torchvision.ops import box_iou

def _filter_and_sort_preds(pred: dict, score_thresh: float):
    """Filter predictions by confidence score and sort by descending score.
 
    Args:
        pred:         Single prediction dict with keys 'boxes', 'labels', 'scores'.
        score_thresh: Minimum score to keep.
 
    Returns:
        (pred_boxes, pred_labels, pred_scores) — all filtered and sorted.
    """
    boxes  = pred["boxes"]
    labels = pred["labels"]
    scores = pred["scores"]
 
    keep   = scores >= score_thresh
    boxes  = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]
 
    order  = torch.argsort(scores, descending=True)
    return boxes[order], labels[order], scores[order]

def _match_predictions_to_gt(
    pred_boxes,
    pred_labels,
    gt_boxes,
    gt_labels,
    iou_thresh: float,
    class_names: dict,
    stats: dict,
    matrix,
):
    """Greedy IoU matching for one image; mutates stats and matrix in place.
 
    For each prediction (highest-confidence first) finds the best unmatched GT
    box. Records TP/FP in stats and the confusion matrix entry. After all
    predictions are processed, records FN for every unmatched GT box.
 
    Args:
        pred_boxes:  Tensor (P, 4) of predicted boxes.
        pred_labels: Tensor (P,) of predicted class ids.
        gt_boxes:    Tensor (G, 4) of ground-truth boxes.
        gt_labels:   Tensor (G,) of ground-truth class ids.
        iou_thresh:  IoU threshold for a match.
        class_names: {str(class_id): class_name} mapping.
        stats:       Per-class accumulator dict (mutated in place).
        matrix:      Confusion matrix ndarray (mutated in place).
    """
    matched_gt = set()
 
    if len(pred_boxes) > 0 and len(gt_boxes) > 0:
        iou_matrix = box_iou(pred_boxes, gt_boxes)  # (P, G)
    else:
        iou_matrix = torch.zeros(len(pred_boxes), len(gt_boxes))
 
    for p_idx, (plabel, iou_row) in enumerate(zip(pred_labels, iou_matrix)):
        class_name = class_names[str(plabel.item())]
 
        if matched_gt:
            iou_row = iou_row.clone()
            iou_row[list(matched_gt)] = -1.0
 
        if len(iou_row) > 0:
            best_iou, best_gt_idx = iou_row.max(dim=0)
            best_iou = best_iou.item()
            best_gt_idx = best_gt_idx.item()
        else:
            best_iou, best_gt_idx = -1.0, -1
 
        if best_iou >= iou_thresh:
            matched_gt.add(best_gt_idx)
            gt_class   = int(gt_labels[best_gt_idx])
            pred_class = int(plabel)
            matrix[gt_class, pred_class] += 1  # TP (diagonal) or class confusion
 
            gt_class_name = class_names[str(gt_class)]
            if gt_class == pred_class:
                stats[class_name]["TP"] += 1
            else:
                stats[class_name]["FP"] += 1
                stats[gt_class_name]["FN"] += 1
        else:
            # ghost prediction — predicted something, no GT box matched
            matrix[0, int(plabel)] += 1        # GT=background, PRED=class x
            stats[class_name]["FP"] += 1
 
    for gt_idx, glabel in enumerate(gt_labels):
        class_name = class_names[str(glabel.item())]
        stats[class_name]["support"] += 1
        if gt_idx not in matched_gt:
            # missed detection — GT existed, no prediction claimed it
            matrix[int(glabel), 0] += 1        # GT=class x, PRED=background
            stats[class_name]["FN"] += 1

def triplets(s):
    """
    Split a whitespace-separated string into groups of three items.

    Args:
        s (str): Input string containing whitespace-separated tokens. The number
            of tokens must be a multiple of three.

    Returns:
        List[List[str]]: A list of sublists, each containing three consecutive
        tokens from the input string.

    Raises:
        AssertionError: If the number of tokens in the input is not a multiple
        of three.
    """
    items = s.split()
    assert len(items) % 3 == 0, "Input length must be a multiple of 3"
    return [items[i:i+3] for i in range(0, len(items), 3)]

def normalize_per_class(per_class):
    out = {}

    for k, v in per_class.items():

        # already string label
        if isinstance(k, str) and not k.isdigit():
            out[k] = v
        else:
            out[str(k)] = v

    return out
    
def get_num_classes(model: nn.Module) -> int:
    """
    Infer number of classes from classification OR detection models.
    """

    # =========================
    # 1. Detection models (torchvision Faster R-CNN style)
    # =========================
    roi_heads = getattr(model, "roi_heads", None)
    if roi_heads is not None:
        box_predictor = getattr(roi_heads, "box_predictor", None)

        if box_predictor is not None:
            cls_score = getattr(box_predictor, "cls_score", None)

            if isinstance(cls_score, nn.Linear):
                return cls_score.out_features

    # =========================
    # 2. Classification head patterns
    # =========================
    for attr in ["classifier", "fc", "head", "heads"]:
        if hasattr(model, attr):
            module = getattr(model, attr)

            # single linear
            if isinstance(module, nn.Linear):
                return module.out_features

            # sequential head
            if isinstance(module, nn.Sequential):
                for layer in reversed(module):
                    if isinstance(layer, nn.Linear):
                        return layer.out_features

    # =========================
    # 3. Last resort (but safer than before)
    #    ONLY consider "final-ish" Linear layers
    # =========================
    linear_layers = [
        m for m in model.modules()
        if isinstance(m, nn.Linear)
    ]

    if linear_layers:
        # heuristic: smallest output dim is usually classifier in detectors
        # (bbox heads are usually larger or multiples of 4)
        best = min(linear_layers, key=lambda m: m.out_features)
        return best.out_features

    raise RuntimeError(f"Could not determine number of classes for {type(model)}")

def handle_class_names_arg(class_names_arg, model):
    """
    Parse class names input and return a mapping from class index to name.

    The function supports three input modes:
    1. None or empty string: infer number of classes from the model.
    2. Single integer: generate default class names using that count.
    3. Comma-separated names: use provided class names.

    Args:
        class_names_arg (Optional[str]): Class names specification. Can be:
            - None or empty string to infer from model
            - A single integer as string (e.g., "10")
            - Comma-separated class names (e.g., "cat,dog,bird")
        model (nn.Module): PyTorch model used when inferring class count.

    Returns:
        Dict[str, str]: Mapping from class index (as string) to class name.

    Raises:
        ValueError: If a single provided value is not a valid integer.
    """
    if class_names_arg is None or str(class_names_arg).strip() == "":
        
        print("# fallback: infer from model")
        num_classes = get_num_classes(model)
        class_names = {str(i): f"class_{i}" for i in range(num_classes)}

    else:
        class_names_arr = [x.strip() for x in class_names_arg.split(",") if x.strip()]

        # Case 1: user provided number of classes
        if len(class_names_arr) == 1:
            try:
                num_classes = int(class_names_arr[0])
                class_names = {str(i): f"class_{i}" for i in range(num_classes)}
            except ValueError:
                raise ValueError(
                    "class_names must be comma-separated names or a single integer"
                )

        # Case 2: user provided names
        else:
            class_names = {str(i): name for i, name in enumerate(class_names_arr)}

    return class_names

def average_detection_stats(all_stats):
    avg_stats = {}
    class_keys = all_stats[0].keys()

    for cls_name in class_keys:
        avg_stats[cls_name] = {}
        metric_keys = all_stats[0][cls_name].keys()

        for metric in metric_keys:
            values = [
                stats[cls_name][metric]
                for stats in all_stats
            ]
            avg_stats[cls_name][metric] = float(np.mean(values))

    return avg_stats

class DetectionDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, targets, transform=None):
        self.image_paths = image_paths
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        target = self.targets[idx]

        boxes = []
        labels = []

        for obj in target:
            boxes.append(obj["bbox"])
            labels.append(obj["label"])

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        labels = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long)

        target_dict = {
            "boxes": boxes,
            "labels": labels
        }

        if self.transform:
            image = self.transform(image)

        return image, target_dict

# ======= COCO STUFF =========

def create_coco_gt(
    image_paths,
    df,
    class_names,
    output_json,
):
    """
    Create a COCO-format ground-truth json.

    Parameters
    ----------
    image_paths : list[str] or list[Path]
        List of image paths (same images as used in the dataset).

    df : pandas dataframe containing:
        file_name
        x_min
        x_max
        y_min
        y_max
        class_id

    class_names : dict
        Example:
            {"0": "cat",
             "1": "dog"}

    output_json : str
        Output json filename.
    """
    images = []
    annotations = []
    ann_id = 1

    # map filename -> image_id
    image_id_map = {}

    for image_id, img_path in enumerate(image_paths, start=1):

        img_path = Path(img_path)
        filename = img_path.name

        with Image.open(img_path) as img:
            width, height = img.size

        images.append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height,
        })

        image_id_map[filename] = image_id

    # annotations
    for _, row in df.iterrows():

        filename = row["file_name"]

        if filename not in image_id_map:
            continue

        x_min = float(row["x_min"])
        x_max = float(row["x_max"])
        y_min = float(row["y_min"])
        y_max = float(row["y_max"])

        w = x_max - x_min
        h = y_max - y_min

        annotations.append({
            "id": ann_id,
            "image_id": image_id_map[filename],
            "category_id": int(row["class_id"]),
            "bbox": [x_min, y_min, w, h],
            "area": w * h,
            "iscrowd": 0,
        })

        ann_id += 1

    # categories
    categories = [
        {
            "id": int(cid),
            "name": name,
            "supercategory": "none",
        }
        for cid, name in sorted(class_names.items(), key=lambda x: int(x[0]))
    ]

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"Saved COCO annotations to {output_json}")

def evaluate_detection_and_create_coco_predictions(
    model,
    loader,
    device,
    class_names,
    image_paths,
    output_json,
    iou_thresh=0.5,
    score_thresh=0.5,
    coco_score_threshold=0.0,
):
    """
    Combined version of evaluate_detection_detailed + create_coco_predictions.
    Runs model inference exactly once per batch and feeds the outputs into
    both the TP/FP/FN/confusion-matrix/mAP bookkeeping AND the COCO predictions
    JSON, instead of running two full passes over the loader.

    Args mirror the two original functions:
        - iou_thresh / score_thresh: used for the detailed per-class stats path
          (same as evaluate_detection_detailed).
        - coco_score_threshold: minimum score for a box to be written into the
          COCO predictions JSON (same as create_coco_predictions's
          score_threshold). Kept separate since the two thresholds were
          allowed to differ before merging (score_thresh vs score_threshold).

    Returns same dict as evaluate_detection_detailed:
        {"map": float, "per_class": dict, "matrix": ndarray}
    Also writes output_json, same as create_coco_predictions did.
    """
    num_classes = len(class_names)
    matrix = np.zeros((num_classes, num_classes), dtype=np.float32)
    metric = MeanAveragePrecision(iou_thresholds=[iou_thresh], class_metrics=True)

    per_class = {
        class_name: {"TP": 0, "FP": 0, "FN": 0, "support": 0}
        for class_name in class_names.values()
    }#stats

    filename_to_image_id = {
        Path(p).name: idx + 1
        for idx, p in enumerate(image_paths)
    }
    predictions = []
    dataset_idx = 0

    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)  # <-- single forward pass, shared by both paths

            preds = [{k: v.cpu() for k, v in o.items()} for o in outputs]
            gts   = [{k: v.cpu() for k, v in t.items()} for t in targets]

            metric.update(preds, gts)

            for pred, gt in zip(preds, gts):
                pred_boxes, pred_labels, pred_scores = _filter_and_sort_preds(pred, score_thresh)
                _match_predictions_to_gt(
                    pred_boxes, pred_labels,
                    gt["boxes"], gt["labels"],
                    iou_thresh, class_names, per_class, matrix,
                )

            # --- COCO predictions, built from the same `preds` we just computed ---
            for pred in preds:
                filename = Path(image_paths[dataset_idx]).name
                image_id = filename_to_image_id[filename]

                boxes = pred["boxes"].numpy()
                labels = pred["labels"].numpy()
                scores = pred["scores"].numpy()

                for box, label, score in zip(boxes, labels, scores):
                    if score < coco_score_threshold:
                        continue
                    x1, y1, x2, y2 = box
                    predictions.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [
                            float(x1),
                            float(y1),
                            float(x2 - x1),
                            float(y2 - y1),
                        ],
                        "score": float(score),
                    })
                dataset_idx += 1

    with open(output_json, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Saved predictions to {output_json}")

    # per_class = stats #_compute_per_class_metrics(stats)
    map_result = metric.compute()

    classes = map_result["classes"]
    aps = map_result["map_per_class"]
    if classes.ndim == 0:
        classes = classes.unsqueeze(0)
        aps = aps.unsqueeze(0)

    per_class_ap = {
        class_names[str(cls_idx.item())]: float(ap)
        for cls_idx, ap in zip(classes, aps)
    }
    for class_name, metrics in per_class.items():
        metrics["map"] = per_class_ap.get(class_name, float("nan"))

    metric.reset()

    return {
        "map": map_result["map"].item(),
        "per_class": per_class,
        "matrix": matrix,
    }

def average_summaries(all_summaries):
    '''
    Average a list of summary dicts produced by collectSummaryResults()
    across multiple epochs into a single dict of the same structure.

    :param all_summaries: list of dicts, each from collectSummaryResults()
    :return: dict with same structure, leaf values averaged across epochs
    '''
    if not all_summaries:
        raise ValueError("all_summaries is empty")

    avg = {
        'coco_summary': {},
        'fbeta_summary': {},
        'best_fbeta': {},
        'average_method': all_summaries[0]['average_method'],  # same for all epochs
    }

    # flat dicts: just average each key across summaries
    for section in ('coco_summary', 'fbeta_summary'):
        for key in all_summaries[0][section]:
            values = [s[section][key] for s in all_summaries]
            avg[section][key] = float(np.nanmean(values))

    # best_fbeta is one level deeper: {key: {score, confThr, precision, recall}}
    for key in all_summaries[0]['best_fbeta']:
        avg['best_fbeta'][key] = {
            field: float(np.nanmean([s['best_fbeta'][key][field] for s in all_summaries]))
            for field in all_summaries[0]['best_fbeta'][key]
        }

    return avg