import requests
from PIL import Image
import io
import torch 
import numpy as np
import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
import json
# from collections import defaultdict
from torchvision.ops import box_iou
import time

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
        # print("clasx names",class_names)
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

def get_prediction_from_image(model, display_image, device):
    """
    Predict the class of a single image, dispatching by model kind.

    A string ``model`` is treated as an API URL; otherwise the image is run
    through the local model on ``device``.

    Args:
        model: A torch model, or an API URL string for remote prediction.
        display_image (np.ndarray): CHW image array to classify.
        device (torch.device): Device the local model runs on.

    Returns:
        dict: prediction of the image represented by the bounding boxes of detection, 
        labels for classes of the boxes, and the scores of each detection
    """
    if isinstance(model, str):
        return get_prediction_from_image_api(model, display_image)
    image = torch.tensor(display_image).unsqueeze(0).float()
    image = image.to(device)

    model.eval(); model.to(device)
    with torch.no_grad():
        outputs = model(image)
    pred = outputs[0]

    prediction = {
        "boxes": pred["boxes"].cpu().numpy().tolist(),
        "labels": pred["labels"].cpu().numpy().tolist(),
        "scores": pred["scores"].cpu().numpy().tolist(),
    }
    return prediction

def get_prediction_from_image_api(model, display_image):
    """
    Predict the class of a single image via an HTTP API.

    Mirrors the wire-format handling of the batch detection path:

    * The server is probed via ``/health`` to see whether it advertises uint8
      support. If it does, the image is quantised to uint8 for a smaller
      payload (the server scales it back to ``[0, 1]`` float on-device);
      otherwise float32 is sent.
    * The primary request sends the raw ``.npy`` bytes as
      ``application/octet-stream`` (the fast path served by ``app_uint8.py``).
    * If an older server (``app.py``) rejects that with HTTP 400, the request
      is retried with the legacy multipart ``files={"file": ...}`` upload using
      the original float32 array.

    Args:
        model (str): API URL that accepts a ``.npy`` image and returns a prediction.
        display_image (np.ndarray): CHW image array (float32 in ``[0, 1]``) to classify.

    Returns:
        dict: prediction of the image represented by the bounding boxes of detection,
        labels for classes of the boxes, and the scores of each detection

    Raises:
        requests.HTTPError:
            If the API request returns an error status after any applicable
            fallback has been attempted.
    """
    API_URL = model

    session = requests.Session()
    try:
        # ------------------------------------------------------------------
        # Negotiate wire format once. Newer servers (app_uint8.py) advertise
        # uint8 support via /health; older servers (app.py) may lack /health
        # or not advertise it, so float32 remains the safe default.
        # ------------------------------------------------------------------
        use_uint8 = False
        try:
            base = API_URL.rsplit("/", 1)[0]
            h = session.get(f"{base}/health", timeout=5).json()
            use_uint8 = "uint8" in h.get("accepts_dtypes", [])
        except Exception:
            use_uint8 = False

        # display_image: (C, H, W) float32 in [0, 1] -> NCHW batch of 1
        batch = np.expand_dims(np.asarray(display_image, dtype=np.float32), axis=0)

        if use_uint8:
            batch = (batch * 255.0).round().clip(0, 255).astype(np.uint8)

        buffer = io.BytesIO()
        np.save(buffer, batch)
        payload = buffer.getvalue()

        # --------------------------------------------------------------
        # Fast path: raw .npy bytes as application/octet-stream.
        # --------------------------------------------------------------
        response = session.post(
            API_URL,
            data=payload,
            headers={"Content-Type": "application/octet-stream"},
            timeout=300,
        )

        # --------------------------------------------------------------
        # Compatibility fallback: older servers expect a multipart upload
        # with a field named "file". Only retry on HTTP 400. Re-send as
        # float32, which old servers always understand.
        # --------------------------------------------------------------
        if response.status_code == 400:
            print(
                "Fast raw-body request returned HTTP 400; "
                "retrying with legacy multipart file upload."
            )

            fallback_buffer = io.BytesIO()
            np.save(fallback_buffer, np.expand_dims(np.asarray(display_image, dtype=np.float32), axis=0))
            fallback_buffer.seek(0)

            response = session.post(
                API_URL,
                files={"file": ("array.npy", fallback_buffer, "application/octet-stream")},
                timeout=300,
            )

        response.raise_for_status()
        result = response.json()
    finally:
        session.close()

    # Return the prediction for the single image
    prediction = result["predictions"][0]

    return prediction

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
        AssertionError: If the number of tokens in the input is not a multiple of three.
    """
    items = s.split()
    assert len(items) % 3 == 0, "Input length must be a multiple of 3"
    return [items[i:i+3] for i in range(0, len(items), 3)]

def normalize_per_class(per_class):
    """
    Make all keys strings
    """
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
    Infer the number of output classes from a PyTorch classification model.

    The function attempts to determine the number of classes by inspecting the
    final Linear or Conv2d layer, or common classifier attributes such as
    'fc', 'classifier', 'head', or 'heads'.

    Args:
        model (nn.Module): A PyTorch model assumed to be used for classification.

    Returns:
        int: The inferred number of output classes.

    Raises:
        RuntimeError: If the number of classes cannot be determined from the model.
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
        if isinstance(model, str): #API
            raise ValueError("class_names must be specified if calling model as API")
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
    """
    Take the average for all of the performance metrics over multiple epochs
    """
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
    """
    Lazy image dataset for object detection.

    Loads each image on demand, upscaling it (with its boxes) so the shorter side
    is at least ``min_size`` while preserving aspect ratio, then applies the
    optional transform. Each item is ``(image, {"boxes", "labels"})``, where
    ``boxes`` are ``[x1, y1, x2, y2]`` in pixel coordinates and ``labels`` are
    integer class ids; images with no annotations yield empty box/label tensors.

    Attributes:
        image_paths (List[str]): Image file paths, one per sample.
        targets (List[List[dict]]): Per-image ``{"bbox", "label"}`` annotations.
        transform (callable | None): Optional image transform (e.g. ``ToTensor``),
            applied after resizing.
        min_size (int): Minimum shorter-side length; smaller images are upscaled.
    """
    def __init__(self, image_paths, targets, transform=None, min_size=500):
        self.image_paths = image_paths
        self.targets = targets
        self.transform = transform
        self.min_size = min_size

    def __len__(self):
        return len(self.image_paths)

    def _resize_up(self, image, boxes):
        """Upscale image so the smaller side == min_size, keep aspect ratio.
        Do nothing if the smaller side is already >= min_size."""
        W, H = image.size  # PIL: (width, height)
        shorter_side = min(W, H)

        if shorter_side >= self.min_size:
            return image, boxes  # already big enough, leave as-is

        scale = self.min_size / shorter_side
        new_W = round(W * scale)
        new_H = round(H * scale)

        image = image.resize((new_W, new_H), Image.BILINEAR)

        if boxes.numel() > 0:
            boxes = boxes * scale  # scales x1,y1,x2,y2 uniformly

        return image, boxes

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

        # resize + adjust boxes BEFORE ToTensor (while image is still PIL, in pixel coords)
        image, boxes = self._resize_up(image, boxes)

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
    ordered_ground_truth,
    class_names,
    output_json,
):
    """Write a COCO-format ground-truth json from already-resolved per-image GT.

    Reads each image's dimensions from disk, assigns 1-based image ids in list
    order, converts every ``[x_min, y_min, x_max, y_max]`` box to COCO's
    ``[x, y, w, h]`` form, and emits the ``images``/``annotations``/``categories``
    structure as json.

    Args:
        image_paths (list[str | Path]): Image paths, aligned by position with
            ``ordered_ground_truth`` (``image_paths[i]`` <-> ``ordered_ground_truth[i]``).
        ordered_ground_truth (list[list[dict]]): One entry per image, each a list
            of ``{"bbox": [x_min, y_min, x_max, y_max], "label": class_id}`` dicts
            (the output of ``_resolve_class_ids``).
        class_names (dict): ``{class_id_str: name}`` mapping, e.g.
            ``{"0": "cat", "1": "dog"}``.
        output_json (str): Path to write the COCO json to.

    Raises:
        ValueError: If ``image_paths`` and ``ordered_ground_truth`` differ in length.
    """
    if len(image_paths) != len(ordered_ground_truth):
        raise ValueError(
            f"image_paths ({len(image_paths)}) and ordered_ground_truth "
            f"({len(ordered_ground_truth)}) must be the same length and "
            f"aligned by position."
        )

    images = []
    annotations = []
    ann_id = 1

    for image_id, (img_path, anns) in enumerate(zip(image_paths, ordered_ground_truth), start=1):
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

        for ann in anns:
            x_min, y_min, x_max, y_max = ann["bbox"]
            w = x_max - x_min
            h = y_max - y_min

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": int(ann["label"]),
                "bbox": [x_min, y_min, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            ann_id += 1

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
    """Evaluate a detector and write its COCO predictions json in one pass.

    Runs inference exactly once per batch and feeds each batch's outputs into
    both the TP/FP/FN/confusion-matrix/mAP bookkeeping and the COCO predictions
    json, rather than sweeping the loader twice. Predictions are matched to
    ground truth greedily by IoU for the per-class stats, while the mAP is
    accumulated separately via TorchMetrics.

    Args:
        model: Detection model, or an API URL string (routed through ``predict``).
        loader (torch.utils.data.DataLoader): Loader yielding ``(images, targets)``.
        device (torch.device): Device local inference runs on.
        class_names (dict): ``{class_id_str: name}`` mapping.
        image_paths (list): Image paths, in loader (dataset) order; used to map
            each prediction back to its 1-based COCO ``image_id``.
        output_json (str): Path to write the COCO predictions json to.
        iou_thresh (float, optional): IoU threshold for greedy matching and mAP.
            Defaults to ``0.5``.
        score_thresh (float, optional): Minimum score for a box to enter the
            per-class TP/FP/FN stats. Defaults to ``0.5``.
        coco_score_threshold (float, optional): Minimum score for a box to be
            written into the COCO predictions json. Kept separate from
            ``score_thresh`` so the stats and json cutoffs can differ. Defaults
            to ``0.0``.

    Returns:
        dict: ``{"per_class": dict, "matrix": np.ndarray}`` — the per-class
            ``{TP, FP, FN, support}`` stats and the ``(num_classes, num_classes)``
            detection-matching matrix. The overall and per-class ``map`` are
            filled in by the caller from the COCO accumulator (see
            ``_run_severity_epochs_coco``), so they are not computed here.
    """
    num_classes = len(class_names)
    matrix = np.zeros((num_classes, num_classes), dtype=np.float32)

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

    # ------------------------------------------------------------------
    # If model is an API URL, create one Session and negotiate the wire
    # format (uint8 vs float32) once, before the loop, rather than doing
    # both on every batch. Both are then passed into predict()/predict_api()
    # for every call instead of being re-derived each time.
    # ------------------------------------------------------------------
    session = None
    use_uint8 = False
    if isinstance(model, str):
        session = requests.Session()
        try:
            base = model.rsplit("/", 1)[0]
            h = session.get(f"{base}/health", timeout=5).json()
            use_uint8 = "uint8" in h.get("accepts_dtypes", [])
        except Exception:
            use_uint8 = False
        print(f"wire format: {'uint8' if use_uint8 else 'float32'}")

    try:
        for images, targets in loader:
            targets = [{k: v.cpu() for k, v in t.items()} for t in targets]

            preds = predict(model, images, device, session=session, use_uint8=use_uint8)

            for pred, gt in zip(preds, targets):
                pred_boxes, pred_labels, pred_scores = _filter_and_sort_preds(
                    pred, score_thresh
                )

                _match_predictions_to_gt(
                    pred_boxes,
                    pred_labels,
                    gt["boxes"],
                    gt["labels"],
                    iou_thresh,
                    class_names,
                    per_class,
                    matrix,
                )

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
    finally:
        if session is not None:
            session.close()

    # dataset_idx advances once per prediction in loader order; it must have
    # walked exactly every image. If it hasn't, the loader reordered/dropped
    # samples (e.g. shuffle=True) and every image_id mapping above is wrong.
    if dataset_idx != len(image_paths):
        raise RuntimeError(
            f"prediction/image alignment broke: consumed {dataset_idx} images "
            f"but expected {len(image_paths)}. The loader must be unshuffled and "
            f"order-preserving for the COCO image_id mapping to be valid."
        )

    with open(output_json, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Saved predictions to {output_json}")

    # Note: overall/per-class mAP is NOT computed here anymore. It is derived
    # from the COCO accumulator at the single configured iou_thres by the
    # caller, so there is one consistent AP semantic across the whole plugin.
    return {
        "per_class": per_class,
        "matrix": matrix,
    }

def predict(model, images, device, session=None, use_uint8=False):
    """Run detection inference on a batch, dispatching by model kind.

    A string ``model`` is treated as an API URL; anything else is run locally
    on ``device``.

    Args:
        model: A torch detection model, or an API URL string.
        images (list[torch.Tensor]): Batch of image tensors.
        device (torch.device): Device the local model runs on.
        session (requests.Session | None): Persistent session to reuse for
            the API path, negotiated once by the caller rather than created
            per call. Ignored for the direct-model path.
        use_uint8 (bool): Wire format decided once by the caller via
            ``/health``, rather than re-queried on every call. Ignored for
            the direct-model path.

    Returns:
        list[dict]: One prediction dict per image with CPU tensors for
            ``boxes``, ``labels``, and ``scores``.
    """
    if isinstance(model, str):
        return predict_api(model, images, session=session, use_uint8=use_uint8)
    return predict_direct(model, images, device)

def predict_direct(model, images, device):
    """Run a local detection model over a batch and return CPU predictions.

    Args:
        model (torch.nn.Module): Detection model returning per-image dicts with
            ``boxes``, ``labels``, and ``scores``.
        images (list[torch.Tensor]): Batch of image tensors.
        device (torch.device): Device to run inference on.

    Returns:
        list[dict]: One prediction dict per image, with all tensors moved to CPU.
    """
    model.eval(); model.to(device)

    images = [img.to(device) for img in images]

    with torch.inference_mode():
        outputs = model(images)

    return [{k: v.cpu() for k, v in o.items()} for o in outputs]

def predict_api(api_url, images, session=None, use_uint8=False):
    """Run detection inference on a batch via an HTTP API.

    Stacks the batch, serialises it to ``.npy``, POSTs it to the API URL, and
    reassembles the JSON response into per-image prediction tensors.

    Wire format (``use_uint8``) and session are decided once by the caller
    (typically via one ``/health`` probe before the evaluation loop begins),
    not re-negotiated on every call — /health round-trips and fresh
    TCP/TLS handshakes on every batch were the main cost here previously.

    The primary request uses a raw ``application/octet-stream`` body (fast
    path, served by ``app_uint8.py``); an HTTP 400 falls back to the legacy
    multipart float32 upload for older servers (``app.py``).

    Args:
        api_url (str): API URL accepting a ``.npy`` batch and returning
            ``{"predictions": [...]}``.
        images (list[torch.Tensor]): Batch of image tensors (float32 in ``[0, 1]``).
        session (requests.Session | None): Persistent session to reuse across
            calls. If None, a bare ``requests.post`` is used (a fresh
            connection per call — fine for one-off use, not for a loop).
        use_uint8 (bool): Whether to quantise to uint8 before sending, as
            decided once by the caller via ``/health``. Only send True when
            the target server is known to support it (``app_uint8.py``).

    Returns:
        list[dict]: One prediction dict per image with ``boxes``, ``labels``,
            and ``scores`` tensors.

    Raises:
        requests.HTTPError:
            If the API request returns an error status after any applicable
            fallback has been attempted.
    """
    # images is List[Tensor], float32 in [0, 1]
    batch = torch.stack(images).cpu().numpy()  # (N, C, H, W) float32

    batch_np = batch
    if use_uint8:
        batch_np = (batch * 255.0).round().clip(0, 255).astype(np.uint8)

    buffer = io.BytesIO()
    np.save(buffer, batch_np)
    payload = buffer.getvalue()

    post = session.post if session is not None else requests.post
    t0 = time.perf_counter()
    # Fast path: raw octet-stream body.
    response = post(
        api_url,
        data=payload,
        headers={"Content-Type": "application/octet-stream"},
        timeout=300,
    )

    # Compatibility fallback (multipart, float32) on HTTP 400 only. Always
    # re-serialises from the original float32 batch regardless of
    # use_uint8, so the fallback is correct against old servers that only
    # understand float32.
    if response.status_code == 400:
        print(
            "Fast raw-body request returned HTTP 400; "
            "retrying with legacy multipart file upload."
        )
        fallback_buffer = io.BytesIO()
        np.save(fallback_buffer, batch)  # original float32
        fallback_buffer.seek(0)
        response = post(
            api_url,
            files={"file": ("batch.npy", fallback_buffer, "application/octet-stream")},
            timeout=300,
        )

    t1 = time.perf_counter()
    response.raise_for_status()
    outputs = response.json()["predictions"]
    t2 = time.perf_counter()

    print("~~ HTTP round-trip:", t1 - t0)
    print("~~~ JSON decode:", t2 - t1)
    return [
        {
            "boxes":  torch.tensor(pred["boxes"],  dtype=torch.float32),
            "labels": torch.tensor(pred["labels"], dtype=torch.int64),
            "scores": torch.tensor(pred["scores"], dtype=torch.float32),
        }
        for pred in outputs
    ]

def average_summaries(all_summaries):
    '''Average a list of ``collectSummaryResults()`` dicts across epochs.

    Produces a single dict of the same structure, with every leaf value
    (including the nested ``best_fbeta`` fields) averaged over epochs using
    ``np.nanmean``. ``average_method`` is carried over unchanged since it is the
    same for all epochs.

    Args:
        all_summaries (list[dict]): One dict per epoch, each as returned by
            ``collectSummaryResults()``.

    Returns:
        dict: Same structure as a single summary, with leaf values averaged.

    Raises:
        ValueError: If ``all_summaries`` is empty.
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

def plotMultipleFBetaCurves(curveDfs, curve_name, legend_names, filename):
    '''Overlay one named F-beta curve from several DataFrames onto one plot.

    Pulls the ``curve_name`` row (e.g. ``'precision'``, ``'recall'``, ``'F1'``)
    from each DataFrame and plots them together for comparison, then saves the
    figure to ``filename``.

    Args:
        curveDfs (list[pd.DataFrame]): DataFrames, each as returned by
            ``plotFBetaCurve()``.
        curve_name (str): Which curve (index row) to pull from each DataFrame.
        legend_names (list[str]): Legend label per DataFrame; same length as
            ``curveDfs``.
        filename (str | Path): Output image path.

    Raises:
        ValueError: If ``curveDfs`` and ``legend_names`` differ in length, or if
            ``curve_name`` is absent from any DataFrame.
    '''
    if len(curveDfs) != len(legend_names):
        raise ValueError(f'curveDfs (len={len(curveDfs)}) and legend_names (len={len(legend_names)}) must be the same length')

    fig, ax = plt.subplots(figsize=(12, 9))
    for curveDf, legend_name in zip(curveDfs, legend_names):
        if curve_name not in curveDf.index:
            raise ValueError(f"curve_name '{curve_name}' not found in DataFrame for legend '{legend_name}' (available: {list(curveDf.index)})")
        x = curveDf.loc[curve_name, 'x']
        y = curveDf.loc[curve_name, 'y']
        ax.plot(x, y, label=legend_name)

    ax.set_title(f'{curve_name} comparison')
    ax.set_xlabel('confidence threshold')
    ax.set_ylabel('score')
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.01)
    ax.grid(True)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def plotMultiplePRCurves(curveDfs, curve_name, legend_names, filename):
    '''Overlay one named IoU P-R curve from several DataFrames onto one plot.

    Pulls the ``curve_name`` row (e.g. ``'iou=0.50'``) from each DataFrame and
    plots the precision-vs-recall curves together for comparison, then saves the
    figure to ``filename``.

    Args:
        curveDfs (list[pd.DataFrame]): DataFrames, each as returned by
            ``plotPRCurve()``.
        curve_name (str): Which curve (index row) to pull from each DataFrame.
        legend_names (list[str]): Legend label per DataFrame; same length as
            ``curveDfs``.
        filename (str | Path): Output image path.

    Raises:
        ValueError: If ``curveDfs`` and ``legend_names`` differ in length, or if
            ``curve_name`` is absent from any DataFrame.
    '''
    if len(curveDfs) != len(legend_names):
        raise ValueError(f'curveDfs (len={len(curveDfs)}) and legend_names (len={len(legend_names)}) must be the same length')

    fig, ax = plt.subplots(figsize=(12, 9))
    for curveDf, legend_name in zip(curveDfs, legend_names):
        if curve_name not in curveDf.index:
            raise ValueError(f"curve_name '{curve_name}' not found in DataFrame for legend '{legend_name}' (available: {list(curveDf.index)})")
        x = curveDf.loc[curve_name, 'x']
        y = curveDf.loc[curve_name, 'y']
        ax.plot(x, y, label=legend_name)

    ax.set_title(f'P-R curve comparison ({curve_name})')
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.01)
    ax.grid(True)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)