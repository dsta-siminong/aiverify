import requests
from PIL import Image
import io
import torch 
import torch.nn.functional as F 
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.metrics import precision_score, recall_score  , f1_score  , roc_auc_score
from sklearn.preprocessing import label_binarize
from torchvision.ops import box_iou

def get_prediction_from_image(model, display_image, device):
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
    API_URL = model

    # display_image: (C, H, W)
    batch = np.expand_dims(display_image.astype(np.float32), axis=0)

    buffer = io.BytesIO()
    np.save(buffer, batch)
    buffer.seek(0)

    response = requests.post(
        API_URL,
        files={"file": ("array.npy", buffer, "application/octet-stream")},
    )
    response.raise_for_status()

    result = response.json()

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
        AssertionError: If the number of tokens in the input is not a multiple
        of three.
    """
    items = s.split()
    assert len(items) % 3 == 0, "Input length must be a multiple of 3"
    return [items[i:i+3] for i in range(0, len(items), 3)]

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

def collect_detection_predictions(model, loader, device):
    all_preds = []

    for images, _ in loader:
        preds = predict(model, images, device)
        all_preds.extend(preds)

    return all_preds#torch.stack(all_imgs), 

def predict(model, images, device):
    if isinstance(model, str):
        return predict_api(model, images)
    return predict_direct(model, images, device)

def predict_direct(model, images, device):
    print(type(model), type(images), device)
    model.eval(); model.to(device)

    images = [img.to(device) for img in images]

    with torch.inference_mode():
        outputs = model(images)

    return [{k: v.cpu() for k, v in o.items()} for o in outputs]

def predict_api(api_url, images):
    # images is List[Tensor]
    batch = torch.stack(images).cpu().numpy()

    buffer = io.BytesIO()
    np.save(buffer, batch)
    buffer.seek(0)

    response = requests.post(
        api_url,
        files={"file": ("batch.npy", buffer, "application/octet-stream")},
    )
    response.raise_for_status()

    outputs = response.json()["predictions"]

    preds = []
    for pred in outputs:
        preds.append({
            "boxes": torch.tensor(pred["boxes"], dtype=torch.float32),
            "labels": torch.tensor(pred["labels"], dtype=torch.int64),
            "scores": torch.tensor(pred["scores"], dtype=torch.float32),
        })

    return preds

def image_brittleness(predA, predB, iou_thresh=0.5, alpha=0.5):
    boxesA, labelsA, scoresA = predA["boxes"], predA["labels"], predA["scores"]
    boxesB, labelsB, scoresB = predB["boxes"], predB["labels"], predB["scores"]

    if len(boxesA) == 0:
        return 0.0

    # Use Hungarian matching for stable global assignment
    from scipy.optimize import linear_sum_assignment

    usedB = set()
    drops = []

    # Build cost matrix: only valid matches (same label, IoU >= thresh) get a real cost
    nA, nB = len(boxesA), len(boxesB)
    cost = torch.full((nA, nB), fill_value=float('inf'))

    for i in range(nA):
        for j in range(nB):
            if int(labelsA[i]) != int(labelsB[j]):
                continue
            iou = box_iou(boxesA[i].unsqueeze(0), boxesB[j].unsqueeze(0))[0, 0].item()
            if iou >= iou_thresh:
                cost[i, j] = -iou  # we want max IoU, so negate for min-cost solver

    cost_np = cost.numpy()
    # Replace inf with a large finite number for the solver
    cost_np[cost_np == float('inf')] = 1e9
    row_ind, col_ind = linear_sum_assignment(cost_np)

    matched_B = {col_ind[k]: row_ind[k] for k in range(len(row_ind))
                 if cost[row_ind[k], col_ind[k]].item() < 1e9}
    matched_A = {v: k for k, v in matched_B.items()}

    for i in range(nA):
        scoreA = float(scoresA[i].item())

        if i not in matched_A:
            # Detection in A has no counterpart in B: full confidence lost
            # Check if it's a label flip (same box, wrong label) vs pure miss
            best_iou = 0.0
            for j in range(nB):
                iou = box_iou(boxesA[i].unsqueeze(0), boxesB[j].unsqueeze(0))[0, 0].item()
                best_iou = max(best_iou, iou)
            if best_iou >= iou_thresh:
                # Box is there but label flipped — still a full drop, but flagged differently
                drop = scoreA  # label flip treated as full confidence loss
            else:
                drop = scoreA  # clean miss
        else:
            j = matched_A[i]
            scoreB = float(scoresB[j].item())
            iou = box_iou(boxesA[i].unsqueeze(0), boxesB[j].unsqueeze(0))[0, 0].item()

            score_drop = max(0.0, scoreA - scoreB)                     # range in [0,1]
            loc_drop = 1.0 - iou                                       # already in [0,1]
            drop = alpha * score_drop + (1 - alpha) * loc_drop                   # equal weighting; tunable

        drops.append((scoreA, drop))

    if not drops:
        return 0.0

    # Confidence-weighted mean drop, normalized to [0,1]
    total_weight = sum(s for s, _ in drops)
    brittleness = sum(s * d for s, d in drops) / total_weight
    return brittleness  # guaranteed in [0,1]

class DetectionDataset(torch.utils.data.Dataset):
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

def delta_detections(result, N=0):
    num_A = len(result.predA["boxes"])
    num_B = len(result.predB["boxes"])

    decrease = num_A - num_B

    if N < 0:
        raise ValueError("N must be non-negative")

    # Fractional threshold
    if isinstance(N, float):
        if N > 1:
            raise ValueError("Fractional N must be between 0 and 1")

        if num_A == 0:
            return False

        return decrease / num_A >= N

    # Integer threshold
    return decrease > N

def delta_detections_labels(result, N=0):
    num_A = len(result.predA["boxes"])
    num_labels = len(result.label)

    decrease = num_labels - num_A

    if N < 0:
        raise ValueError("N must be non-negative")

    # Fractional threshold
    if isinstance(N, float):
        if N > 1:
            raise ValueError("Fractional N must be between 0 and 1")

        if num_A == 0:
            return False

        return decrease / num_A <= N

    # Integer threshold
    return decrease < N