import requests
from PIL import Image
import io
import torch 
import numpy as np
import torch.nn as nn
from torchvision.ops import box_iou
from scipy.optimize import linear_sum_assignment

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

def collect_detection_predictions(model, loader, device):
    """
    model can be either:
      - torch.nn.Module
      - str (URL of prediction endpoint)
    """
    all_preds = []

    for images, _ in loader:
        preds = predict(model, images, device)
        all_preds.extend(preds)

    return all_preds

def predict(model, images, device):
    """Run detection inference on a batch, dispatching by model kind.

    A string ``model`` is treated as an API URL; anything else is run locally
    on ``device``.

    Args:
        model: A torch detection model, or an API URL string.
        images (list[torch.Tensor]): Batch of image tensors.
        device (torch.device): Device the local model runs on.

    Returns:
        list[dict]: One prediction dict per image with CPU tensors for
            ``boxes``, ``labels``, and ``scores``.
    """
    if isinstance(model, str):
        return predict_api(model, images)
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
    print(type(model), type(images), device)
    model.eval(); model.to(device)

    images = [img.to(device) for img in images]

    with torch.inference_mode():
        outputs = model(images)

    return [{k: v.cpu() for k, v in o.items()} for o in outputs]

def predict_api(api_url, images):
    """Run detection inference on a batch via an HTTP API.

    Stacks the batch, serialises it to ``.npy``, POSTs it to the API URL, and
    reassembles the JSON response into per-image prediction tensors.

    The server is probed via ``/health`` to see whether it advertises uint8
    support. If it does, the batch is quantised to uint8 for a smaller payload
    (the server scales it back to ``[0, 1]`` float on-device); otherwise
    float32 is sent. The primary request uses a raw ``application/octet-stream``
    body (fast path); an HTTP 400 falls back to the legacy multipart float32
    upload for older servers.

    Args:
        api_url (str): API URL accepting a ``.npy`` batch and returning
            ``{"predictions": [...]}``.
        images (list[torch.Tensor]): Batch of image tensors (float32 in ``[0, 1]``).

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

    session = requests.Session()
    try:
        use_uint8 = False
        try:
            base = api_url.rsplit("/", 1)[0]
            h = session.get(f"{base}/health", timeout=5).json()
            use_uint8 = "uint8" in h.get("accepts_dtypes", [])
        except Exception:
            use_uint8 = False

        batch_np = batch
        if use_uint8:
            batch_np = (batch * 255.0).round().clip(0, 255).astype(np.uint8)

        buffer = io.BytesIO()
        np.save(buffer, batch_np)
        payload = buffer.getvalue()

        # Fast path: raw octet-stream body.
        response = session.post(
            api_url,
            data=payload,
            headers={"Content-Type": "application/octet-stream"},
            timeout=300,
        )

        # Compatibility fallback (multipart, float32) on HTTP 400 only.
        if response.status_code == 400:
            print(
                "Fast raw-body request returned HTTP 400; "
                "retrying with legacy multipart file upload."
            )
            fallback_buffer = io.BytesIO()
            np.save(fallback_buffer, batch)  # original float32
            fallback_buffer.seek(0)
            response = session.post(
                api_url,
                files={"file": ("batch.npy", fallback_buffer, "application/octet-stream")},
                timeout=300,
            )

        response.raise_for_status()
        outputs = response.json()["predictions"]
    finally:
        session.close()

    preds = []
    for pred in outputs:
        preds.append({
            "boxes": torch.tensor(pred["boxes"], dtype=torch.float32),
            "labels": torch.tensor(pred["labels"], dtype=torch.int64),
            "scores": torch.tensor(pred["scores"], dtype=torch.float32),
        })

    return preds

def image_brittleness(predA, predB, iou_thresh=0.5, alpha=0.5):
    """
    Returns the brittleness of images from the prediction boxes of before and after.

    Args:
        predA: predictions (boxes, labels, scores) of before corruption
        predB: predictions (boxes, labels, scores) of after corruption
        iou_thresh: iou threshold
        alpha: scale of how much to owe to detection matching vs iou score

    Returns:
        Array of brittleness scores, from 0 to 1 for each image.
    """
    boxesA, labelsA, scoresA = predA["boxes"], predA["labels"], predA["scores"]
    boxesB, labelsB, scoresB = predB["boxes"], predB["labels"], predB["scores"]

    if len(boxesA) == 0:
        return 0.0

    drops = []

    nA, nB = len(boxesA), len(boxesB)

    # One vectorized IoU pass, reused everywhere below: the cost matrix, the
    # unmatched-box "is the box still there?" check, and the matched-pair
    # localization drop all read from this single (nA, nB) tensor instead of
    # re-calling box_iou per pair.
    if nB > 0:
        iou_mat = box_iou(boxesA, boxesB)          # (nA, nB)
    else:
        iou_mat = torch.zeros((nA, 0))

    # Build cost matrix: only valid matches (same label, IoU >= thresh) get a
    # real cost; everything else stays at +inf (no match allowed).
    cost = torch.full((nA, nB), fill_value=float('inf'))
    if nB > 0:
        same_label = labelsA.view(nA, 1) == labelsB.view(1, nB)   # (nA, nB) bool
        valid = same_label & (iou_mat >= iou_thresh)
        # we want max IoU, so negate for the min-cost solver
        cost[valid] = -iou_mat[valid]

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
            best_iou = float(iou_mat[i].max().item()) if nB > 0 else 0.0
            if best_iou >= iou_thresh:
                # Box is there but label flipped — still a full drop, but flagged differently
                drop = scoreA  # label flip treated as full confidence loss
            else:
                drop = scoreA  # clean miss
        else:
            j = matched_A[i]
            scoreB = float(scoresB[j].item())
            iou = float(iou_mat[i, j].item())

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

def _count_correct_detections(pred, gt_objects, iou_thres=0.5, score_thres=0.5):
    """How many GT objects are correctly detected.

    A GT object counts as detected if there's a predicted box with a matching
    label, IoU >= iou_thres, and score >= score_thres. Matching is greedy
    (highest-IoU valid pairs first, one-to-one) rather than Hungarian —
    cheap and good enough for a filter predicate.

    Args:
        pred (dict): Prediction dict with "boxes", "labels", "scores" tensors.
        gt_objects (list[dict]): GT annotations, each with "bbox" and "label".
        iou_thres (float): Minimum IoU for a predicted box to match a GT box.
        score_thres (float): Minimum confidence for a predicted box to be
            considered at all.

    Returns:
        int: Number of GT objects matched by a correct, confident prediction.
    """
    boxes, labels, scores = pred["boxes"], pred["labels"], pred["scores"]

    keep = [i for i in range(len(boxes)) if float(scores[i]) >= score_thres]
    if not gt_objects or not keep:
        return 0

    gt_boxes = torch.tensor([o["bbox"] for o in gt_objects], dtype=torch.float32)
    gt_labels = [int(o["label"]) for o in gt_objects]

    pb = torch.as_tensor(boxes, dtype=torch.float32)[keep]
    iou = box_iou(pb, gt_boxes)  # (P, G)

    matched_gt = set()
    correct = 0
    # Greedy: for each kept prediction, take the highest-IoU unmatched GT
    # box that shares its label and clears the IoU threshold.
    for pi in range(iou.shape[0]):
        best_g, best_v = -1, iou_thres
        for gi in range(iou.shape[1]):
            if gi in matched_gt:
                continue
            if int(labels[keep[pi]]) != gt_labels[gi]:
                continue
            v = float(iou[pi, gi])
            if v >= best_v:
                best_g, best_v = gi, v
        if best_g >= 0:
            matched_gt.add(best_g)
            correct += 1
    return correct

def detected_well_before(result, iou_thres=0.5, score_thres=0.5, min_recall=0.75):
    """Was this image actually detected well before the corruption?

    Unlike the old count-based ``delta_detections_labels``, this checks
    correctness: a GT object only counts as found if a prediction matches it
    on label + IoU + score. ``min_recall`` is the fraction of GT objects that
    must be correctly detected for the image to qualify as "correct before".

    Args:
        result: A result with ``.label`` (GT objects) and ``.predA`` (dict of
            boxes/labels/scores before corruption).
        iou_thres (float): IoU threshold for a match to count.
        score_thres (float): Score threshold for a prediction to count.
        min_recall (float): Minimum fraction of GT objects that must be
            correctly detected, in [0, 1].

    Returns:
        bool: True if the model found at least ``min_recall`` of the GT
            objects before corruption.
    """
    n_gt = len(result.label)
    if n_gt == 0:
        return False

    correct = _count_correct_detections(result.predA, result.label, iou_thres, score_thres)
    return correct / n_gt >= min_recall

def lost_detections_after(result, iou_thres=0.5, score_thres=0.5, min_drop=0.5):
    """Did correct detections degrade after the corruption?

    Unlike the old count-based ``delta_detections``, this compares the number
    of *correct* detections before vs. after, so swapping boxes to the wrong
    class (while keeping the same box count) is correctly flagged as a drop.

    Args:
        result: A result with ``.label`` (GT objects), ``.predA`` (before),
            and ``.predB`` (after).
        iou_thres (float): IoU threshold for a match to count.
        score_thres (float): Score threshold for a prediction to count.
        min_drop (float): Minimum fractional drop in correct detections
            (before -> after) required to count as "wrong after", in [0, 1].

    Returns:
        bool: True if correct detections dropped by at least ``min_drop``.
            False if there were no correct detections before (nothing to lose).
    """
    before = _count_correct_detections(result.predA, result.label, iou_thres, score_thres)
    after = _count_correct_detections(result.predB, result.label, iou_thres, score_thres)
    if before == 0:
        return False

    return (before - after) / before >= min_drop