import requests
from PIL import Image
import io
import torch 
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
# from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import time
import resource
def mem_mb():
    """
    Report the peak resident memory of this process.

    Reads the process's maximum RSS from ``resource.getrusage`` and converts it
    from kilobytes to megabytes.

    Returns:
        float: Peak resident set size in megabytes.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

def evaluate_detection(model, loader, device, iou_threshold=0.5):
    """
    Evaluate a model over a loader, dispatching by model kind.

    A string ``model`` is treated as an API URL and evaluated remotely; anything
    else is evaluated locally on ``device``.

    Args:
        model: A torch model, or an API URL string for remote evaluation.
        loader (torch.utils.data.DataLoader): Data loader to evaluate over.
        device (torch.device): Device the local model runs on.
        iou_threshold (float): threshold for intersection over union (iou) calcs

    Returns:
        Tuple[float, np.ndarray, np.ndarray]: Accuracy (percent), predicted
            labels, and true labels.
    """
    if isinstance(model, str):
        print("STRING MODEL!")
        print(model)
        return evaluate_detection_api(model, loader, iou_threshold)
    else:
        print("Direct model")
        print(type(model), type(model), device)
        return evaluate_detection_direct(model, loader, device, iou_threshold)

def evaluate_detection_direct(model, loader, device, iou_threshold=0.5):
    """Evaluate an object detection model using mean Average Precision (mAP).

    Computes the mAP score at a specified IoU threshold over all samples in
    the provided dataloader. Predictions and ground-truth annotations are
    accumulated using TorchMetrics' MeanAveragePrecision metric.

    Args:
        model (torch.nn.Module):
            Object detection model that accepts a list of images and returns
            a list of prediction dictionaries containing keys such as
            ``boxes``, ``scores``, and ``labels``.

        loader (torch.utils.data.DataLoader):
            DataLoader yielding batches of ``(images, targets)``, where:

            - ``images`` is a list of image tensors.
            - ``targets`` is a list of dictionaries containing ground-truth
              annotations (e.g., ``boxes`` and ``labels``).

        device (torch.device):
            Device on which inference is performed (e.g., CPU or CUDA device).

        iou_threshold (float, optional):
            Intersection over Union (IoU) threshold used for mAP computation.
            Defaults to ``0.5``.

    Returns:
        float:
            The mAP value at the specified IoU threshold.
    """
    metric = MeanAveragePrecision(iou_thresholds=[iou_threshold])

    model.eval(); model.to(device)
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            print(f"[mem before model()] {mem_mb():.1f} MB")
            outputs = model(images)
            print(f"[mem after model()] {mem_mb():.1f} MB")

            preds = [{k: v.cpu() for k, v in o.items()} for o in outputs]
            gts = [{k: v.cpu() for k, v in t.items()} for t in targets]

            metric.update(preds, gts)

    result = metric.compute()
    metric.reset()
    return result["map"].item() #generalize in future

import concurrent.futures

def evaluate_detection_api(model, loader, iou_threshold=0.5):
    """
    Evaluate an object detection model served behind an HTTP API.
 
    Each batch is serialised to ``.npy`` and POSTed to the API URL.
    Predictions from the JSON response are accumulated into a
    MeanAveragePrecision metric and returned as mAP.
 
    Wire format is negotiated via ``/health``: if the server advertises
    ``uint8`` support (app_uint8.py), the batch is quantised to uint8 for a
    ~4x smaller upload and the server scales it back to [0,1] float on-device
    before inference, so the detection model receives exactly the same [0,1]
    float input as with float32. Older servers that do not advertise uint8
    fall back to float32.

    Differences from the classification version:
      - Response payloads are larger and variable (boxes + scores + labels per
        detected object), so JSON decode time may be non-trivial on dense
        scenes — tracked separately.
 
    The next DataLoader batch is prefetched in a background thread so that
    data loading overlaps with the current batch's HTTP request.
 
    The primary request format is a raw ``application/octet-stream`` body
    (fast path). If the server rejects that with HTTP 400, the request is
    retried as a multipart upload with float32 payload for compatibility with
    older servers.
 
    Args:
        model (str): Base API URL, e.g. "http://model-server:8000/predict_fast".
        loader (torch.utils.data.DataLoader): Data loader yielding
            (List[Tensor], List[Dict]) pairs as torchvision detection loaders do.
        iou_threshold (float): IoU threshold for mAP computation.
 
    Returns:
        float: mean Average Precision (mAP) at the given IoU threshold.
 
    Raises:
        requests.HTTPError: If an API request returns an error status after
            any applicable fallback has been attempted.
    """
    API_URL = model
    metric = MeanAveragePrecision(iou_thresholds=[iou_threshold])
 
    session = requests.Session()
 
    # ------------------------------------------------------------------
    # Capability negotiation.
    #
    # Newer servers (app_uint8.py) advertise uint8 support via /health. uint8
    # is purely a smaller-payload wire format: the server scales it back to
    # [0,1] float on-device before inference, so the detection model receives
    # exactly the same [0,1] float input as with float32. Older servers may
    # lack /health or not advertise it, so float32 remains the safe default.
    # ------------------------------------------------------------------
    use_uint8 = False
    try:
        base = API_URL.rsplit("/", 1)[0]
        h = session.get(f"{base}/health", timeout=5).json()
        use_uint8 = "uint8" in h.get("accepts_dtypes", [])
    except Exception:
        use_uint8 = False

    print(f"wire format: {'uint8' if use_uint8 else 'float32'}")
 
    # ------------------------------------------------------------------
    # Prefetch the next batch in a background thread so data loading
    # overlaps with the HTTP request for the current batch.
    # ------------------------------------------------------------------
    it = iter(loader)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
 
    def fetch_next():
        try:
            return next(it)
        except StopIteration:
            return None
 
    next_future = executor.submit(fetch_next)
    loop_end = time.perf_counter()
 
    try:
        while True:
            batch = next_future.result()
            if batch is None:
                break
 
            images, targets = batch
            data_load_time = time.perf_counter() - loop_end
            print("data loading (dataset->batch):", data_load_time)
 
            # Kick off the next batch immediately so it overlaps with
            # serialization + HTTP below.
            next_future = executor.submit(fetch_next)
 
            # --------------------------------------------------------------
            # Serialize.
            # torchvision detection loaders yield List[Tensor]; stack to NCHW.
            # Quantise to uint8 when the server supports it (4x smaller upload,
            # scaled back to [0,1] on-device); otherwise send float32.
            # --------------------------------------------------------------
            batch_np = torch.stack(images).numpy()  # (N, C, H, W) float32

            if use_uint8:
                batch_np = (batch_np * 255.0).round().clip(0, 255).astype(np.uint8)

            buffer = io.BytesIO()
            np.save(buffer, batch_np)
            payload = buffer.getvalue()
 
            # --------------------------------------------------------------
            # Fast path: raw octet-stream body.
            # --------------------------------------------------------------
            t0 = time.perf_counter()
 
            response = session.post(
                API_URL,
                data=payload,
                headers={"Content-Type": "application/octet-stream"},
                timeout=300,
            )
 
            # --------------------------------------------------------------
            # Compatibility fallback for older servers (multipart, float32).
            # Only triggered on HTTP 400 (parse/format rejection), not on
            # 500 (inference error) or other status codes.
            # --------------------------------------------------------------
            if response.status_code == 400:
                print(
                    "Fast raw-body request returned HTTP 400; "
                    "retrying with legacy multipart upload."
                )
                print("========================")
                print(response.status_code, response.reason)
                print(response.text)
                print("========================")
 
                # Always re-send as float32 in the fallback regardless of
                # what was negotiated — old servers only understand float32.
                fallback_np = torch.stack(images).numpy()
                fallback_buffer = io.BytesIO()
                np.save(fallback_buffer, fallback_np)
 
                response = session.post(
                    API_URL,
                    files={
                        "file": (
                            "batch.npy",
                            fallback_buffer,
                            "application/octet-stream",
                        )
                    },
                    timeout=300,
                )
 
            t1 = time.perf_counter()
            response.raise_for_status()
 
            # --------------------------------------------------------------
            # Decode response. Detection responses can be large (many boxes),
            # so JSON decode time is tracked separately.
            # --------------------------------------------------------------
            preds_json = response.json()["predictions"]
            t2 = time.perf_counter()
 
            print("- HTTP round-trip:", t1 - t0)
            print("-- JSON decode:", t2 - t1)
 
            # --------------------------------------------------------------
            # Convert to torchmetrics format.
            # --------------------------------------------------------------
            preds = [
                {
                    "boxes":  torch.tensor(p["boxes"],  dtype=torch.float32),
                    "scores": torch.tensor(p["scores"], dtype=torch.float32),
                    "labels": torch.tensor(p["labels"], dtype=torch.int64),
                }
                for p in preds_json
            ]
 
            gts = [{k: v for k, v in t.items()} for t in targets]
 
            metric.update(preds, gts)
            loop_end = time.perf_counter()
 
    finally:
        executor.shutdown()
        session.close()
 
    result = metric.compute()
    metric.reset()
    return result["map"].item() #generalize in future

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

    Mirrors the wire-format handling of :func:`evaluate_detection_api`:

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
        AssertionError: If the number of tokens in the input is not a multiple
        of three.
    """
    items = s.split()
    assert len(items) % 3 == 0, "Input length must be a multiple of 3"
    return [items[i:i+3] for i in range(0, len(items), 3)]

def augmentation_gradient_det(
    model, 
    test_loader, 
    device, 
    aug_class, 
    plot_graphs=False, 
    directory=Path(), 
    num_epochs=1,
    iou_threshold=0.5,
):
    """
    Measure how detection mAP degrades as augmentation severity increases.

    Evaluates the clean loader (severity "None") and each of the augmentation's
    severities, averaging over ``num_epochs`` reseeded runs per severity for
    non-deterministic augmentations (forced to 1 epoch when deterministic), then
    fits a line through the mAP-vs-severity points to summarize the trend.

    Args:
        model: Detection model, or an API URL string (routed to the API path).
        test_loader (DataLoader): Clean loader yielding ``(images, targets)``.
        device (torch.device): Device inference runs on.
        aug_class: The ``Augmentation`` instance (severities, seeding, corruption).
        plot_graphs (str | bool, optional): Graphing library (e.g. ``'matplotlib'``)
            to render the figure with, or ``False`` to skip plotting. Defaults to
            ``False``.
        directory (Path, optional): Directory to save the figure into. Defaults to
            the current directory.
        num_epochs (int, optional): Reseeded repeats per severity for
            non-deterministic augmentations. ``None`` is treated as 1. Defaults to 1.
        iou_threshold (float, optional): IoU threshold for mAP. Defaults to ``0.5``.

    Returns:
        Tuple[float, List[float], Path]: The best-fit gradient of mAP vs severity,
            the mAP per severity (clean first), and the saved figure path.
    """
    num_epochs = 1 if num_epochs is None else num_epochs
    num_epochs = 1 if aug_class.deterministic else num_epochs
    print("===")
    print("Aug name", aug_class.name)
    print(f"Evaluating on severity 0/None...")
    base_map = evaluate_detection(model, test_loader, device, iou_threshold)
    print(f"mAP at severity 0/None: {base_map:.4f}")
    severities = aug_class.severities #[x for x in range(len(aug_class.severities))]
    maps = [base_map]
    for severity_idx, severity in enumerate(severities):
        print(f"Evaluating on severity {severity}...")
        all_map = []
        for i in range(num_epochs):
            seed = 1000*i + severity_idx 
            aug_class.set_seed(seed)
            corrupted_loader = aug_class.corr_func_dataloader(test_loader, severity_idx=severity)
            corr_map = evaluate_detection(model, corrupted_loader, device, iou_threshold)
            all_map.append(corr_map)
            print(f"epoch {i+1}: {corr_map}")
            del corrupted_loader

        final_map = sum(all_map)/len(all_map)
        maps.append(final_map)
        print(f"mAP at severity {severity}: {final_map:.4f}")

    # Plot results
    fig_path = directory / f"accuracy_vs_severity_{aug_class.name}.png"
    fig = None
    if plot_graphs is not False:
        fig = plot_accuracy_vs_severity(maps, ["None"]+severities, plot_graphs)  
        fig.savefig(fig_path)
        plt.close()
    return best_fit_gradient(list(range(len(severities)+1)), maps), maps, fig_path

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

# ==== HELPER FUNCTIONS FOR AUGMENTATION GRADIENT ====

def plot_accuracy_vs_severity(accuracies, severities=None, graph_lib='matplotlib'):
    """Plots the accuracy/performance of model changes against severities (of data augmentation)

    Args:
        accuracies (list): list of accuracies or performances
        severities (list, optional): list of integers representing severities. Defaults to None.
        graph_lib (str, optional): graphing library in python. Defaults to 'matplotlib'.

    Raises:
        ValueError: For invalid graphing library given

    Returns:
        figure: resultant graph
    """
    if graph_lib == 'matplotlib':
        return plot_accuracy_vs_severity_mpl(accuracies, severities)
    elif graph_lib == 'plotly':
        return plot_accuracy_vs_severity_plotly(accuracies, severities)
    else:
        raise ValueError('not valid graphing library')

def plot_accuracy_vs_severity_mpl(accuracies, severities=None):
    """Plots the accuracy/performance of model changes against severities in matplotlib

    Args:
        accuracies (list): list of accuracies or performances
        severities (list, optional): list of integers representing severities. Defaults to None.

    Returns:
        figure: resultant graph
    """
    if severities is None:
        severities = list(range(len(accuracies)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(severities, accuracies, marker='o', linestyle='-', color='b')
    ax.set_xlabel("Severity")
    ax.set_ylabel("mAP value")
    ax.set_title("Mean Average Precision (mAP) vs Severity")
    ax.set_ylim(0, 1)
    ax.set_xticks(severities)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(False)
    fig.tight_layout()  

    return fig

def plot_accuracy_vs_severity_plotly(accuracies, severities=None):
    """Plots the accuracy/performance of model changes against severities in plotly

    Args:
        accuracies (list): list of accuracies or performances
        severities (list, optional): list of integers representing severities. Defaults to None.

    Returns:
        figure: resultant graph
    """
    if severities is None:
        severities = list(range(len(accuracies)))

    fig = go.Figure()

    # Add line plot with markers
    fig.add_trace(go.Scatter(
        x=severities,
        y=accuracies,
        mode='lines+markers',
        line=dict(color='blue'),
        marker=dict(size=8),
        name='Accuracy'
    ))

    # Update layout
    fig.update_layout(
        title='Model Accuracy vs Severity',
        xaxis_title='Severity',
        yaxis_title='Accuracy',
        xaxis=dict(tickmode='array', tickvals=severities),
        yaxis=dict(range=[0, 1] if max(accuracies) <= 1 else None),
        width=800,
        height=500,
        template='simple_white'
    )

    # fig.show()
    return fig

def best_fit_gradient(x_values, y_values):
    """
    Calculate the gradient (slope) of the best-fit line using the least squares method.
    
    Args:
        x_values (list or array): Independent variable values.
        y_values (list or array): Dependent variable values.
    
    Returns:
        float: Slope of the best-fit line.
    """
    x_mean = np.mean(x_values)
    y_mean = np.mean(y_values)
    
    numerator = np.sum((x_values - x_mean) * (y_values - y_mean))
    denominator = np.sum((x_values - x_mean) ** 2)
    
    return numerator / denominator

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