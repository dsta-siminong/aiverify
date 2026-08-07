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

def evaluate_detection_api(model, loader, iou_threshold=0.5):
    """
    Evaluate a model served behind an HTTP API over a data loader.

    Each batch is serialised to ``.npy`` and POSTed to the API URL; predictions
    from the JSON response are compared against the batch targets.

    Args:
        model (str): API URL that accepts a ``.npy`` batch and returns predictions.
        loader (torch.utils.data.DataLoader): Data loader to evaluate over.

    Returns:
        float: mean Average Precision (mAP)

    Raises:
        requests.HTTPError: If any API request returns an error status.
    """
    API_URL = model
    metric = MeanAveragePrecision(iou_thresholds=[iou_threshold])

    for images, targets in loader:

        batch = torch.stack(images)
        batch_np = batch.numpy()

        buffer = io.BytesIO()
        np.save(buffer, batch_np)
        buffer.seek(0)
        t0 = time.perf_counter()
        response = requests.post(
            API_URL,
            files={"file": ("batch.npy", buffer, "application/octet-stream")},
        )
        t1 = time.perf_counter()
        response.raise_for_status()

        preds_json = response.json()["predictions"]
        t2 = time.perf_counter()
        print("- HTTP round-trip:", t1 - t0)
        print("-- JSON decode:", t2 - t1)

        preds = []
        for pred in preds_json:
            preds.append({
                "boxes": torch.tensor(pred["boxes"], dtype=torch.float32),
                "scores": torch.tensor(pred["scores"], dtype=torch.float32),
                "labels": torch.tensor(pred["labels"], dtype=torch.int64),
            })

        gts = [{k: v for k, v in t.items()} for t in targets]

        metric.update(preds, gts)

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

    Serialises the image to ``.npy``, POSTs it to the API URL, and returns the
    predicted class from the JSON response.

    Args:
        model (str): API URL that accepts a ``.npy`` image and returns a prediction.
        display_image (np.ndarray): Image array to classify.

    Returns:
        dict: prediction of the image represented by the bounding boxes of detection, 
        labels for classes of the boxes, and the scores of each detection

    Raises:
        requests.HTTPError: If the API request returns an error status.
    """
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