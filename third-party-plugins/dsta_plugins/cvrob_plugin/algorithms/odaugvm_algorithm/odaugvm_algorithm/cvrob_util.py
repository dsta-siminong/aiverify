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
import gc

def evaluate_detection(model, loader, device, iou_threshold=0.5):
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

    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            preds = [{k: v.cpu() for k, v in o.items()} for o in outputs]
            gts = [{k: v for k, v in t.items()} for t in targets]

            metric.update(preds, gts)

    result = metric.compute()
    metric.reset()
    return result["map"].item() #generalize in future

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
    num_epochs = 1 if num_epochs is None else num_epochs
    num_epochs = 1 if aug_class.deterministic else num_epochs
    print("===")
    print("Aug name", aug_class.name)
    print(f"Evaluating on severity 0/None...")
    base_map = evaluate_detection(model, test_loader, device)
    print(f"mAP at severity 0/None: {base_map:.4f}")
    severities = aug_class.severities #[x for x in range(len(aug_class.severities))]
    maps = [base_map]
    for severity_idx, severity in enumerate(severities):
        print(f"Evaluating on severity {severity}...")
        all_map = []
        for i in range(num_epochs):
            seed = 1000*i + severity_idx 
            aug_class.set_seed(seed)
            corrupted_loader = aug_class.corr_func_dataloader(test_loader, severity_idx=severity) #TO BE FIXED
            corr_map = evaluate_detection(model, corrupted_loader, device, iou_threshold)
            all_map.append(corr_map)
            print(f"epoch {i+1}: {corr_map}")

            del corrupted_loader
            torch.cuda.empty_cache()
            gc.collect()
        final_map = sum(all_map)/len(all_map)
        maps.append(final_map)
        print(f"mAP at severity {severity}: {final_map:.4f}")

    # Plot results
    fig = None
    if plot_graphs is not False:
        fig = plot_accuracy_vs_severity(maps, ["None"]+severities, plot_graphs)  
    fig_path = directory / f"accuracy_vs_severity_{aug_class.name}.png"
    fig.savefig(fig_path)
    plt.close()
    return best_fit_gradient(list(range(len(severities)+1)), maps), maps, fig_path

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
        loat: Slope of the best-fit line.
    """
    x_mean = np.mean(x_values)
    y_mean = np.mean(y_values)
    
    numerator = np.sum((x_values - x_mean) * (y_values - y_mean))
    denominator = np.sum((x_values - x_mean) ** 2)
    
    return numerator / denominator

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