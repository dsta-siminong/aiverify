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

from collections import defaultdict
from torchvision.ops import box_iou

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

def evaluate_detection_detailed(
    model,
    loader,
    device,
    class_names,
    iou_thresh=0.5,
    score_thresh=0.5
):
    num_classes = len(class_names)
    matrix = np.zeros((num_classes, num_classes), dtype=np.float32)
    metric = MeanAveragePrecision(
        iou_thresholds=[iou_thresh]
    )

    # per-class accumulators
    stats = {
        class_name: {
            "TP": 0,
            "FP": 0,
            "FN": 0,
            "support": 0,
        }
        for class_name in class_names.values()
    }

    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            images = [
                img.to(device)
                for img in images
            ]
            targets = [
                {
                    k: v.to(device)
                    for k, v in t.items()
                }
                for t in targets
            ]
            outputs = model(images)
            print("\n=== LABEL SANITY CHECK ===")
            print("GT labels (batch):", [torch.unique(t["labels"]).tolist() for t in targets])
            print("Pred labels (batch):", [torch.unique(o["labels"]).tolist() for o in outputs])
            print("Class mapping keys:", list(class_names.keys()))
            preds = [
                {
                    k: v.cpu()
                    for k, v in o.items()
                }
                for o in outputs
            ]
            gts = [
                {
                    k: v.cpu()
                    for k, v in t.items()
                }
                for t in targets
            ]
            metric.update(preds, gts)
            print("\n=== IMAGE SUMMARY ===")
            print("num preds:", sum(len(o["boxes"]) for o in preds))
            print("num gts:", sum(len(t["boxes"]) for t in gts))

            # =================================================
            # PER-IMAGE MATCHING
            # =================================================

            for pred, gt in zip(preds, gts):

                pred_boxes = pred["boxes"]
                pred_labels = pred["labels"]
                pred_scores = pred["scores"]

                print("\n=== SCORE DISTRIBUTION ===")
                if len(pred_scores) > 0:
                    print("Raw scores min/max:",
                        pred_scores.min().item(),
                        pred_scores.max().item())
                else:
                    print("No predictions")
                print("Num predictions:", len(pred_scores))

                gt_boxes = gt["boxes"]
                gt_labels = gt["labels"]

                # score filtering
                keep = pred_scores >= score_thresh

                print("Kept after threshold:", keep.sum().item(), "/", len(keep))

                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]
                pred_scores = pred_scores[keep]

                # 🔥 CRITICAL: sort by confidence descending
                order = torch.argsort(pred_scores, descending=True)

                pred_boxes = pred_boxes[order]
                pred_labels = pred_labels[order]
                pred_scores = pred_scores[order]

                matched_gt = set()
                # print("\n=== MATCH SUMMARY ===")
                # print("Matched GTs:", len(matched_gt))
                # print("Total GTs:", len(gt_boxes))

                # -----------------------------------------
                # process predictions
                # -----------------------------------------
                for pbox, plabel in zip(pred_boxes, pred_labels):
                    class_name = class_names[str(plabel.item())]
                    best_iou = 0
                    best_gt_idx = -1

                    for gt_idx, (gbox, glabel) in enumerate(zip(gt_boxes, gt_labels)):
                        if gt_idx in matched_gt:
                            continue
                        iou = box_iou(
                            pbox.unsqueeze(0),
                            gbox.unsqueeze(0)
                        )[0, 0].item()
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx

                    if best_iou >= iou_thresh:
                        matched_gt.add(best_gt_idx)
                        gt_class = int(gt_labels[best_gt_idx])
                        pred_class = int(plabel)
                        matrix[gt_class, pred_class] += 1       # TP (diagonal) or class confusion (off-diagonal)

                        gt_class_name = class_names[str(gt_class)]
                        if gt_class == pred_class:
                            stats[class_name]["TP"] += 1
                        else:
                            stats[class_name]["FP"] += 1
                            stats[gt_class_name]["FN"] += 1
                    else:
                        # ghost prediction — predicted something, no GT box matched
                        matrix[0, int(plabel)] += 1             # GT=background, PRED=class x
                        stats[class_name]["FP"] += 1

                for gt_idx, glabel in enumerate(gt_labels):
                    class_name = class_names[str(glabel.item())]
                    stats[class_name]["support"] += 1
                    if gt_idx not in matched_gt:
                        # missed detection — GT existed, no prediction claimed it
                        matrix[int(glabel), 0] += 1             # GT=class x, PRED=background
                        stats[class_name]["FN"] += 1


                print("\n=== FINAL MATCH SUMMARY ===")
                print("Matched GTs:", len(matched_gt))
                print("Total GTs:", len(gt_boxes))
                print("Matched indices:", matched_gt)
    # =====================================================
    # FINAL METRICS
    # =====================================================


    # print("UNIQUE GT CLASSES IN MATRIX:", np.unique(np.where(matrix > 0)[0]))
    # print("UNIQUE PRED CLASSES IN MATRIX:", np.unique(np.where(matrix > 0)[1]))
    per_class = {}

    for class_name, s in stats.items():

        TP = s["TP"]
        FP = s["FP"]
        FN = s["FN"]

        precision = (
            TP / (TP + FP)
            if (TP + FP) > 0 else 0.0
        )

        recall = (
            TP / (TP + FN)
            if (TP + FN) > 0 else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        per_class[class_name] = {
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": s["support"],
        }

    map_result = metric.compute()


    print("\n=== FINAL MATRIX CHECK ===")
    print("Matrix sum:", np.diag(matrix).sum(), matrix.sum())
    print("Diagonal sum:", np.trace(matrix))
    print("Off-diagonal sum:", matrix.sum() - np.trace(matrix))
    print(set(pred_labels.tolist()), set(gt_labels.tolist()))
    # print(1/0)
    return {
        "map_50": map_result["map_50"].item(),
        "per_class": per_class,
        "matrix": matrix
    }

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
    
def augmentation_gradient_det(model, test_loader, device, aug_class, plot_graphs=False, directory=Path(), num_epochs=1):
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
            corr_map = evaluate_detection(model, corrupted_loader, device)
            all_map.append(corr_map)
            print(f"epoch {i+1}: {corr_map}")
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
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy vs Severity")
    y_limit = 100
    if max(accuracies) <= 1:
        y_limit = 1
    ax.set_ylim(0, y_limit)
    ax.set_xticks(severities)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True)
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