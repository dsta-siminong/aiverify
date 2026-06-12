import requests
from PIL import Image
from io import BytesIO
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

def compute_detection_brittleness(predA_list, predB_list, gt_list, iou_thresh=0.5):

    results = []

    for i in range(len(gt_list)):

        predA = predA_list[i]
        predB = predB_list[i]
        gt = gt_list[i]

        matched_A = match_with_gt(predA, gt, iou_thresh)
        matched_B = match_with_gt(predB, gt, iou_thresh)

        brittleness_vals = []

        for gt_idx in gt["boxes"].shape[0]:

            score_A = matched_A.get(gt_idx, 0.0)
            score_B = matched_B.get(gt_idx, 0.0)

            brittleness_vals.append(score_A - score_B)

        brittleness = max(brittleness_vals) if brittleness_vals else 0.0

        results.append({
            "index": i,
            "brittleness": brittleness,
            "predA": predA,
            "predB": predB,
            "gt": gt
        })

    return results

# def collect_detection_scores(model, loader, device, score_mode="max"):
#     model.eval()

#     all_scores = []
#     all_imgs = []

#     with torch.no_grad():
#         for images, _ in loader:
#             images = [img.to(device) for img in images]
#             outputs = model(images)

#             outputs = [
#                 {k: v.cpu() for k, v in o.items()}
#                 for o in outputs
#             ]

#             for img, out in zip(images, outputs):
#                 scores = out["scores"]

#                 if len(scores) == 0:
#                     image_score = 0.0
#                 else:
#                     if score_mode == "max":
#                         image_score = scores.max().item()
#                     elif score_mode == "mean":
#                         image_score = scores.mean().item()
#                     else:
#                         raise ValueError("Unknown score_mode")

#                 all_scores.append(image_score)
#                 all_imgs.append(img.cpu())

#     return torch.stack(all_imgs), torch.tensor(all_scores)

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

def collect_probs(model, dataloader, device):
    """
    Collect model predictions, labels, and input images from a DataLoader.

    The function runs the model in evaluation mode over all batches in the dataloader,
    computes softmax probabilities for each batch, and accumulates the input images,
    predicted probabilities, and ground truth labels.

    Args:
        model (nn.Module): A PyTorch model for which predictions are collected.
        dataloader (DataLoader): PyTorch DataLoader providing input batches (images and labels).
        device (torch.device): Device to run the model on (e.g., CPU or GPU).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - images: Tensor of all input images concatenated across batches.
            - probs: Tensor of predicted probabilities for each input.
            - labels: Tensor of ground truth labels for each input.
    """
    model.eval()

    probs = []
    labels = []
    images = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            p = torch.nn.functional.softmax(logits, dim=1)

            probs.append(p.cpu())
            labels.append(y.cpu())
            images.append(x.cpu())

    return (
        torch.cat(images),
        torch.cat(probs),
        torch.cat(labels),
    )

    return torch.cat(probs), torch.cat(labels)

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

# ==== OTHER FUNCTIONS THAT ARE NOT USED FOR THIS WHOLE ALGO BUT I DON'T WANT TO DELETE THEM YET ====

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64*8*8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_image_from_url(image_url):
    """
    Downloads an image from a URL and converts it to a NumPy array.

    Args:
        image_url (str): The URL pointing to the image.

    Returns:
        np.ndarray: The image as a NumPy array.
    """
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    image_array = np.array(image)
    return image_array

def get_image_from_path(image_path):
    """
    Loads an image from a local file path and converts it to a NumPy array.

    Args:
        image_path (str): The path to the image file.

    Returns:
        np.ndarray: The image as a NumPy array.
    """
    image = Image.open(image_path)
    image_np = np.array(image)
    return image_np

def get_logits(model, dataloader, device):
    """
    Get the features, or the inputs before the last layer

    Args:
        test_loader (torch.Dataloader): data loader for test data
        model (torch.nn.Module): torch model
        device (torch.device): device model is on

    Returns: 
        logits (np.array): array of outputs just before they pass through the softmax/max/last layer for prediction
    """
    labels = np.empty((0,))

    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        with tqdm(dataloader) as progress:
            for batch_idx, (data, label) in enumerate(progress):
                data, label = data, label.long()  # No need to move to GPU, stay on CPU
                data = data.to(device)
                label = label.to(device)
                feature = model(data)  # Forward pass

                labels = np.concatenate((labels, label.cpu()))  # Ensure labels are on CPU
                if batch_idx == 0:
                    features = feature.detach().cpu()  # Ensure features are on CPU
                else:
                    features = np.concatenate((features, feature.detach().cpu()), axis=0)
    
    return features, labels

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
    ax.set_xticks(severities)
    ax.grid(True)
    
    plt.show()
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

    fig.show()
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

def evaluate_1img(model, device, img_array):
    """
    Evaluate a single image using a PyTorch model and return predicted probabilities and label.

    Args:
        model (nn.Module): A PyTorch model for image classification.
        device (torch.device): Device to run the model on (CPU or GPU).
        img_array (numpy.ndarray or PIL.Image.Image): Input image as a NumPy array or PIL Image.

    Returns:
        Tuple[numpy.ndarray, int]:
            - probs: Softmax probabilities for each class as a NumPy array.
            - label: Predicted class index as an integer.
    """
    img_tensor = transforms.ToTensor()(img_array).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(img_tensor.to(device)).cpu()
        probs = torch.softmax(output, 1).numpy()[0]
        label = torch.max(output, 1)[1][0].item()

    return probs, label

def get_metric_dict():
    d = {
        "accuracy": get_accuracy,
        "correct_class_proba": get_correct_class_proba,
        "max_proba": get_max_proba,
        "f1": get_f1,
        "recall": get_recall,
        "precision": get_precision,
        "auc": get_auc,
        "ece": get_ece
    }
    return d

def get_accuracy(base_acc, y_pred, y_true, features, labels, pred_probs):
    return base_acc/100

def get_precision(base_acc, y_pred, y_true, features, labels, pred_probs):
    return precision_score(y_true, y_pred, average='macro', zero_division=np.nan)

def get_recall(base_acc, y_pred, y_true, features, labels, pred_probs):
    return recall_score(y_true, y_pred, average='macro', zero_division=np.nan)

def get_f1(base_acc, y_pred, y_true, features, labels, pred_probs):
    return f1_score(y_true, y_pred, average='macro')

def get_auc(base_acc, y_pred, y_true, features, labels, pred_probs):
    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
    y_pred_proba_filtered = pred_probs[:, np.unique(y_true)]
    return roc_auc_score(y_true_bin, y_pred_proba_filtered, average='macro', multi_class='ovr')

def get_correct_class_proba(base_acc, y_pred, y_true, features, labels, pred_probs):
    return np.array([x[labels[i]] for i,x in enumerate(pred_probs)]).mean()

def get_max_proba(base_acc, y_pred, y_true, features, labels, pred_probs):
    return np.array([max(x) for i,x in enumerate(pred_probs)]).mean()

def get_f1(base_acc, y_pred, y_true, features, labels, pred_probs):
    return f1_score(y_true, y_pred, average='macro')

def get_ece(base_acc, y_pred, y_true, features, labels, pred_probs, n_bins=20):
    accs = y_pred == y_true
    bin_boundaries = np.linspace(0,1,n_bins+1)
    confs = np.max(pred_probs, axis=1)
    ece = 0
    for i in range(n_bins):
        bin_l, bin_u = bin_boundaries[i], bin_boundaries[i+1]
        in_bin = (confs > bin_l) & (confs <= bin_u)
        if in_bin.any():
            bin_acc = accs[in_bin].mean()
            bin_conf = confs[in_bin].mean()
            ece += np.abs(bin_acc - bin_conf) * in_bin.mean()

    return float(ece)

def collect_detection_scores(model, loader, device, score_mode="max"):
    model.eval()

    all_scores = []
    all_imgs = []

    with torch.no_grad():
        for images, _ in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for img, out in zip(images, outputs):

                scores = out["scores"]

                if len(scores) == 0:
                    image_score = 0.0
                else:
                    if score_mode == "max":
                        image_score = scores.max().item()
                    elif score_mode == "mean":
                        image_score = scores.mean().item()
                    elif score_mode == "sum":
                        image_score = scores.sum().item()

                all_scores.append(image_score)
                all_imgs.append(img.cpu())

    return torch.stack(all_imgs), torch.tensor(all_scores)

def collect_detection_predictions(model, loader, device):
    model.eval()
    all_imgs = []
    all_preds = []

    with torch.no_grad():
        for images, _ in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for img, out in zip(images, outputs):
                all_imgs.append(img.cpu())
                all_preds.append({k: v.cpu() for k, v in out.items()})

    return torch.stack(all_imgs), all_preds


def image_brittleness(predA, predB, iou_thresh=0.5):
    boxesA, labelsA, scoresA = predA["boxes"], predA["labels"], predA["scores"]
    boxesB, labelsB, scoresB = predB["boxes"], predB["labels"], predB["scores"]

    if len(boxesA) == 0:
        return 0.0

    usedB = set()
    drops = []

    orderA = torch.argsort(scoresA, descending=True)

    for ai in orderA.tolist():
        boxA = boxesA[ai]
        labelA = int(labelsA[ai].item())
        scoreA = float(scoresA[ai].item())

        best_j = None
        best_iou = 0.0
        best_scoreB = 0.0

        for bj in range(len(boxesB)):
            if bj in usedB:
                continue
            if int(labelsB[bj].item()) != labelA:
                continue

            iou = box_iou(boxA.unsqueeze(0), boxesB[bj].unsqueeze(0))[0, 0].item()
            if iou > best_iou:
                best_iou = iou
                best_j = bj
                best_scoreB = float(scoresB[bj].item())

        if best_j is None or best_iou < iou_thresh:
            drop = scoreA
        else:
            usedB.add(best_j)
            drop = max(0.0, scoreA - best_scoreB) + scoreA * (1.0 - best_iou)

        drops.append(drop)

    return max(drops) if drops else 0.0

def serialize_detection(pred):
    return {
        k: v.cpu().tolist() for k,v in pred.items()
    }

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