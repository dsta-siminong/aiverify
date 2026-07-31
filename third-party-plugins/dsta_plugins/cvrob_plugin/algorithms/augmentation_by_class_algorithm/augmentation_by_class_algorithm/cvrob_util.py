import requests
# from PIL import Image
# from io import BytesIO
import io
import torch 
import numpy as np
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# from tqdm import tqdm
import torch.nn as nn
import time
import resource
def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 

def evaluate(model, loader, device):
    if isinstance(model, str):
        print("STRING MODEL!")
        print(model)
        return evaluate_via_api(model, loader)
    else:
        print("Direct model")
        print(type(model), type(model), device)
        return evaluate_direct(model, loader, device)

def evaluate_via_api(model, loader):
    API_URL = model
    correct, total = 0, 0
    predicted_labels, true_labels = [], []

    session = requests.Session()

    for inputs, targets in loader:
        batch_np = inputs.numpy()

        buffer = io.BytesIO()
        np.save(buffer, batch_np)
        buffer.seek(0)
        t0 = time.perf_counter()
        response = session.post(
            API_URL,
            files={"file": ("batch.npy", buffer, "application/octet-stream")},
            timeout=300,
        )
        t1 = time.perf_counter()
        response.raise_for_status()

        result = response.json()
        t2 = time.perf_counter()
        print("HTTP round-trip:", t1 - t0)
        print("JSON decode:", t2 - t1)
        
        predicted = np.array(result["predictions"])
        targets_np = targets.numpy()

        correct += (predicted == targets_np).sum()
        total += len(targets_np)

        predicted_labels.extend(predicted)
        true_labels.extend(targets_np)

    session.close()

    return (
        100 * correct / total,
        np.array(predicted_labels),
        np.array(true_labels),
    )

def evaluate_direct(model, loader, device):
    """
    Evaluate model using data from loader

    Args:
        model (torch.nn.Module): torch model
        loader (torch.Dataloader): data loader
        device (torch.device): device model is on
    Returns: 
        tuple:
            accuracy (float): percentage of correctly predicted labels
            predicted_labels (np.array): predictions output by label
            true_labels (np.array): ground truth labels
    """
    model.eval(); model.to(device)
    correct, total = 0, 0
    predicted_labels, true_labels = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            print(f"[mem before model()] {mem_mb():.1f} MB")
            outputs = model(inputs)
            print(f"[mem after model()] {mem_mb():.1f} MB")
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(targets.cpu().numpy())
    return 100 * correct / total, np.array(predicted_labels), np.array(true_labels)

def get_prediction_from_image(model, display_image, device):
    if isinstance(model, str):
        return get_prediction_from_image_api(model, display_image)
    image = torch.tensor(display_image).unsqueeze(0).float()
    image = image.to(device)
    model = model.float(); model.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, prediction = torch.max(outputs, 1)
    prediction = prediction.item()
    return prediction

def get_prediction_from_image_api(model, display_image):
    API_URL = model
    buffer = io.BytesIO()
    np.save(buffer, display_image)
    buffer.seek(0)

    response = requests.post(
        API_URL,
        files={"file": ("array.npy", buffer, "application/octet-stream")},
    )
    response.raise_for_status()
    result = response.json()

    prediction = result["prediction"]  # already a plain int, no .item() needed
    return prediction

def triplets(s):
    """
    Split a whitespace-separated string provided by the custom parameters argument into groups of three items.
    This is for the future customization of the augmentation dictionary

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

def average_all_reports(reports):
    """
    Compute the element-wise average of sklearn classification report dictionaries.

    This function expects a list of report dictionaries with identical structure,
    where each class contains metric values (e.g., precision, recall, f1-score),
    and a top-level 'accuracy' key may also be present.

    Args:
        reports (List[Dict[str, Any]]): List of report dictionaries. Each report
            should have the same keys and nested metric structure.

    Returns:
        Dict[str, Any]: A dictionary with the same structure as the input reports,
        where each metric value is replaced by the average across all reports.
    """
    avg = {}
    for c in reports[0].keys(): #c is a class
        if c == 'accuracy':
            avg[c] = float(np.mean([r[c] for r in reports]))
            continue

        avg[c] = {}
        for metric in reports[0][c].keys():
            values = [r[c][metric] for r in reports]
            avg[c][metric] = float(np.mean(values))
    return avg

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
    # 1. Look for last Linear layer
    last_linear = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
    if last_linear is not None:
        return last_linear.out_features

    # 2. Fallback: look for last Conv layer (e.g., some classifiers end with conv)
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is not None:
        return last_conv.out_channels

    # 3. Fallback: try classifier / fc attributes
    for attr in ["fc", "classifier", "head", "heads"]:
        if hasattr(model, attr):
            module = getattr(model, attr)
            if isinstance(module, nn.Linear):
                return module.out_features
            elif isinstance(module, nn.Sequential):
                for layer in reversed(module):
                    if isinstance(layer, nn.Linear):
                        return layer.out_features

    raise RuntimeError("Could not determine number of classes.")

def handle_class_names_arg(class_names_arg, model):
    """
    Parse class names input and return a mapping from class index to name.

    The function supports three input modes:
    1. None or empty string: infer number of classes from the model, then does (2)
    2. Single integer: generate default class names using that count.
    3. Comma-separated names: use provided class names directly.

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
