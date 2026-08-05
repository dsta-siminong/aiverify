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
from torch.utils.data import Dataset
from torchvision import transforms
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

def evaluate(model, loader, device):
    """
    Evaluate a model over a loader, dispatching by model kind.

    A string ``model`` is treated as an API URL and evaluated remotely; anything
    else is evaluated locally on ``device``.

    Args:
        model: A torch model, or an API URL string for remote evaluation.
        loader (torch.utils.data.DataLoader): Data loader to evaluate over.
        device (torch.device): Device the local model runs on.

    Returns:
        Tuple[float, np.ndarray, np.ndarray]: Accuracy (percent), predicted
            labels, and true labels.
    """
    if isinstance(model, str):
        print("STRING MODEL!")
        print(model)
        return evaluate_via_api(model, loader)
    else:
        print("Direct model")
        print(type(model), type(model), device)
        return evaluate_direct(model, loader, device)

def evaluate_via_api(model, loader):
    """
    Evaluate a model served behind an HTTP API over a data loader.

    Each batch is serialised to ``.npy`` and POSTed to the API URL; predictions
    from the JSON response are compared against the batch targets.

    Args:
        model (str): API URL that accepts a ``.npy`` batch and returns predictions.
        loader (torch.utils.data.DataLoader): Data loader to evaluate over.

    Returns:
        Tuple[float, np.ndarray, np.ndarray]: Accuracy (percent), predicted
            labels, and true labels.

    Raises:
        requests.HTTPError: If any API request returns an error status.
    """
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
    """
    Predict the class of a single image, dispatching by model kind.

    A string ``model`` is treated as an API URL; otherwise the image is run
    through the local model on ``device``.

    Args:
        model: A torch model, or an API URL string for remote prediction.
        display_image (np.ndarray): CHW image array to classify.
        device (torch.device): Device the local model runs on.

    Returns:
        int: The predicted class index.
    """
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
    """
    Predict the class of a single image via an HTTP API.

    Serialises the image to ``.npy``, POSTs it to the API URL, and returns the
    predicted class from the JSON response.

    Args:
        model (str): API URL that accepts a ``.npy`` image and returns a prediction.
        display_image (np.ndarray): Image array to classify.

    Returns:
        int: The predicted class index.

    Raises:
        requests.HTTPError: If the API request returns an error status.
    """
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
    
def augmentation_gradient(model, test_loader, device, aug_class, plot_graphs=False, directory=Path(), num_epochs=1):
    """
    Evaluates how the model performance varies against the given augmentation/corruption

    Augments across severity 1-5 (and 0) and outputs the performance change

    Args:
        model: model
        test_loader (torch.dataloader): test data loader
        device (torch.device): device model is on
        corr_func (function): corruption function to take in images (np array) / give corrupted dataloader
        plot_graphs: either False for no graph, or string for which graphing library to use
        corr_kwargs (dict): corruption arguments

    Returns:
        Tuple:
            best_fit_gradient (float): best fit line gradient of graph of performance vs severity (of augmentation)
            accuracies (list): list of floats of performance metric 
            fig (figure): outputs figure of plot_graphs library if not plot_graphs not False, else None
    """

    print(f"[mem at the start of augmentation_gradient stuff] {mem_mb():.1f} MB")
    num_epochs = 1 if num_epochs is None else num_epochs
    num_epochs = 1 if aug_class.deterministic else num_epochs
    print("===")
    print("Aug name", aug_class.name)
    print(f"Evaluating on severity 0/None...")
    print(f"[mem before first evaluate] {mem_mb():.1f} MB")
    base_acc, _ , _ = evaluate(model, test_loader, device)
    print(f"[mem after first evaluate] {mem_mb():.1f} MB")
    print(f"Accuracy at severity 0/None: {base_acc:.4f}")
    severities = aug_class.severities #[x for x in range(len(aug_class.severities))]
    accuracies = [base_acc]

    for severity_idx, severity in enumerate(severities):
        print(f"Evaluating on severity {severity}...")
        print(f"[mem before corr_func_dataloader] {mem_mb():.1f} MB")
        all_acc = []
        for i in range(num_epochs):
            seed = 1000*i + severity_idx 
            aug_class.set_seed(seed)
            corrupted_loader = aug_class.corr_func_dataloader(test_loader, severity_idx=severity)
            print(f"[mem after corr_func_dataloader] {mem_mb():.1f} MB")
            acc, _,_ = evaluate(model, corrupted_loader, device)
            print(f"[mem after evaluate] {mem_mb():.1f} MB")
            all_acc.append(acc)
            print(f"epoch {i+1}: {acc}")
        final_acc = sum(all_acc)/len(all_acc)
        accuracies.append(final_acc)
        print(f"Accuracy at severity {severity}: {final_acc:.4f}")

    # Plot results
    fig_path = directory / f"accuracy_vs_severity_{aug_class.name}.png"
    fig = None
    if plot_graphs is not False:
        fig = plot_accuracy_vs_severity(accuracies, ["None"]+severities, plot_graphs)  
        fig.savefig(fig_path)
        plt.close()
    return best_fit_gradient(list(range(len(severities)+1)), accuracies), accuracies, fig_path

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
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Accuracy vs Severity")
    ax.set_ylim(0, 100)
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

class ImageDataset(Dataset):
    """
    Lazy image dataset that decodes each image from disk on access.

    Stores only file paths and labels, decoding and transforming an image only
    when indexed, so memory stays proportional to the batch rather than the set.
    """

    def __init__(self, image_paths, labels):
        """
        Store the image paths and labels for lazy loading.

        Args:
            image_paths (list[str]): Image file paths.
            labels (list): Label per image, aligned with ``image_paths``.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transforms.ToTensor()

    def __len__(self):
        """
        Return the number of images in the dataset.

        Returns:
            int: Count of image paths.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Load, transform, and return the image and label at ``idx``.

        Opens the image as RGB and applies ``self.transform`` to produce a
        (3, H, W) tensor.

        Args:
            idx (int): Index of the item to fetch.

        Returns:
            Tuple[torch.Tensor, Any]: The transformed image tensor and its label.
        """
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)  # (3, H, W)
        label = self.labels[idx]
        return image, label

def pad_collate(batch):
    """
    Collate variable-sized images into a batch by zero-padding.

    Pads every image on the bottom and right up to the batch's maximum height and
    width, then stacks them with their labels into tensors.

    Args:
        batch (list): Sequence of ``(image_tensor, label)`` pairs, where each
            image is a (C, H, W) tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Stacked padded images and a long
            tensor of labels.
    """
    images, labels = zip(*batch)

    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    padded_images = []

    for img in images:
        _, h, w = img.shape

        # Pad on the bottom and right
        pad = (0, max_w - w,   # left, right
               0, max_h - h)   # top, bottom

        padded_images.append(nn.functional.pad(img, pad, value=0))

    images = torch.stack(padded_images)
    labels = torch.tensor(labels, dtype=torch.long)

    return images, labels