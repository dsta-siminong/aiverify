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

def evaluate(model, loader, device):
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
    model.eval()
    correct, total = 0, 0
    predicted_labels, true_labels = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(targets.cpu().numpy())
    return 100 * correct / total, np.array(predicted_labels), np.array(true_labels)

def collect_probs(model, dataloader, device):
    model.eval()

    probs = []
    labels = []
    images = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            p = F.softmax(logits, dim=1)

            probs.append(p.cpu())
            labels.append(y.cpu())
            images.append(x.cpu())

    return (
        torch.cat(images),
        torch.cat(probs),
        torch.cat(labels),
    )

    return torch.cat(probs), torch.cat(labels)

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
    img_tensor = transforms.ToTensor()(img_array).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(img_tensor.to(device)).cpu()
        probs = torch.softmax(output, 1).numpy()[0]
        label = torch.max(output, 1)[1][0].item()

    return probs, label

def triplets(s):
    items = s.split()
    assert len(items) % 3 == 0, "Input length must be a multiple of 3"
    return [items[i:i+3] for i in range(0, len(items), 3)]

# def get_corruption_helpers(library='albumentations'):
#     """
#     Returns the corruption parameters associated with a given library

#     Two-length tuples: first being corruption function (pure from library), second being parameters to be passed in

#     Args:
#         library (str): corruption library name
    
#     Returns:
#         Tuple:
#             augmentation_list (list): list of two-length tuples 
#             augmentation_str (list): list of names of the given libraries
#             corrupt_func (function): custom corrupt function: takes in images np array, severity, and augmentation parameters and returns corrupted images np array
#     """
#     if library in ['albumentations', 'album']:
#         a1, a2 = get_album_augmentations_list()
#         return a1, a2, corrupt_func_album
#     if library in ['nrtk']:
#         n1, n2 = get_nrtk_augmentations_list()
#         return n1, n2, corrupt_func_nrtk
#     if library in ['imagecorruptions', 'imagecorr', 'imagecorrupt', 'ic', 'imcor']:
#         i1, i2 = get_imagecorrupt_augmentations_list()
#         return i1, i2, corrupt_func_imagecorrupt
#     if library in ['augly']:
#         u1, u2 = get_augly_augmentations_list()
#         return u1, u2, corrupt_func_augly
#     else:
#         raise Exception('Invalid library called')

# def get_corrupted_dataloader(testloader, corr_func, severity=1, corr_kwargs=None):
#     """
#     Make a dataloader with data that is of the augmented/corrupted version of the original dataloader

#     Args:
#         testloader (torch.Dataloader): original dataloader
#         corr_func (function): corruption function  that takes in ([numpy array of images], severity, corr_kwargs)
#         severity (int); extent of severity between 0-5.
#         corr_kwargs (dict): dictionary of extra parameters to send into corr_func

#     Returns:
#         torch.utils.data.DataLoader: corrupted version of original dataloader
#     """
#     if corr_kwargs is None:
#         corr_kwargs = {}  # Default to an empty dictionary
    
#     corrupted_images = []
#     corrupted_labels = []
    
#     for images, labels in testloader:
#         images_np = (images * 255).byte().numpy().transpose(0, 2, 3, 1)  # Convert to HWC format and uint8
        
#         # Apply corruption function with provided parameters
#         corrupted = corr_func(images_np, severity, corr_kwargs)
        
#         corrupted = torch.tensor(corrupted.transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0  # Convert back to CHW format and normalize
#         corrupted_images.append(corrupted)
#         corrupted_labels.append(labels)
    
#     corrupted_dataset = torch.utils.data.TensorDataset(torch.cat(corrupted_images), torch.cat(corrupted_labels))
#     return torch.utils.data.DataLoader(corrupted_dataset, batch_size=128, shuffle=False)

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
    # print(len(np.unique(y_true)))
    # print(pred_probs.shape[1])

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

def collect_probs(model, dataloader, device):
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
    Try to infer the number of output classes from a PyTorch image classification model.
    Works for most architectures by inspecting the last linear/conv layer.
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