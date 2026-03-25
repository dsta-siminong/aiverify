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
    ax.set_ylim(0, 100)
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

def triplets(s):
    items = s.split()
    assert len(items) % 3 == 0, "Input length must be a multiple of 3"
    return [items[i:i+3] for i in range(0, len(items), 3)]

def augmentation_gradient(model, test_loader, device, aug_class, plot_graphs=False, directory=Path()):
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
    print("===")
    print("Aug name", aug_class.name)
    print(f"Evaluating on severity 0/None...")
    base_acc, _ , _ = evaluate(model, test_loader, device)
    print(f"Accuracy at severity 0/None: {base_acc:.4f}")
    severities = aug_class.severities #[x for x in range(len(aug_class.severities))]
    accuracies = [base_acc]
    for severity in severities:
        print(f"Evaluating on severity {severity}...")
        corrupted_loader = aug_class.corr_func_dataloader(test_loader, severity_idx=severity)
        acc, _,_ = evaluate(model, corrupted_loader, device)
        accuracies.append(acc)
        print(f"Accuracy at severity {severity}: {acc:.4f}")

    # Plot results
    fig = None
    if plot_graphs is not False:
        fig = plot_accuracy_vs_severity(accuracies, ["None"]+severities, plot_graphs)  
    fig_path = directory / f"accuracy_vs_severity_{aug_class.name}.png"
    fig.savefig(fig_path)
    plt.close()
    return best_fit_gradient(list(range(len(severities)+1)), accuracies), accuracies, fig_path

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