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
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

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

import concurrent.futures

def evaluate_via_api(model, loader):
    """
    Evaluate a model served behind an HTTP API over a data loader.

    Each batch is serialised to ``.npy`` and POSTed to the API URL; predictions
    from the JSON response are compared against the batch targets.

    The primary request format is a raw ``application/octet-stream`` body,
    which is the fast path. If an older API rejects that format with HTTP 400,
    the request is retried using the legacy multipart ``files=`` format.

    The next DataLoader batch is prefetched in a background thread so that
    data loading can overlap with the current batch's HTTP request.

    Args:
        model (str): API URL that accepts a ``.npy`` batch and returns predictions.
        loader (torch.utils.data.DataLoader): Data loader to evaluate over.

    Returns:
        Tuple[float, np.ndarray, np.ndarray]:
            Accuracy (percent), predicted labels, and true labels.

    Raises:
        requests.HTTPError:
            If an API request returns an error status after any applicable
            fallback has been attempted.
    """
    API_URL = model
    correct, total = 0, 0
    predicted_labels, true_labels = [], []

    session = requests.Session()

    # ------------------------------------------------------------------
    # Negotiate wire format once.
    #
    # Newer servers may advertise uint8 support. Older servers may not
    # have /health or may not advertise it, so float32 remains the safe
    # default.
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
    # Prefetch the next batch while the current batch is being sent to
    # the API. Only one batch is prefetched so memory usage stays bounded.
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
            # Wait for the current/pre-fetched batch.
            batch = next_future.result()

            if batch is None:
                break

            inputs, targets = batch

            data_load_time = time.perf_counter() - loop_end
            print("data loading (dataset->batch):", data_load_time)

            # Immediately start loading the following batch. This happens
            # concurrently with serialization and the HTTP request below.
            next_future = executor.submit(fetch_next)

            # --------------------------------------------------------------
            # Serialize batch.
            # --------------------------------------------------------------
            batch_np = inputs.numpy()

            if use_uint8:
                batch_np = (
                    (batch_np * 255.0)
                    .round()
                    .clip(0, 255)
                    .astype(np.uint8)
                )

            buffer = io.BytesIO()
            np.save(buffer, batch_np)
            payload = buffer.getvalue()

            # --------------------------------------------------------------
            # Fast path:
            #
            # Send the raw .npy bytes directly as application/octet-stream.
            # This avoids multipart/form-data overhead.
            # --------------------------------------------------------------
            t0 = time.perf_counter()

            response = session.post(
                API_URL,
                data=payload,
                headers={"Content-Type": "application/octet-stream"},
                timeout=300,
            )

            # --------------------------------------------------------------
            # Compatibility fallback:
            #
            # Older versions of the API may expect a multipart upload with
            # a field named "file". Only retry for HTTP 400, since that is
            # the behavior of the old implementation.
            # --------------------------------------------------------------
            if response.status_code == 400:
                print(
                    "Fast raw-body request returned HTTP 400; "
                    "retrying with legacy multipart file upload."
                )

                print("========================")
                print(response.status_code)
                print(response.reason)
                print(response.text)
                print("========================")

                fallback_np = inputs.numpy()  # original float32, re-derived from inputs
                fallback_buffer = io.BytesIO()
                np.save(fallback_buffer, fallback_np)

                response = session.post(
                    API_URL,
                    files={"file": ("batch.npy", fallback_buffer, "application/octet-stream")},
                    timeout=300,
                )

            t1 = time.perf_counter()

            # Raise for any error remaining after the fallback.
            response.raise_for_status()

            # --------------------------------------------------------------
            # Decode response.
            # --------------------------------------------------------------
            result = response.json()
            t2 = time.perf_counter()

            print("HTTP round-trip:", t1 - t0)
            print("JSON decode:", t2 - t1)

            predicted = np.array(result["predictions"])
            targets_np = targets.numpy()

            # --------------------------------------------------------------
            # Accumulate metrics.
            # --------------------------------------------------------------
            correct += (predicted == targets_np).sum()
            total += len(targets_np)

            predicted_labels.extend(predicted)
            true_labels.extend(targets_np)

            loop_end = time.perf_counter()

    finally:
        # Ensure resources are cleaned up even if an API request or
        # DataLoader raises an exception.
        executor.shutdown()
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

    Mirrors the wire-format handling of :func:`evaluate_via_api`, but for a
    single image rather than a batch:

    * The server is probed via ``/health`` to see whether it advertises uint8
      support. If it does, the image is quantised to uint8 for a smaller
      payload; otherwise float32 is sent.
    * The primary request sends the raw ``.npy`` bytes as
      ``application/octet-stream`` (the fast path served by ``app_uint8.py``).
    * If an older server (``app.py``) rejects that with HTTP 400, the request
      is retried with the legacy multipart ``files={"file": ...}`` upload using
      the original float32 array.

    A single (CHW) image is sent, so both server versions respond with a
    ``"prediction"`` key.

    Args:
        model (str): API URL that accepts a ``.npy`` image and returns a prediction.
        display_image (np.ndarray): CHW image array (float32 in ``[0, 1]``) to classify.

    Returns:
        int: The predicted class index.

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

        # --------------------------------------------------------------
        # Serialize image.
        # --------------------------------------------------------------
        image_np = np.asarray(display_image)

        if use_uint8:
            image_np = (
                (image_np * 255.0)
                .round()
                .clip(0, 255)
                .astype(np.uint8)
            )

        buffer = io.BytesIO()
        np.save(buffer, image_np)
        payload = buffer.getvalue()

        # --------------------------------------------------------------
        # Fast path: raw .npy bytes as application/octet-stream, avoiding
        # multipart/form-data overhead.
        # --------------------------------------------------------------
        response = session.post(
            API_URL,
            data=payload,
            headers={"Content-Type": "application/octet-stream"},
            timeout=300,
        )

        # --------------------------------------------------------------
        # Compatibility fallback: older servers expect a multipart upload
        # with a field named "file". Only retry on HTTP 400, matching the
        # behavior of the old implementation.
        # --------------------------------------------------------------
        if response.status_code == 400:
            print(
                "Fast raw-body request returned HTTP 400; "
                "retrying with legacy multipart file upload."
            )

            fallback_buffer = io.BytesIO()
            np.save(fallback_buffer, np.asarray(display_image))  # original float32
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
    
def augmentation_gradient(model, test_loader, device, aug_class, plot_graphs=False, directory=Path(), num_epochs=1, display_idx=None):
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
        display_idx (int, optional): if given, capture the scored dataset and the
            prediction at this index per severity so the caller can build display
            samples without a second corrupt+inference pass.

    Returns:
        Tuple:
            best_fit_gradient (float): best fit line gradient of graph of performance vs severity (of augmentation)
            accuracies (list): list of floats of performance metric
            fig (figure): outputs figure of plot_graphs library if not plot_graphs not False, else None
            display_scored (dict): maps severity label ("None" plus each severity) to ``(dataset, prediction_at_display_idx)`` from the scored pass; empty when ``display_idx`` is None.
            
    """

    print(f"[mem at the start of augmentation_gradient stuff] {mem_mb():.1f} MB")
    num_epochs = 1 if num_epochs is None else num_epochs
    num_epochs = 1 if aug_class.deterministic else num_epochs
    print("===")
    print("Aug name", aug_class.name)
    print(f"Evaluating on severity 0/None...")
    print(f"[mem before first evaluate] {mem_mb():.1f} MB")
    base_acc, base_y_pred, _ = evaluate(model, test_loader, device)
    print(f"[mem after first evaluate] {mem_mb():.1f} MB")
    print(f"Accuracy at severity 0/None: {base_acc:.4f}")
    severities = aug_class.severities #[x for x in range(len(aug_class.severities))]
    accuracies = [base_acc]

    # Per-severity display data pulled from the same scored pass (shuffle=False,
    # so display_idx aligns with y_pred). The clean/"None" case reuses the base pass.
    display_scored = {}
    if display_idx is not None:
        display_scored["None"] = (test_loader.dataset, base_y_pred[display_idx])

    for severity_idx, severity in enumerate(severities):
        print(f"Evaluating on severity {severity}...")
        print(f"[mem before corr_func_dataloader] {mem_mb():.1f} MB")
        all_acc = []
        corrupted_loader = None; y_pred = None
        for i in range(num_epochs):
            seed = 1000*i + severity_idx
            aug_class.set_seed(seed)
            corrupted_loader = aug_class.corr_func_dataloader(test_loader, severity_idx=severity)
            print(f"[mem after corr_func_dataloader] {mem_mb():.1f} MB")
            acc, y_pred, _ = evaluate(model, corrupted_loader, device)
            print(f"[mem after evaluate] {mem_mb():.1f} MB")
            all_acc.append(acc)
            print(f"epoch {i+1}: {acc}")
        final_acc = sum(all_acc)/len(all_acc)
        accuracies.append(final_acc)
        print(f"Accuracy at severity {severity}: {final_acc:.4f}")
        # Display the last epoch's realization: its dataset and predictions. The
        # seed set on the last epoch persists, so dataset[display_idx] reproduces
        # exactly the pixels the model scored.
        if display_idx is not None:
            display_scored[str(severity)] = (corrupted_loader.dataset, y_pred[display_idx])

    # Plot results
    fig_path = directory / f"accuracy_vs_severity_{aug_class.name}.png"
    fig = None
    if plot_graphs is not False:
        fig = plot_accuracy_vs_severity(accuracies, ["None"]+severities, plot_graphs)
        fig.savefig(fig_path)
        plt.close()
    return best_fit_gradient(list(range(len(severities)+1)), accuracies), accuracies, fig_path, display_scored

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

    def __init__(self, image_paths, labels, cache=True):
        """
        Store the image paths and labels for lazy loading.

        Args:
            image_paths (list[str]): Image file paths.
            labels (list): Label per image, aligned with ``image_paths``.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transforms.ToTensor()
        self.cache = cache
        self._cache = {}


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
        if self.cache and idx in self._cache:
            return self._cache[idx], self.labels[idx]

        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)

        if self.cache:
            self._cache[idx] = image

        return image, self.labels[idx]

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