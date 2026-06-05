import torch 
import numpy as np
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import plotly.graph_objects as go
# from augly.image import blur, brightness, random_noise, contrast, color_jitter, pixelization, sharpen
# from augly.image import aug_np_wrapper
# from .cvrob_util import evaluate
from plotly.subplots import make_subplots
# from imagecorruptions import corrupt
import matplotlib.pyplot as plt

import base64
from PIL import Image
import io
from pathlib import Path
from dataclasses import dataclass
import json 

@dataclass
class BrittlenessResultIndiv:
    """
    Represents the brittleness evaluation result for a single image.

    Attributes:
        index (int): Index of the input sample.
        label (int): True class label of the input.
        predA (int): Predicted class for the initial input (before).
        predB (int): Predicted class for the resultant (of the corruption) input (after).
        pA (float): Probability of the predicted class for the first input.
        pB (float): Probability of the predicted class for the second input.
        brittleness (float): Measure of change or instability between predictions.
    """
    index: int
    label: int
    predA: int
    predB: int
    pA: float
    pB: float
    brittleness: float

@dataclass
class BrittlenessResult:
    """
    Represents a collection of brittleness evaluation results for multiple images.

    Attributes:
        results (list): List of BrittlenessResultIndiv objects for each input.
        imgsA (list): List or tensor of the initial set of input images (before).
        imgsB (list): List or tensor of the resultant (of the corruption) set of input images (after).
        probs_A (list): List or tensor of predicted probabilities for imgsA.
        probs_B (list): List or tensor of predicted probabilities for imgsB.
        labels (list): List or tensor of true labels for all inputs.
    """
    results: list
    imgsA: list
    imgsB: list 
    probs_A: list
    probs_B: list 
    labels: list

def brittle_res_indiv_to_dict(bri):
    """
    Convert a single BrittlenessResultIndiv object to a dictionary.

    Args:
        bri (BrittlenessResultIndiv): The brittleness result to convert.

    Returns:
        dict: Dictionary containing keys 'index', 'label', 'predA', 'predB',
                'pA', 'pB', and 'brittleness' with corresponding values.
    """
    d ={
        "index": bri.index,
        "label": bri.label,
        "predA": bri.predA,
        "predB": bri.predB,
        "pA": bri.pA,
        "pB": bri.pB,
        "brittleness": bri.brittleness
    }
    return d 

def brittle_res_to_dict(br):
    def safe_convert(val):
        if isinstance(val, torch.Tensor):
            return val.numpy().tolist()
        elif isinstance(val, np.ndarray):
            return val.tolist()
        else:
            return val  # already a list/dict, pass through as-is

    d = {
        "results": [brittle_res_indiv_to_dict(r) for r in br.results],
        "imgsA": None,#safe_convert(br.imgsA),
        "imgsB": None,#safe_convert(br.imgsB),
        "probs_A": safe_convert(br.probs_A),
        "probs_B": safe_convert(br.probs_B),
        "labels": safe_convert(br.labels),
    }
    return d

def visualize_topk_matplotlib(
    results_sorted,
    imgs_A, imgs_B,
    scores_A, scores_B,
    K=10,
    class_names=None,   # optional only for GT boxes
    transform=None,
    directory=Path(),
    image_paths=None
):
    topk = results_sorted[:K]

    fig, axes = plt.subplots(K, 2, figsize=(12, 3 * K), constrained_layout=True)

    if K == 1:
        axes = np.array([axes])

    for row, res in enumerate(topk):
        i = res.index

        idx = Path(str(image_paths[i])).name if image_paths else i

        imgA = unnormalize(imgs_A[i], transform)
        imgB = unnormalize(imgs_B[i], transform)

        scoreA = scores_A[i]
        scoreB = scores_B[i]

        axes[row, 0].imshow(imgA)
        axes[row, 0].axis("off")
        axes[row, 0].text(
            0.02, 0.98,
            f"idx: {idx}\nA score={scoreA:.3f}",
            transform=axes[row, 0].transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

        axes[row, 1].imshow(imgB)
        axes[row, 1].axis("off")
        axes[row, 1].text(
            0.02, 0.98,
            f"idx: {idx}\nB score={scoreB:.3f}\nΔ={res.brittleness:.3f}",
            transform=axes[row, 1].transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

    save_path = directory / "brittleness_topk.png"
    plt.savefig(save_path)
    plt.close(fig)

    return save_path


def visualize_topk_plotly(
    results_sorted,
    imgs_A, imgs_B,
    scores_A, scores_B,
    K=10,
    class_names=None,
    transform=None,
    directory=Path(),
    image_paths=None
):
    topk = results_sorted[:K]

    fig = make_subplots(
        rows=K,
        cols=3,
        column_widths=[0.4, 0.2, 0.4],
        horizontal_spacing=0.05,
        vertical_spacing=0.05,
        specs=[[{"type": "image"}, {"type": "xy"}, {"type": "image"}] for _ in range(K)],
    )

    for r, res in enumerate(topk, start=1):
        i = res.index
        idx = Path(str(image_paths[i])).name if image_paths else i

        imgA = unnormalize(imgs_A[i], transform)
        imgB = unnormalize(imgs_B[i], transform)

        scoreA = scores_A[i]
        scoreB = scores_B[i]

        fig.add_trace(go.Image(z=imgA), row=r, col=1)

        fig.add_trace(
            go.Scatter(
                x=[0.5], y=[0.5],
                mode="text",
                text=[(
                    f"<b>idx:</b> {idx}<br><br>"
                    f"<b>A score:</b> {scoreA:.3f}<br>"
                    f"<b>B score:</b> {scoreB:.3f}<br>"
                    f"<b>Δ brittleness:</b> {res.brittleness:.3f}"
                )],
                showlegend=False
            ),
            row=r, col=2
        )

        fig.update_xaxes(visible=False, row=r, col=2)
        fig.update_yaxes(visible=False, row=r, col=2)

        fig.add_trace(go.Image(z=imgB), row=r, col=3)

    fig.update_layout(
        height=360 * K,
        showlegend=False,
        title="Top-K Most Brittle Images (Detection)",
        template="plotly_white"
    )

    save_path = directory / "brittleness_topk.html"
    fig.write_html(save_path, include_plotlyjs="cdn")

    return save_path

def visualize_in_html(
    results_sorted,
    imgsA, imgsB,
    scoresA, scoresB,
    labels,
    class_names=None,
    transform=None,
    directory=Path(),
    image_paths=None
):
    imageA_list = []
    imageB_list = []
    info_list = []

    for res in results_sorted:
        i = res.index
        idx = Path(str(image_paths[i])).name if image_paths else i

        imgA_b64 = "data:image/png;base64," + tensor_to_base64(imgsA[i], transform)
        imgB_b64 = "data:image/png;base64," + tensor_to_base64(imgsB[i], transform)

        scoreA = scoresA[i].item() if hasattr(scoresA[i], "item") else float(scoresA[i])
        scoreB = scoresB[i].item() if hasattr(scoresB[i], "item") else float(scoresB[i])

        gt = labels[i]  # KEEP AS DETECTION STRUCTURE

        imageA_list.append(f'"{imgA_b64}"')
        imageB_list.append(f'"{imgB_b64}"')

        info_list.append(
            f'"'
            f'<b>Index:</b> {idx}<br>'
            f'<b>Brittleness Δ:</b> {res.brittleness:.3f}<br><br>'
            f'<b>Score A:</b> {scoreA:.3f}<br>'
            f'<b>Score B:</b> {scoreB:.3f}<br><br>'
            f'<b>GT objects:</b> {len(gt)}'
            f'"'
        )

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <title>Brittleness Carousel</title>

    <style>
    body {{ font-family: Arial, sans-serif; }}
    #container {{ text-align: center; margin-top: 30px; }}
    .images {{ display: flex; justify-content: center; gap: 40px; }}
    img {{ max-width: 320px; max-height: 320px; border: 1px solid #ccc; }}
    button {{ padding: 10px 20px; font-size: 16px; margin: 10px; }}
    #info {{ margin-top: 15px; font-size: 16px; }}
    </style>
    </head>

    <body>
    <div id="container">

    <h2>Most Brittle Images (A → B)</h2>

    <div class="images">
    <div>
        <h3>Before Corruption</h3>
        <img id="imgA">
    </div>
    <div>
        <h3>After Corruption</h3>
        <img id="imgB">
    </div>
    </div>

    <div id="info"></div>

    <button onclick="prev()">⬅ Prev</button>
    <button onclick="next()">Next ➡</button>

    </div>

    <script>
    let imagesA = {json.dumps(imageA_list)};
    let imagesB = {json.dumps(imageB_list)};
    let infos   = {json.dumps(info_list)};

    let idx = 0;

    function show() {{
        document.getElementById("imgA").src = imagesA[idx];
        document.getElementById("imgB").src = imagesB[idx];
        document.getElementById("info").innerHTML =
            infos[idx] + "<br><br>" + (idx+1) + " / " + imagesA.length;
    }}

    function next() {{
        idx = (idx + 1) % imagesA.length;
        show();
    }}

    function prev() {{
        idx = (idx - 1 + imagesA.length) % imagesA.length;
        show();
    }}

    document.addEventListener("keydown", function(e) {{
        if (e.key === "ArrowRight") next();
        if (e.key === "ArrowLeft") prev();
    }});

    show();
    </script>

    </body>
    </html>
    """
    save_path = directory / "brittleness_carousel.html"
    with open(save_path, "w") as f:
        f.write(html)

    print("Saved brittleness_carousel.html")
    return save_path

# ==== HELPER FUNCTIONS ====

def unnormalize(img_tensor, transform=None):
    """
    Convert a possibly normalized image tensor to a displayable HWC NumPy array.

    Args:
        img_tensor (torch.Tensor): Image tensor of shape (C, H, W), possibly normalized.
        transform (torchvision.transforms, optional): The transform used during dataset
            preprocessing to extract mean and std for unnormalization.

    Returns:
        numpy.ndarray: Image array of shape (H, W, C) with values scaled to [0, 1]
        suitable for visualization.
    """
    stats = extract_normalize(transform)

    img = img_tensor.clone()

    if stats is not None:
        mean, std = stats
        mean = torch.tensor(mean).view(-1,1,1)
        std  = torch.tensor(std).view(-1,1,1)
        img = img * std + mean

    # Always make display-safe
    img = img - img.min()
    img = img / (img.max() + 1e-8)

    return img.permute(1,2,0).numpy()

def get_topk_predictions(probs, k=3):
    """
    Get the top-k predicted class indices and their probabilities.

    Args:
        probs (torch.Tensor): Tensor of predicted probabilities (1D or batch 2D).
        k (int, optional): Number of top predictions to return. Defaults to 3.

    Returns:
        List[Tuple[int, float]]: List of tuples containing (class_index, probability)
        for the top-k predictions.
    """
    vals, inds = probs.topk(k)
    return list(zip(inds.tolist(), vals.tolist()))

def get_between_columns_x(fig):
    """
    Compute the midpoint x-coordinate between the first two x-axes of a Plotly figure.

    Args:
        fig (plotly.graph_objs.Figure): Plotly figure object with at least two x-axes.

    Returns:
        float: Midpoint between the end of the first x-axis and the start of the second.
    """    
    x1 = fig.layout.xaxis.domain
    x2 = fig.layout.xaxis2.domain
    return 0.5 * (x1[1] + x2[0])

def tensor_to_base64(img_tensor, transform=None):
    """
    Convert a C,H,W image tensor to a base64-encoded PNG string.

    Args:
        img_tensor (torch.Tensor): Image tensor with shape (C, H, W), values in [0,1].
        transform (torchvision.transforms, optional): Transform used during preprocessing,
            used to unnormalize the tensor if needed.

    Returns:
        str: Base64-encoded PNG image suitable for embedding in HTML or JSON.
    """
    img = unnormalize(img_tensor, transform)#img_tensor.permute(1,2,0).numpy()
    img = (img * 255).astype(np.uint8)
    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def extract_normalize(transform):
    """
    Extract the mean and standard deviation from a torchvision Normalize transform.

    Args:
        transform (torchvision.transforms or None): Transform object to inspect.

    Returns:
        Tuple[List[float], List[float]] or None: Returns (mean, std) if a Normalize
        transform is present, else None.
    """
    if transform is None:
        return None

    if isinstance(transform, transforms.Normalize):
        return transform.mean, transform.std

    if isinstance(transform, transforms.Compose):
        for t in transform.transforms:
            if isinstance(t, transforms.Normalize):
                return t.mean, t.std

    return None

# ==== OTHER FUNCTIONS THAT ARE NOT USED FOR THIS WHOLE ALGO BUT I DON'T WANT TO DELETE THEM YET ====
