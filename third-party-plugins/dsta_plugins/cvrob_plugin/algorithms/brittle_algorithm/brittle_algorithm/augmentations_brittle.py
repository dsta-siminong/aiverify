import torch 
import numpy as np
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import plotly.graph_objects as go
# from augly.image import blur, brightness, random_noise, contrast, color_jitter, pixelization, sharpen
# from augly.image import aug_np_wrapper
from .cvrob_util import evaluate, collect_probs
from plotly.subplots import make_subplots
# from imagecorruptions import corrupt
import matplotlib.pyplot as plt
import json
import base64
from PIL import Image
import io
from pathlib import Path
from dataclasses import dataclass

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
    """
    Convert a BrittlenessResult object to a dictionary with JSON-serializable data.

    Args:
        br (BrittlenessResult): The brittleness results collection to convert.

    Returns:
        dict: Dictionary containing:
            - 'results': List of dictionaries for each individual result.
            - 'imgsA': List representation of imgsA tensor.
            - 'imgsB': List representation of imgsB tensor.
            - 'probs_A': List representation of probs_A tensor.
            - 'probs_B': List representation of probs_B tensor.
            - 'labels': List representation of labels tensor.
    """
    d = {
        "results": [brittle_res_indiv_to_dict(r) for r in br.results],
        "imgsA": br.imgsA.numpy().tolist(),
        "imgsB": br.imgsB.numpy().tolist(),
        "probs_A": br.probs_A.numpy().tolist(),
        "probs_B": br.probs_B.numpy().tolist(),
        "labels": br.labels.numpy().tolist(),
    }
    return d

def visualize_topk_matplotlib(
    results_sorted, 
    imgs_A, imgs_B, 
    probs_A, probs_B, 
    K=10,
    class_names=None, 
    transform=None,
    directory=Path(),
    image_paths=None
):

    topk = results_sorted[:K]
    top_k_indices = [item.index for item in topk]
    # print("&& TOPK INDICES 2", top_k_indices)

    fig, axes = plt.subplots(
        K, 2,
        figsize=(12, 3 * K),
        constrained_layout=True
    )
    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_visible(True)

    fragments_dir = directory / "fragments"
    fragments_dir.mkdir(parents=True, exist_ok=True)
    fragment_paths = []

    print(f"TRANSFORM: {transform}")
    for row, res in enumerate(topk):
        i = res.index
        if image_paths is not None:
            idx = Path(str(image_paths[i])).name
        else:
            idx = i

        imgA = unnormalize(imgs_A[i], transform)
        imgB = unnormalize(imgs_B[i], transform)

        # predA = get_topk_predictions(probs_A[i], k=1)[0]
        # predB = get_topk_predictions(probs_B[i], k=1)[0]
        # predA_int = predA[0]; initial_class_prob_A = predA[1]
        # predB_int = predB[0]; resultant_class_prob_B = predB[1]
        # classA = predA[0]
        # classB = predB[0]
        # if class_names is not None:
        #     classA = class_names[predA[0]]
        #     classB = class_names[predB[0]]
        # print('&'*16)
        # print(predA_int , res.predA)
        # print(predB_int , res.predB)
        # print(initial_class_prob_A , probs_A[i][predA_int].item())
        # print(resultant_class_prob_B, probs_B[i][predB_int].item())
        # print('&'*16)

        predA_int = res.predA
        predB_int = res.predB
        classA = predA_int
        classB = predB_int
        if class_names is not None:
            classA = class_names[predA_int]
            classB = class_names[predB_int]

        initial_class_prob_A = probs_A[i][predA_int].item()
        resultant_initial_class_prob_B = probs_B[i][predA_int].item()
        resultant_class_prob_B = probs_B[i][predB_int].item()
        # resultant_initial_class_prob_B = predA[1]-res.brittleness
        # resultant_initial_class_prob_B_0 = probs_B[i][predA_int].item()
        # print(resultant_initial_class_prob_B_0 , resultant_initial_class_prob_B)
        # assert resultant_initial_class_prob_B_0 == resultant_initial_class_prob_B

        axes[row, 0].imshow(imgA)
        axes[row, 0].axis("off")
        axes[row, 0].text(
            0.02, 0.98,
            f"image path: {idx}\nA (before) | pred={classA}\n\npred_proba of class {classA}={initial_class_prob_A:.3f}",
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
            f"image path: {idx}\nB (after) | pred={classB}\n"
            f"pred_proba of {classB}={resultant_class_prob_B:.3f}\n"
            f"pred_proba of {classA}={resultant_initial_class_prob_B:.3f} | Δ={res.brittleness:.3f}",
            transform=axes[row, 1].transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

        # Save this row as its own 1x2 figure
        fig_row, axes_row = plt.subplots(
            1, 2,
            figsize=(12, 3),
            constrained_layout=True
        )

        for ax in axes_row:
            for spine in ax.spines.values():
                spine.set_visible(True)

        axes_row[0].imshow(imgA)
        axes_row[0].axis("off")
        axes_row[0].text(
            0.02, 0.98,
            f"image path: {idx}\nA (before) | pred={classA}\n\npred_proba of class {classA}={initial_class_prob_A:.3f}",
            transform=axes_row[0].transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

        axes_row[1].imshow(imgB)
        axes_row[1].axis("off")
        axes_row[1].text(
            0.02, 0.98,
            f"image path: {idx}\nB (after) | pred={classB}\n"
            f"pred_proba of {classB}={resultant_class_prob_B:.3f}\n"
            f"pred_proba of {classA}={resultant_initial_class_prob_B:.3f} | Δ={res.brittleness:.3f}",
            transform=axes_row[1].transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

        fragment_path = fragments_dir / f"brittleness_top_{row + 1}.png"
        fragment_paths.append(fragment_path)
        fig_row.savefig(fragment_path)
        plt.close(fig_row)

    save_path = directory / f"brittleness_topk.png"
    plt.savefig(save_path)
    plt.close(fig)

    return save_path, fragment_paths

def visualize_topk_plotly(
    results_sorted, 
    imgs_A, imgs_B, 
    probs_A, probs_B, 
    K=10, 
    class_names=None, 
    transform=None,
    directory=Path(),
    image_paths=None
):
    def tensor_to_plotly_img(x, transform=None):
        """
        Convert C,H,W tensor in 0..1 -> H,W,C uint8 0..255
        """
        x = unnormalize(x, transform) #x.permute(1,2,0).numpy()  # H,W,C
        x = (x * 255).astype("uint8")
        return x
    print(f"TRANSFORM: {transform}")
    topk = results_sorted[:K]

    fig = make_subplots(
        rows=K,
        cols=3,
        column_widths=[0.35, 0.3, 0.35],  # third column narrower
        horizontal_spacing=0.05,
        vertical_spacing=0.05,
        specs=[[{"type": "image"}, {"type": "xy"}, {"type": "image"}] for _ in range(K)],
    )

    for r, res in enumerate(topk, start=1):
        i = res.index
        if image_paths is not None:
            idx = Path(str(image_paths[i])).name
        else:
            idx = i

        # predA = get_topk_predictions(probs_A[i], k=1)[0]
        # predB = get_topk_predictions(probs_B[i], k=1)[0]
        # predA_int = predA[0]; initial_class_prob_A = predA[1]
        # predB_int = predB[0]; resultant_class_prob_B = predB[1]
        # classA = predA[0]
        # classB = predB[0]
        # if class_names is not None:
        #     classA = class_names[predA[0]]
        #     classB = class_names[predB[0]]
        # print('&'*16)
        # print(predA_int , res.predA)
        # print(predB_int , res.predB)
        # print(initial_class_prob_A , probs_A[i][predA_int].item())
        # print(resultant_class_prob_B, probs_B[i][predB_int].item())
        # print('&'*16)

        predA_int = res.predA
        predB_int = res.predB
        classA = predA_int
        classB = predB_int
        if class_names is not None:
            classA = class_names[predA_int]
            classB = class_names[predB_int]

        initial_class_prob_A = probs_A[i][predA_int].item()
        resultant_initial_class_prob_B = probs_B[i][predA_int].item()
        resultant_class_prob_B = probs_B[i][predB_int].item()
        # resultant_initial_class_prob_B = predA[1]-res.brittleness
        # resultant_initial_class_prob_B_0 = probs_B[i][predA_int].item()
        # print(resultant_initial_class_prob_B_0 , resultant_initial_class_prob_B)
        # assert resultant_initial_class_prob_B_0 == resultant_initial_class_prob_B

        fig.add_trace(
            go.Image(z=imgA),
            row=r, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=[0.5],
                y=[0.5],
                mode="text",
                text=[(
                    f"<b>File path:</b> {idx} <br><br>"
                    f"<b>Prediction</b><br>"
                    f"A: ŷ={classA} | B: ŷ={classB} <br><br>"
                    f"<b>Before Corruption:</b><br>"
                    f"Predict Proba for Class {classA}: {initial_class_prob_A:.3f}<br><br>"
                    f"<b>After Corruption:</b><br>"
                    f"Predict Proba for Class {classA}:  {resultant_initial_class_prob_B:.3f}<br>"
                    f"Δ={res.brittleness:.3f}<br>"
                    f"Predict_Proba for Class {classB}: {resultant_class_prob_B:.3f}"
                )],
                textposition="middle center",
                textfont=dict(size=12),
                showlegend=False
            ),
            row=r,
            col=2
        )
        fig.update_xaxes(range=[0, 1], visible=False, row=r, col=2)
        fig.update_yaxes(range=[0, 1], visible=False, row=r, col=2)
        fig.add_trace(
            go.Image(z=imgB),
            row=r, col=3
        )

    fig.update_xaxes(showticklabels=False, visible=False)
    fig.update_yaxes(showticklabels=False, visible=False)
    fig.update_layout(
        height=360 * K,
        showlegend=False,
        title_text="Top-K Most Brittle Images (A → B)",
        margin=dict(l=0, r=0, t=60, b=0),
        autosize=True,
        template="plotly_white"
    )
    save_path = directory / "brittleness_top_k.html"

    fig.write_html(
        save_path,
        full_html=True,
        include_plotlyjs="inline",
        config={"responsive": True}
    )
    return save_path

def visualize_in_html(
    results_sorted, 
    imgs_A, imgs_B, 
    probsA, probsB, 
    labels, 
    class_names=None, 
    transform=None,
    directory=Path(),
    image_paths=None
):
    # Prepare image & info lists
    img_base64_list = []
    imageA_list = []
    imageB_list = []
    info_list = []

    for res in results_sorted:
        i = res.index
        if image_paths is not None:
            idx = Path(str(image_paths[i])).name
        else:
            idx = i
        # images
        imgA_b64 = "data:image/png;base64," + tensor_to_base64(imgs_A[i], transform, jpeg_quality=85)
        imgB_b64 = "data:image/png;base64," + tensor_to_base64(imgs_B[i], transform, jpeg_quality=85)

        # predictions
        predA_cls = probsA[i].argmax().item()
        predB_cls = probsB[i].argmax().item()

        predA_p = probsA[i, predA_cls].item()
        predB_p = probsB[i, predB_cls].item()

        gt = labels[i].item()

        imageA_list.append(imgA_b64)
        imageB_list.append(imgB_b64)

        if class_names is not None:
            predA_cls = class_names[predA_cls]
            predB_cls = class_names[predB_cls]
            gt = class_names[gt]

        bar_pct = int(res.brittleness * 100)
        bar_color = (
            "#2ecc71" if res.brittleness < 0.25 else
            "#f1c40f" if res.brittleness < 0.5  else
            "#e67e22" if res.brittleness < 0.75 else
            "#e74c3c"
        )

        info_list.append(
            # Ground truth
            f'<b>Image:</b> {idx} <br>'
            f'<b>Ground Truth class:</b> {gt}<br><br>'
            # Before
            f'<span style="color:#1a6fbb"><b>▶ Before corruption</b></span><br>'
            f'&nbsp;&nbsp;Prediction: <b>{predA_cls}</b> &nbsp;|&nbsp; Confidence of {predA_cls} <b>{predA_p:.3f}</b><br><br>'
            # After
            f'<span style="color:#c0392b"><b>▶ After corruption</b></span><br>'
            f'&nbsp;&nbsp;Prediction: <b>{predB_cls}</b> &nbsp;|&nbsp; Confidence of {predB_cls}: <b>{predB_p:.3f}</b><br>'
            f'&nbsp;&nbsp;Confidence of {predA_cls} <b>{(predA_p - res.brittleness):.3f}</b><br><br>'
            # Brittleness score + bar
            f'<b>Brittleness Δ: {res.brittleness:.2f} / 1.00</b><br>'
            f'<div style="background:#ddd;border-radius:4px;height:10px;width:300px;display:inline-block;margin:4px 0">'
            f'<div style="background:{bar_color};width:{bar_pct}%;height:10px;border-radius:4px"></div></div><br>'
            f'<small style="color:#888">'
            f'Measures how much the model\'s best detection degraded after corruption.<br>'
            f'0 = no change &nbsp;·&nbsp; 1 = detection confidently mispredicted'
            f'</small>'
        )

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <title>Brittleness Carousel</title>
    <style>
    body {{ font-family: Arial, sans-serif; background: #f9f9f9; }}
    #container {{ text-align: center; margin-top: 30px; }}
    .images {{ display: flex; justify-content: center; gap: 40px; flex-wrap: wrap; }}
    .img-panel {{ display: flex; flex-direction: column; align-items: center; }}
    .img-panel h3 {{ margin-bottom: 6px; }}
    img {{ max-width: 320px; max-height: 320px; border: 2px solid #ccc; border-radius: 4px; }}
    button {{ padding: 10px 24px; font-size: 16px; margin: 10px; border-radius: 4px;
                border: none; background: #333; color: #fff; cursor: pointer; }}
    button:hover {{ background: #555; }}
    #info {{ margin: 16px auto; max-width: 620px; text-align: left;
            background: #fff; border: 1px solid #ddd; border-radius: 6px;
            padding: 16px 20px; font-size: 14px; line-height: 1.7; }}
    #counter {{ font-size: 13px; color: #888; margin-top: 6px; }}
    </style>
    </head>
    <body>
    <div id="container">
    <h2>Most Brittle Images (Before → After Corruption)</h2>
    <div class="images">
        <div class="img-panel">
        <h3 style="color:#1a6fbb">Before Corruption</h3>
        <img id="imgA">
        </div>
        <div class="img-panel">
        <h3 style="color:#c0392b">After Corruption</h3>
        <img id="imgB">
        </div>
    </div>
    <div id="info"></div>
    <div id="counter"></div>
    <br>
    <button onclick="prev()">⬅ Prev</button>
    <button onclick="next()">Next ➡</button>
    </div>

    <script>
    let imagesA = {json.dumps(imageA_list)};
    let imagesB = {json.dumps(imageB_list)};
    let infos = {json.dumps(info_list)};

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

# def get_topk_predictions(probs, k=3):
#     """
#     Get the top-k predicted class indices and their probabilities.

#     Args:
#         probs (torch.Tensor): Tensor of predicted probabilities (1D or batch 2D).
#         k (int, optional): Number of top predictions to return. Defaults to 3.

#     Returns:
#         List[Tuple[int, float]]: List of tuples containing (class_index, probability)
#         for the top-k predictions.
#     """
#     vals, inds = probs.topk(k)
#     return list(zip(inds.tolist(), vals.tolist()))

def tensor_to_base64(img_tensor, transform=None, max_size: int = None, jpeg_quality: int = None):
    """
    Convert a C,H,W image tensor to a base64-encoded image string.

    Args:
        img_tensor (torch.Tensor): Image tensor with shape (C, H, W), values in [0,1].
        transform: Optional preprocessing transform, used to unnormalize if needed.
        max_size (int, optional): Downscale so the longest edge is at most this many pixels.
        jpeg_quality (int, optional): If set (1-95), encode as JPEG. Otherwise PNG.

    Returns:
        str: Base64-encoded image. Caller prepends the data URI prefix.
    """
    img = unnormalize(img_tensor, transform)
    img = (img * 255).astype(np.uint8)
    pil_img = Image.fromarray(img)

    if max_size is not None:
        w, h = pil_img.size
        scale = max_size / max(w, h)
        if scale < 1.0:
            pil_img = pil_img.resize(
                (int(w * scale), int(h * scale)),
                Image.LANCZOS
            )

    buffer = io.BytesIO()
    if jpeg_quality is not None:
        pil_img.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    else:
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

def visualize_topk_without_plotly(
    results_sorted,
    imgs_A, imgs_B,
    probs_A, probs_B,
    K=10,
    class_names=None,
    transform=None,
    directory=Path(),
    image_paths=None
):
    topk = results_sorted[:K]

    rows_html = []

    for rank, res in enumerate(topk, start=1):
        i = res.index

        if image_paths is not None:
            idx = Path(str(image_paths[i])).name
        else:
            idx = i

        imgA_b64 = tensor_to_base64(
            imgs_A[i],
            transform,
            jpeg_quality=85
        )
        imgB_b64 = tensor_to_base64(
            imgs_B[i],
            transform,
            jpeg_quality=85
        )

        # predA = get_topk_predictions(probs_A[i], k=1)[0]
        # predB = get_topk_predictions(probs_B[i], k=1)[0]
        # predA_int = predA[0]; initial_class_prob_A = predA[1]
        # predB_int = predB[0]; resultant_class_prob_B = predB[1]
        # classA = predA[0]
        # classB = predB[0]
        # if class_names is not None:
        #     classA = class_names[predA[0]]
        #     classB = class_names[predB[0]]
        # print('&'*16)
        # print(predA_int , res.predA)
        # print(predB_int , res.predB)
        # print(initial_class_prob_A , probs_A[i][predA_int].item())
        # print(resultant_class_prob_B, probs_B[i][predB_int].item())
        # print('&'*16)

        predA_int = res.predA
        predB_int = res.predB
        classA = predA_int
        classB = predB_int
        if class_names is not None:
            classA = class_names[predA_int]
            classB = class_names[predB_int]

        initial_class_prob_A = probs_A[i][predA_int].item()
        resultant_initial_class_prob_B = probs_B[i][predA_int].item()
        resultant_class_prob_B = probs_B[i][predB_int].item()
        # resultant_initial_class_prob_B = predA[1]-res.brittleness
        # resultant_initial_class_prob_B_0 = probs_B[i][predA_int].item()
        # print(resultant_initial_class_prob_B_0 , resultant_initial_class_prob_B)
        # assert resultant_initial_class_prob_B_0 == resultant_initial_class_prob_B

        classA_html = f'<span class="classA">{classA}</span>'
        classB_html = f'<span class="classB">{classB}</span>'

        if classA == classB:
            after_classB_line = ""
        else:
            after_classB_line = (f"P({classB_html}) = {resultant_class_prob_B:.3f}")

        info = f"""
        <div class="info">
            <h3>#{rank} Most Brittle Image</h3>

            <p><b>Image:</b> {idx}</p>

            <p>
                <b>Prediction:</b><br>
                Before: {classA_html}<br>
                After: {classB_html}
            </p>

            <p>
                <b>Before Corruption</b><br>
                P({classA_html}) = {initial_class_prob_A:.3f}
            </p>

            <p>
                <b>After Corruption</b><br>
                P({classA_html}) = {resultant_initial_class_prob_B:.3f}<br>
                Δ = {res.brittleness:.3f}<br>
                {after_classB_line}
            </p>
        </div>
        """

        rows_html.append(f"""
        <div class="row">
            <div class="image-container">
                <img src="data:image/png;base64,{imgA_b64}">
                <div class="caption before-caption">Before</div>
            </div>

            {info}

            <div class="image-container">
                <img src="data:image/png;base64,{imgB_b64}">
                <div class="caption after-caption">After</div>
            </div>
        </div>
        """)

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">

<style>

.before-caption {{
    color: blue;
}}

.after-caption {{
    color: orange;
}}

.classA {{
    color: blue;
    font-weight: bold;
}}

.classB {{
    color: orange;
    font-weight: bold;
}}

body {{
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 20px;
    background: #fafafa;
}}

.container {{
    display: flex;
    flex-direction: column;
    gap: 30px;
}}

.row {{
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(250px, 0.8fr) minmax(0, 1fr);
    gap: 20px;

    background: white;
    padding: 20px;
    border-radius: 12px;

    box-shadow: 0 2px 8px rgba(0,0,0,0.08);

    align-items: center;
}}

.image-container {{
    text-align: center;
}}

.image-container img {{
    width: 100%;
    max-width: 400px;
    height: auto;
    border-radius: 8px;
}}

.caption {{
    margin-top: 8px;
    font-weight: bold;
    text-align: center;
}}

.info {{
    min-width: 0;
    overflow-wrap: anywhere;
    word-break: break-word;
    line-height: 1.5;
    text-align: center;
}}

.info h3 {{
    text-align: center;
    text-decoration: underline;
}}

@media (max-width: 900px) {{
    .row {{
        grid-template-columns: 1fr;
    }}
}}

</style>

</head>

<body>

<div class="container">
{''.join(rows_html)}
</div>

</body>
</html>
"""

    save_path = directory / "brittleness_topk.html"

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html)

    return save_path