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
    """
    def serialize_detection_indiv(pred):
        return {
            "boxes": pred["boxes"].tolist(),
            "labels": pred["labels"].tolist(),
            "scores": pred["scores"].tolist(),
        }

    d = {
        "index": bri.index,
        "label": bri.label,
        "predA": serialize_detection_indiv(bri.predA),
        "predB": serialize_detection_indiv(bri.predB),
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
            return val

    def serialize_detection(preds):
        out = []
        for pred in preds:
            out.append({
                "boxes": pred["boxes"].tolist(),
                "labels": pred["labels"].tolist(),
                "scores": pred["scores"].tolist(),
            })
        return out

    d = {
        "results": [brittle_res_indiv_to_dict(r) for r in br.results],
        "imgsA": None,
        "imgsB": None,
        "probs_A": serialize_detection(br.probs_A),
        "probs_B": serialize_detection(br.probs_B),
        "labels": safe_convert(br.labels),
    }
    return d


# ==== BRITTLENESS HELPERS ====

def normalise_brittleness(raw: float, scoreA: float) -> float:
    """
    Normalise a raw image_brittleness value to [0, 1].

    image_brittleness returns an unbounded penalty for the worst-degraded detection:
      - No match found:   drop = scoreA                          → max = scoreA
      - Match found:      drop = max(0, scoreA-scoreB) + scoreA*(1-IoU)
                                                                  → max = 2*scoreA

    Dividing by 2*scoreA maps the range to [0, 1], where:
      0   = no degradation at all (scoreA==scoreB and IoU==1)
      0.5 = either full confidence drop with perfect localisation,
            or perfect confidence with zero overlap
      1.0 = full confidence drop AND zero overlap (worst possible)

    Args:
        raw (float): Output of image_brittleness().
        scoreA (float): Top detection confidence score before corruption (scoreA of the
                        worst-affected detection). Use scores_A[i]["scores"].max().

    Returns:
        float: Normalised brittleness in [0, 1]. Returns 0.0 if scoreA == 0.
    """
    if scoreA <= 0.0:
        return 0.0
    return min(raw / (2.0 * scoreA), 1.0)


def brittleness_label(norm: float) -> str:
    """Return a short plain-English severity label for a normalised brittleness value."""
    if norm < 0.25:
        return "Stable"
    elif norm < 0.5:
        return "Mildly brittle"
    elif norm < 0.75:
        return "Moderately brittle"
    else:
        return "Highly brittle"


def _detection_summary(scores_dict) -> tuple:
    """
    Return (n_detections, top_confidence) from a detection output dict.
    scores_dict is {"boxes":..., "labels":..., "scores": Tensor}.
    """
    s = scores_dict["scores"]
    n = len(s)
    top = float(s.max().item()) if n > 0 else 0.0
    return n, top


# ==== VISUALISATION FUNCTIONS ====

def visualize_topk_matplotlib(
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

    fig, axes = plt.subplots(K, 2, figsize=(12, 3 * K), constrained_layout=True)

    if K == 1:
        axes = np.array([axes])

    for row, res in enumerate(topk):
        i = res.index
        idx = Path(str(image_paths[i])).name if image_paths else i

        imgA = unnormalize(imgs_A[i], transform)
        imgB = unnormalize(imgs_B[i], transform)

        n_A, top_A = _detection_summary(scores_A[i])
        n_B, top_B = _detection_summary(scores_B[i])

        raw_brit = res.brittleness
        norm_brit = normalise_brittleness(raw_brit, top_A)
        label = brittleness_label(norm_brit)

        det_delta = n_B - n_A
        det_delta_str = (
            f"+{det_delta}" if det_delta > 0
            else str(det_delta) if det_delta < 0
            else "±0"
        )

        axes[row, 0].imshow(imgA)
        axes[row, 0].axis("off")
        axes[row, 0].text(
            0.02, 0.98,
            (
                f"image: {idx}\n"
                f"BEFORE corruption\n"
                f"Detections: {n_A}  |  Top conf: {top_A:.2f}"
            ),
            transform=axes[row, 0].transAxes,
            va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

        axes[row, 1].imshow(imgB)
        axes[row, 1].axis("off")
        axes[row, 1].text(
            0.02, 0.98,
            (
                f"image: {idx}\n"
                f"AFTER corruption\n"
                f"Detections: {n_B} ({det_delta_str})  |  Top conf: {top_B:.2f}\n"
                f"Brittleness: {norm_brit:.2f} / 1.00  →  {label}\n"
                f"(raw penalty: {raw_brit:.3f})"
            ),
            transform=axes[row, 1].transAxes,
            va="top", ha="left", fontsize=9,
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

        # Encode as JPEG (10-30x smaller than raw z= array)
        imgA_b64 = "data:image/jpeg;base64," + tensor_to_base64(imgs_A[i], transform, jpeg_quality=85)
        imgB_b64 = "data:image/jpeg;base64," + tensor_to_base64(imgs_B[i], transform, jpeg_quality=85)

        n_A, top_A = _detection_summary(scores_A[i])
        n_B, top_B = _detection_summary(scores_B[i])

        raw_brit = res.brittleness
        norm_brit = normalise_brittleness(raw_brit, top_A)
        label = brittleness_label(norm_brit)

        det_delta = n_B - n_A
        det_delta_str = (
            f"+{det_delta} more" if det_delta > 0
            else f"{abs(det_delta)} fewer" if det_delta < 0
            else "no change"
        )

        fig.add_trace(go.Image(source=imgA_b64), row=r, col=1)

        fig.add_trace(
            go.Scatter(
                x=[0.5], y=[0.5],
                mode="text",
                text=[(
                    f"<b>{idx}</b><br><br>"
                    f"<b>Before</b><br>"
                    f"Detections: {n_A}<br>"
                    f"Top conf: {top_A:.2f}<br><br>"
                    f"<b>After</b><br>"
                    f"Detections: {n_B} ({det_delta_str})<br>"
                    f"Top conf: {top_B:.2f}<br><br>"
                    f"<b>Brittleness</b><br>"
                    f"{norm_brit:.2f} / 1.00<br>"
                    f"<i>{label}</i><br>"
                    f"<span style='color:grey;font-size:0.85em'>"
                    f"raw: {raw_brit:.3f}</span>"
                )],
                showlegend=False
            ),
            row=r, col=2
        )

        fig.update_xaxes(visible=False, row=r, col=2)
        fig.update_yaxes(visible=False, row=r, col=2)

        fig.add_trace(go.Image(source=imgB_b64), row=r, col=3)

    fig.update_layout(
        height=360 * K,
        showlegend=False,
        title="Top-K Most Brittle Images (Detection)",
        template="plotly_white"
    )

    save_path = directory / "brittleness_topk.html"
    fig.write_html(save_path, include_plotlyjs="inline")

    return save_path


def visualize_in_html(
    results_sorted,
    imgsA, imgsB,
    scoresA, scoresB,
    labels,
    class_names=None,
    transform=None,
    directory=Path(),
    image_paths=None,
    max_size: int = 320,
    jpeg_quality: int = 85,
):
    """
    Build a side-by-side carousel HTML of the most brittle images.

    max_size: longest edge in pixels each image is resized to before encoding.
    jpeg_quality: JPEG quality 1-95. Lower = smaller file.
                  220 images at 320px / q=85 ≈ 3-7 MB total.
    """
    imageA_list = []
    imageB_list = []
    info_list = []

    fmt = "jpeg" if jpeg_quality is not None else "png"
    mime = f"data:image/{fmt};base64,"

    for res in results_sorted:
        i = res.index
        idx = Path(str(image_paths[i])).name if image_paths else i

        imgA_b64 = mime + tensor_to_base64(imgsA[i], transform, max_size=max_size, jpeg_quality=jpeg_quality)
        imgB_b64 = mime + tensor_to_base64(imgsB[i], transform, max_size=max_size, jpeg_quality=jpeg_quality)

        n_A, top_A = _detection_summary(scoresA[i])
        n_B, top_B = _detection_summary(scoresB[i])

        gt = labels[i]
        n_gt = len(gt)

        raw_brit = res.brittleness
        norm_brit = normalise_brittleness(raw_brit, top_A)
        label = brittleness_label(norm_brit)

        det_delta = n_B - n_A
        det_delta_str = (
            f"+{det_delta} more" if det_delta > 0
            else f"{abs(det_delta)} fewer" if det_delta < 0
            else "same number"
        )

        # Normalised brittleness bar: filled portion as a percentage
        bar_pct = int(norm_brit * 100)
        bar_color = (
            "#2ecc71" if norm_brit < 0.25 else
            "#f1c40f" if norm_brit < 0.5  else
            "#e67e22" if norm_brit < 0.75 else
            "#e74c3c"
        )

        imageA_list.append(imgA_b64)
        imageB_list.append(imgB_b64)

        info_list.append(
            # Ground truth
            f'<b>Image:</b> {idx} &nbsp;|&nbsp; <b>Ground truth objects:</b> {n_gt}<br><br>'
            # Before
            f'<span style="color:#1a6fbb"><b>▶ Before corruption</b></span><br>'
            f'&nbsp;&nbsp;Detections: <b>{n_A}</b> &nbsp;|&nbsp; Top confidence: <b>{top_A:.2f}</b><br><br>'
            # After
            f'<span style="color:#c0392b"><b>▶ After corruption</b></span><br>'
            f'&nbsp;&nbsp;Detections: <b>{n_B}</b> ({det_delta_str}) &nbsp;|&nbsp; Top confidence: <b>{top_B:.2f}</b><br><br>'
            # Brittleness score + bar
            f'<b>Brittleness: {norm_brit:.2f} / 1.00 — {label}</b><br>'
            f'<div style="background:#ddd;border-radius:4px;height:10px;width:300px;display:inline-block;margin:4px 0">'
            f'<div style="background:{bar_color};width:{bar_pct}%;height:10px;border-radius:4px"></div></div><br>'
            f'<small style="color:#888">'
            f'Measures how much the model\'s best detection degraded after corruption.<br>'
            f'0 = no change &nbsp;·&nbsp; 1 = detection fully lost or mislocalised<br>'
            f'(raw penalty: {raw_brit:.3f})'
            f'</small>'
        )

    html = f"""<!DOCTYPE html>
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
let infos   = {json.dumps(info_list)};
let idx = 0;

function show() {{
    document.getElementById("imgA").src = imagesA[idx];
    document.getElementById("imgB").src = imagesB[idx];
    document.getElementById("info").innerHTML = infos[idx];
    document.getElementById("counter").textContent = (idx+1) + " / " + imagesA.length;
}}
function next() {{ idx = (idx + 1) % imagesA.length; show(); }}
function prev() {{ idx = (idx - 1 + imagesA.length) % imagesA.length; show(); }}
document.addEventListener("keydown", function(e) {{
    if (e.key === "ArrowRight") next();
    if (e.key === "ArrowLeft")  prev();
}});
show();
</script>
</body>
</html>"""

    save_path = directory / "brittleness_carousel.html"
    with open(save_path, "w") as f:
        f.write(html)

    print("Saved brittleness_carousel.html")
    return save_path


# ==== HELPER FUNCTIONS ====

def unnormalize(img_tensor, transform=None):
    """
    Convert a possibly normalized image tensor to a displayable HWC NumPy array.
    """
    stats = extract_normalize(transform)
    img = img_tensor.clone()

    if stats is not None:
        mean, std = stats
        mean = torch.tensor(mean).view(-1, 1, 1)
        std  = torch.tensor(std).view(-1, 1, 1)
        img  = img * std + mean

    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return img.permute(1, 2, 0).numpy()


def get_topk_predictions(probs, k=3):
    """Get the top-k predicted class indices and their probabilities."""
    vals, inds = probs.topk(k)
    return list(zip(inds.tolist(), vals.tolist()))


def get_between_columns_x(fig):
    """Compute the midpoint x-coordinate between the first two x-axes of a Plotly figure."""
    x1 = fig.layout.xaxis.domain
    x2 = fig.layout.xaxis2.domain
    return 0.5 * (x1[1] + x2[0])


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
    """Extract mean and std from a torchvision Normalize transform, if present."""
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