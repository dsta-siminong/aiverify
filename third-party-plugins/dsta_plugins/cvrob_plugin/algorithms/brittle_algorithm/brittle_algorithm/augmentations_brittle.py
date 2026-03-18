import torch 
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import plotly.graph_objects as go
from augly.image import blur, brightness, random_noise, contrast, color_jitter, pixelization, sharpen
from augly.image import aug_np_wrapper
from .cvrob_util import (plot_accuracy_vs_severity,
                        evaluate,
                        best_fit_gradient,
                        collect_probs)
from plotly.subplots import make_subplots
from imagecorruptions import corrupt
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import base64
from PIL import Image
import io
from pathlib import Path
from dataclasses import dataclass

@dataclass
class BrittlenessResultIndiv:
    index: int
    label: int
    predA: int
    predB: int
    pA: float
    pB: float
    brittleness: float

@dataclass
class BrittlenessResult:
    results: list
    imgsA: list
    imgsB: list 
    probs_A: list
    probs_B: list 
    labels: list

def brittle_res_indiv_to_dict(bri):
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
    d = {
        "results": [brittle_res_indiv_to_dict(r) for r in br.results],
        "imgsA": br.imgsA.numpy().tolist(),
        "imgsB": br.imgsB.numpy().tolist(),
        "probs_A": br.probs_A.numpy().tolist(),
        "probs_B": br.probs_B.numpy().tolist(),
        "labels": br.labels.numpy().tolist(),
    }
    return d

# def brittle_method_simple(
#     model, 
#     test_loader, 
#     device, 
#     corr_func, 
#     augmentation_list,
#     augmentation_str,
#     transform=None,
#     augmentation_method=None,
#     severities=(0,1),
#     # top_proportion=0.05,
#     class_names=None
# ):  
#     # _, y_preds, _ = evaluate(model, test_loader, device)
#     corr_kwargs = None
#     for k, (m,d) in zip(augmentation_str, augmentation_list):
#         if k == augmentation_method:
#             d['aug_method'] = m
#             corr_kwargs = d 
#     if corr_kwargs == None:
#         raise ValueError("aug method not valid")

#     if severities[0] == 0:
#         loader_A = test_loader
#     else: 
#         loader_A = get_corrupted_dataloader(test_loader, 
#                                             corr_func, 
#                                             severity=severities[0],
#                                             corr_kwargs=corr_kwargs)
#     loader_B = get_corrupted_dataloader(test_loader, 
#                                         corr_func, 
#                                         severity=severities[1],
#                                         corr_kwargs=corr_kwargs)

#     imgs_A, probs_A, labels = collect_probs(model, loader_A, device)
#     imgs_B, probs_B, _      = collect_probs(model, loader_B, device)

#     N = len(labels)
#     idx = torch.arange(N)

#     pA = probs_A[idx, labels]
#     pB = probs_B[idx, labels]

#     brittleness = pA - pB

#     results_all = [
#         BrittlenessResultIndiv(
#             index=i,
#             label=int(labels[i]),
#             predA=probs_A[i].argmax().item(),
#             predB=probs_B[i].argmax().item(),
#             pA=float(pA[i]),
#             pB=float(pB[i]),
#             brittleness=float(brittleness[i]),
#         ) for i in range(N)
#     ]
#     # Sort (most brittle first)
#     results_all_sorted = sorted(results_all, key=lambda x: x.brittleness, reverse=True)

#     b_result = BrittlenessResult(
#         results = results_all_sorted,
#         imgsA = imgs_A, 
#         imgsB = imgs_B, 
#         probs_A = probs_A,
#         probs_B = probs_B,
#         labels = labels
#     )

#     return b_result

def unnormalize(img_tensor, transform=None):
    """
    img_tensor: C,H,W tensor (possibly normalized)
    transform: torchvision transform used on the dataset

    Returns: H,W,C numpy array suitable for visualization
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
    vals, inds = probs.topk(k)
    return list(zip(inds.tolist(), vals.tolist()))

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
    fig, axes = plt.subplots(
        K, 2,
        figsize=(12, 3 * K),
        constrained_layout=True
    )
    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_visible(True)

    print(f"TRANSFORM: {transform}")
    for row, res in enumerate(topk):
        i = res.index
        if image_paths is not None:
            idx = Path(str(image_paths[i])).name
        else:
            idx = i

        imgA = unnormalize(imgs_A[i], transform)
        imgB = unnormalize(imgs_B[i], transform)

        predA = get_topk_predictions(probs_A[i], k=1)[0]
        predB = get_topk_predictions(probs_B[i], k=1)[0]

        classA = predA[0]
        classB = predB[0]
        if class_names is not None:
            classA = class_names[predA[0]]
            classB = class_names[predB[0]]

        # axes[row, 0].imshow(imgA)
        # axes[row, 0].set_title(
        #     f"image idx: {idx}\n"
        #     f"A (before) | pred={classA}\n"
        #     f"\n"
        #     f"pred_proba of classA={predA[1]:.3f}"
        # )
        # axes[row, 0].axis("off")

        axes[row, 0].imshow(imgA)
        axes[row, 0].axis("off")
        axes[row, 0].text(
            0.02, 0.98,
            f"image idx: {idx}\nA (before) | pred={classA}\n\npred_proba of classA={predA[1]:.3f}",
            transform=axes[row, 0].transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )
        
        # plt.tight_layout()

        axes[row, 1].imshow(imgB)
        axes[row, 1].axis("off")
        axes[row, 1].text(
            0.02, 0.98,
            f"image idx: {idx}\nB (after) | pred={classB}\npred_proba of classB={predB[1]:.3f}\npred_proba of classA={(predA[1]-res.brittleness):.3f} | Δ={res.brittleness:.3f}",
            transform=axes[row, 1].transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

        # axes[row, 1].imshow(imgB)
        # axes[row, 1].set_title(
        #     f"image idx: {idx}\n"
        #     f"B (after) | pred={classB}\n"
        #     f"pred_proba of classB={predB[1]:.3f}\n"
        #     f"pred_proba of classA={(predA[1]-res.brittleness):.3f} | Δ={res.brittleness:.3f}"
        # )
        # axes[row, 1].axis("off")
        # plt.tight_layout()

    # plt.tight_layout()
    save_path = directory / "brittleness_top_k_gaussianblur.png"
    plt.savefig(save_path)
    print("SAVING FIGURE")
    return save_path
    #plt.show(block=False)  # show without blocking
    #input("Press Enter to close the figure and continue...")  # optional, keeps it open

def get_between_columns_x(fig):
    x1 = fig.layout.xaxis.domain
    x2 = fig.layout.xaxis2.domain
    return 0.5 * (x1[1] + x2[0])

def get_row_center_y(fig, r):
    # Left column y-axis for row r
    axis_index = 2 * r - 1
    yaxis = getattr(fig.layout, "yaxis" if axis_index == 1 else f"yaxis{axis_index}")
    y0, y1 = yaxis.domain
    return 0.5 * (y0 + y1)

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
        column_widths=[0.4, 0.2, 0.4],  # third column narrower
        horizontal_spacing=0.05,
        vertical_spacing=0.05,
        specs=[[{"type": "image"}, {"type": "xy"}, {"type": "image"}] for _ in range(K)],
    )
    # between_x = get_between_columns_x(fig)

    for r, res in enumerate(topk, start=1):
        i = res.index
        if image_paths is not None:
            idx = Path(str(image_paths[i])).name
        else:
            idx = i
        # row_center_y = get_row_center_y(fig, r)
        imgA = tensor_to_plotly_img(imgs_A[i], transform)
        imgB = tensor_to_plotly_img(imgs_B[i], transform)

        predA = get_topk_predictions(probs_A[i], k=1)[0]
        predB = get_topk_predictions(probs_B[i], k=1)[0]
        classA = predA[0]
        classB = predB[0]
        if class_names is not None:
            classA = class_names[predA[0]]
            classB = class_names[predB[0]]

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
                    f"<b>idx:</b> {idx} <br><br>"
                    f"<b>Prediction</b><br>"
                    f"A: ŷ={classA} | B: ŷ={classB} <br>"
                    f"<b>Predict_Proba Class A</b><br>"
                    f"A: {predA[1]:.3f} | B: {(predA[1]-res.brittleness):.3f}<br>"
                    f"Δ={res.brittleness:.3f}<br>"
                    f"<b>Predict_Proba Class B</b><br>"
                    f"B: {predB[1]:.3f}"
                )],
                textposition="middle center",
                textfont=dict(size=16),
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
    save_path = directory / "brittleness_top_k_gaussianblur.html"

    fig.write_html(
        save_path,
        full_html=True,
        include_plotlyjs="cdn",
        config={"responsive": True}
    )
    return save_path

def tensor_to_base64(img_tensor, transform=None):
    """
    Convert C,H,W tensor in [0,1] to base64 PNG string
    """
    img = unnormalize(img_tensor, transform)#img_tensor.permute(1,2,0).numpy()
    img = (img * 255).astype(np.uint8)
    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def visualize_in_html(
    results_sorted, 
    imgsA, imgsB, 
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
        imgA_b64 = "data:image/png;base64," + tensor_to_base64(imgsA[i], transform)
        imgB_b64 = "data:image/png;base64," + tensor_to_base64(imgsB[i], transform)

        # predictions
        predA_cls = probsA[i].argmax().item()
        predB_cls = probsB[i].argmax().item()

        predA_p = probsA[i, predA_cls].item()
        predB_p = probsB[i, predB_cls].item()

        gt = labels[i].item()

        imageA_list.append(f'"{imgA_b64}"')
        imageB_list.append(f'"{imgB_b64}"')

        if class_names is not None:
            predA_cls = class_names[predA_cls]
            predB_cls = class_names[predB_cls]
            gt = class_names[gt]

        info_list.append(
            f'"'
            f'<b>Index:</b> {idx}<br>'
            f'<b>Ground Truth class:</b> {gt}<br><br>'
            f'<b>Predictions:</b><br>'
            f'<b>A:</b> ŷ={predA_cls} | <b>B:</b> ŷ={predB_cls}<br><br>'
            f'<b>Class A predict_proba:</b><br>'
            f'<b>A:</b> {predA_p:.3f} | <b>B:</b> {(predA_p - res.brittleness):.3f}<br>'
            f'<b>Brittleness Δ:</b> {res.brittleness:.3f}<br><br>'
            f'<b>Class B predict_proba:</b><br>'
            f'<b>B:</b> {predB_p:.3f}'
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
        <h3>A (Before)</h3>
        <img id="imgA">
    </div>
    <div>
        <h3>B (After)</h3>
        <img id="imgB">
    </div>
    </div>

    <div id="info"></div>

    <button onclick="prev()">⬅ Prev</button>
    <button onclick="next()">Next ➡</button>

    </div>

    <script>
    let imagesA = [{",".join(imageA_list)}];
    let imagesB = [{",".join(imageB_list)}];
    let infos   = [{",".join(info_list)}];

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

    # # Insert into HTML
    # html_filled = html.replace("IMAGES_LIST", ",".join(img_base64_list))
    # html_filled = html_filled.replace("INFOS_LIST", ",".join(info_list))

    # # Save
    # with open("brittle_images_carousel.html", "w") as f:
    #     f.write(html_filled)

    # print("Saved HTML file: brittle_images_carousel.html")

def extract_normalize(transform):
    """
    Returns (mean, std) if a Normalize transform exists, else None.
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

def brittle_method(
    model, 
    test_loader, 
    device, 
    aug_dict,
    transform=None,
    augmentation_method=None,
    severities=(0,1),
):  
    aug_class = None
    for k, v in aug_dict.items():
        if k == augmentation_method:
            aug_class = v
            break 
    if aug_class == None:
        raise ValueError("aug method not valid")

    if severities[0] in ["None", 0]:
        loader_A = test_loader
    else: 
        loader_A = aug_class.corr_func_dataloader(test_loader, severity_idx = severities[0])
    loader_B = aug_class.corr_func_dataloader(test_loader, severity_idx = severities[1])


    imgs_A, probs_A, labels = collect_probs(model, loader_A, device)
    imgs_B, probs_B, _      = collect_probs(model, loader_B, device)

    N = len(labels)
    idx = torch.arange(N)

    pA = probs_A[idx, labels]
    pB = probs_B[idx, labels]

    brittleness = pA - pB

    results_all = [
        BrittlenessResultIndiv(
            index=i,
            label=int(labels[i]),
            predA=probs_A[i].argmax().item(),
            predB=probs_B[i].argmax().item(),
            pA=float(pA[i]),
            pB=float(pB[i]),
            brittleness=float(brittleness[i]),
        ) for i in range(N)
    ]
    # Sort (most brittle first)
    results_all_sorted = sorted(results_all, key=lambda x: x.brittleness, reverse=True)

    b_result = BrittlenessResult(
        results = results_all_sorted,
        imgsA = imgs_A, 
        imgsB = imgs_B, 
        probs_A = probs_A,
        probs_B = probs_B,
        labels = labels
    )

    return b_result
# =============================================================================================

if __name__ == "__main__":
    from cvrob_util import SimpleCNN
    from torchvision import datasets
    import torchvision.transforms as transforms

    device = torch.device("cpu")#"cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    # test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
    # class_names = train_dataset.classes
    # ===============================================================================================================

    print("noisy indices done! let's go to: robustness evaluation")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load('label_noise_simplecnn.h5', weights_only=True))

    clean_test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
    clean_test_loader = torch.utils.data.DataLoader(clean_test_dataset, batch_size=128, shuffle=False)
    class_names = clean_test_dataset.classes

    augmentation_list, augmentation_str, corrupt_func = get_corruption_helpers('album')

    b_result = brittle_method_simple(
        model, 
        clean_test_loader, 
        device, 
        corrupt_func, 
        augmentation_list,
        augmentation_str,
        transform=transform,
        augmentation_method="Gaussian Blur",
        severities=(0,1),
        # top_proportion=0.05,
        class_names=class_names
    )

    results = [
        r for r in b_result.results
        if r.predA == r.label and r.predB != r.label
    ]
    # results_sorted = sorted(results, key=lambda x: x.brittleness, reverse=True)

    print(f"TRANSFORM: {transform}")
    visualize_topk_matplotlib(results, b_result.imgsA, b_result.imgsB, b_result.probs_A, b_result.probs_B,  K=10, class_names=class_name, transform=transform)
    visualize_topk_plotly(results, b_result.imgsA, b_result.imgsB, b_result.probs_A, b_result.probs_B, K=10, class_names=class_names, transform=transform)
    visualize_in_html(results, b_result.imgsA, b_result.imgsB, b_result.probs_A, b_result.probs_B, b_result.labels, class_names=class_names, transform=transform)

    # # probability of the correct class
    # idx = torch.arange(len(labels))
    # pA = probs_A[idx, labels]
    # pB = probs_B[idx, labels]

    # brittleness = pA - pB
    # sorted_indices = torch.argsort(brittleness, descending=True)

    # # Most brittle images first
    top_proportion=0.05
    K = int(top_proportion*len(b_result.results)) if top_proportion < 1 else top_proportion
    most_brittle = b_result.results[:K]

    # return most_brittle