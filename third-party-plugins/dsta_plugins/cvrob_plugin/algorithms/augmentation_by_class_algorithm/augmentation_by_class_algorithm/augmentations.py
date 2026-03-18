import torch 
import numpy as np
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import albumentations as A
from albumentations.pytorch import ToTensorV2
from nrtk.impls.perturb_image.generic.cv2.blur import (
    AverageBlurPerturber, 
    GaussianBlurPerturber, 
    MedianBlurPerturber
)
from nrtk.impls.perturb_image.generic.PIL.enhance import (
    BrightnessPerturber,
    ColorPerturber,
    ContrastPerturber,
    SharpnessPerturber,
)
from nrtk.impls.perturb_image.generic.skimage.random_noise import (
    GaussianNoisePerturber,
    PepperNoisePerturber,
    SaltAndPepperNoisePerturber,
    SaltNoisePerturber,
    SpeckleNoisePerturber,
)
import plotly.graph_objects as go
from augly.image import blur, brightness, random_noise, contrast, color_jitter, pixelization, sharpen
from augly.image import aug_np_wrapper
from .cvrob_util import (plot_accuracy_vs_severity,
                        evaluate,
                        get_logits,
                        best_fit_gradient)
from plotly.subplots import make_subplots
from imagecorruptions import corrupt
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon
from .augmentations_class import *

def corrupt_and_plot_generic_plotly(img, augmentations, aug_names, corrupt_func, filename=None):
    """Method for corrupting (read: one!) an image and plotting it using: plotly

    Takes the base image, and severity=1 and severity=2 images.

    Args:
        img (np.array): np.array representing image
        augmentations (list): list of two-length tuples 
        aug_names (list): list of names of the given libraries
        corrupt_func (function): custom corrupt function: takes in images np array, severity, and augmentation parameters and returns corrupted images np array
        filename (str, optional): string of filename to save the matrix of augmented images. Defaults to None.
        graph_lib (str, optional): graphing library used. Defaults to 'matplotlib'.

    Raises:
        ValueError: invalid graphing library provided. Must be either matplotlib or plotly.

    Returns:
        fig (plotly.Figure): (matrix) figure of augmented image.
    """
    severities = [0, 1, 2]
    rows = len(augmentations)
    cols = len(severities)
    
    # Create a subplot grid
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[f"Severity {s}" for s in severities] * (1 if rows == 1 else 0),
                        vertical_spacing=0.02, horizontal_spacing=0.02)

    for i, (aug_class, param_dict) in enumerate(augmentations):
        param_dict['aug_method'] = aug_class
        for j, severity in enumerate(severities):
            # Apply corruption
            corrupted_img = img if severity == 0 else corrupt_func([img], severity, param_dict)[0]
            # Convert to uint8 if needed
            if corrupted_img.dtype != np.uint8:
                corrupted_img = np.clip(corrupted_img, 0, 255).astype(np.uint8)

            # Add image to subplot
            fig.add_trace(
                go.Image(z=corrupted_img),
                row=i + 1,
                col=j + 1
            )

            # Custom Y-axis label (as annotation)
            if j == 0:
                fig.add_annotation(
                    text=aug_names[i],
                    xref="paper", yref="paper",
                    x=0, y=1 - (i / rows),
                    showarrow=False,
                    font=dict(size=12),
                    xanchor="right",
                    yanchor="middle"
                )

    # Set layout
    fig.update_layout(
        height=150 * rows,
        width=200 * cols,
        title_text="Corrupted Image Grid (by Severity & Augmentation)",
        showlegend=False
    )

    # Hide axes
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=i, col=j)
            fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=i, col=j)

    # Save to file if filename is provided
    if filename:
        fig.write_image(filename)

    fig.show()
    return fig

def get_max_severities(augmentation_dict):
    res = -1
    for aug_name, aug_class in augmentation_dict.items():
        if len(aug_class.severities) > res:
            res = len(aug_class.severities)
    return res

def corrupt_and_plot_generic_mpl(img, augmentation_dict_old, filename=None):

    augmentation_dict = {k:v for k,v in augmentation_dict_old.items() if k != "None"}
    max_severities = 3 #get_max_severities(augmentation_dict)
    fig, axes = plt.subplots(len(augmentation_dict), max_severities, figsize=(15, 15))
    
    for i, (aug_name, aug_class) in enumerate(augmentation_dict.items()):

        # param_dict['aug_method'] = aug_class
        for severity in range(max_severities):
            j = severity            
            if severity != 0:
                # print('=',aug_class.severities, severity)
                corrupted_img = aug_class.corr_func_one_img(img, severity)
                #corrupted_img = corrupt_func([img], severity, param_dict)[0]
            else:
                corrupted_img = img
            axes[i, j].imshow(corrupted_img)
            #axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_ylabel(aug_name, fontsize=12, rotation=0, labelpad=40, verticalalignment='center')
            severity_list = ["None"] + aug_class.severities
            axes[i, j].set_title(f'Severity {severity_list[severity]}', fontsize=12)
    
    # for j, severity in enumerate(severities):
    #     axes[0, j].set_title(f'Severity {severity}', fontsize=12)
    
    plt.tight_layout()
    plt.show()

    if filename is not None:
        fig.savefig(filename)

    return fig

def corrupt_and_plot_generic(img, augmentation_dict, filename=None, graph_lib='matplotlib'):
    """Header method for corrupting (read: one!) an image and plotting it

    Args:
        img (np.array): np.array representing image
        augmentations (list): list of two-length tuples 
        aug_names (list): list of names of the given libraries
        corrupt_func (function): custom corrupt function: takes in images np array, severity, and augmentation parameters and returns corrupted images np array
        filename (str, optional): string of filename to save the matrix of augmented images. Defaults to None.
        graph_lib (str, optional): graphing library used. Defaults to 'matplotlib'.

    Raises:
        ValueError: invalid graphing library provided. Must be either matplotlib or plotly.

    Returns:
        fig: (matrix) figure of augmented image.
    """
    if graph_lib == 'matplotlib':
        return corrupt_and_plot_generic_mpl(img, augmentation_dict, filename)
    elif graph_lib == 'plotly':
        return corrupt_and_plot_generic_plotly(img, augmentation_dict, filename)
    else:
        raise ValueError('not valid graphing library')

def get_nrtk_augmentations_list(seed=42):
    """
    Get the augmentation list and string for nrtk.

    Args:
        seed (int): random seed (fixed for now)

    Returns:
        augmentation_list (list): list of two-length tuples 
        augmentation_str (list): list of names of the given libraries
    """
    perturbations = [
        # (SaltNoisePerturber, {'rng': seed, 'amount': lambda s: 0.15 * s}),
        # (PepperNoisePerturber, {'rng': np.random.default_rng(seed), 'amount': lambda s: 0.15 * s}),
        (SaltAndPepperNoisePerturber, {'rng': np.random.default_rng(seed), 'amount': lambda s: 0.15 * s}),
        (GaussianNoisePerturber, {'rng': seed, 'mean': lambda s: 0.1 * s, 'var': lambda s: 0.01 * s}),
        # (SpeckleNoisePerturber, {'rng': seed, 'mean': lambda s: 0.3 * s, 'var': lambda s: 0.01 * s}),
        # (AverageBlurPerturber, {'ksize': lambda s: 11 + 2 * s}),
        (GaussianBlurPerturber, {'ksize': lambda s: 11 + 4 * s}),
        # (MedianBlurPerturber, {'ksize': lambda s: 11 + 2 * s}),
        (BrightnessPerturber, {'factor': lambda s: 1 + 0.15 * s}),
        # (ColorPerturber, {'factor': lambda s: 1 - 0.15 * s}),
        (ContrastPerturber, {'factor': lambda s: 1 + 0.2 * s}),
        (SharpnessPerturber, {'factor': lambda s: 1 - 0.2 * s}),
    ]
    perturb_names = ["Salt Pepper Noise", "Gaussian Noise", "Gaussian Blur", "Brightness", "Contrast", "Sharpness"]
    # perturb_names = ["Salt Noise", "Pepper Noise", "Salt Pepper Noise", "Gaussian Noise", "Speckle Noise", \
    #                  "Average Blur", "Gaussian Blur", "Median Blur", "Brightness", "Color", "Contrast", "Sharpness"]
    return perturbations, perturb_names

def get_augly_augmentations_list():
    """
    Get the augmentation list and string for augly.

    Args:
        None.

    Returns:
        augmentation_list (list): list of two-length tuples 
        augmentation_str (list): list of names of the given libraries
    """
    augmentations = [
        (blur, {'radius': lambda s: s}),  # Corrected: Replaced 'severity' with lambda
        (brightness, {'factor': lambda s: 1 + 0.1 * s}),
        (contrast, {'factor': lambda s: s}),  # Corrected
        (random_noise, {'mean': lambda s: 0.0001 * s, 'var': lambda s: 0.00001})
    ]
    aug_str = ['blur', 'brightness', 'contrast', 'random_noise']
    return augmentations, aug_str

def get_imagecorrupt_augmentations_list():
    """
    Get the augmentation list and string for imagecorrupt.

    Args:
        None.
        
    Returns:
        augmentation_list (list): list of two-length tuples 
        augmentation_str (list): list of names of the given libraries
    """
    # gaussian_noise, shot_noise, impulse_noise, defocus_blur,
    #                 glass_blur, motion_blur, zoom_blur, snow, frost, fog,
    #                 brightness, contrast, elastic_transform, pixelate,
    #                 jpeg_compression, speckle_noise, gaussian_blur, spatter,
    #                 saturate
    augmentations = [
        (corrupt, {'corrname': 'gaussian_noise'}),
        (corrupt, {'corrname': 'fog'}),
        (corrupt, {'corrname': 'brightness'}),
        (corrupt, {'corrname': 'zoom_blur'}),
    ]
    aug_str = ['gaussian_noise', 'fog', 'brightness', 'zoom_blur']
    return augmentations, aug_str

def corrupt_and_plot_multiple(
        augmentation_libraries,
        img_array,
        filename,
        graph_lib='matplotlib'
):
    fig_dict = {}
    for lib in augmentation_libraries:
        # augmentation_list, augmentation_str, corrupt_func = get_corruption_helpers(lib)
        augmentation_dict = make_augmentation_dict(lib)
        fig = corrupt_and_plot_generic(img_array, 
                                       augmentation_dict,
                                       filename.replace('lib', lib),
                                       graph_lib='matplotlib')
        fig_dict[lib] = fig

    return fig_dict

def augmentation_class_confidences(model, test_loader, device, aug_class, plot_graphs=False):
    features, labels = get_logits(model, test_loader, device)
    labels = [int(l) for l in labels]
    pred_probs = softmax(features, axis=1)
    severities = [0, 1, 2]
    all_probs = [pred_probs]
    all_labels = [labels]
    for severity in severities[1:]:
        print(f"Evaluating on severity {severity}...")
        corrupted_loader=  aug_class.corr_func_dataloader(test_loader, severity)
        # corrupted_loader = get_corrupted_dataloader(test_loader, 
        #                                             corr_func, 
        #                                             severity=severity,
        #                                             corr_kwargs=corr_kwargs)
        features, labels = get_logits(model, corrupted_loader, device)
        labels = [int(l) for l in labels]
        pred_probs = softmax(features, axis=1)
        all_probs.append(pred_probs)
        all_labels.append(labels)
    return np.array(all_probs), np.array(all_labels)

def augmentation_class_confidences_method(
    model, 
    test_loader, 
    device, 
    augmentation_dict,
    # augmentation_func_wrapper, 
    # augmentation_list,
    # augmentation_str,
    plot_graphs=False
):
    """Big header method for determining how model behaves with different augmentation severities

    Iterate through the augmentation iterables for each augmentation technique (e.g. blur, contrast, etc for imagecorr)

    Args:
        model (torch.nn): model (torch model for now)
        test_loader (torch.utils.data.DataLoader): data loader
        device (torch.device): cuda/cpu device
        augmentation_func_wrapper (function): corruption function (e.g. corrupt_func_album)
        augmentation_list (list): list of tuples for augmentation
        augmentation_str (str): list of augmentation techniques for given augmentation method
        plot_graphs (bool, optional): False or the graphing library to plot graph. Defaults to False.

    Returns:
        Tuple: 
        - augmentation_dict (dict): dictionary of augmentations <-> techniques with respective gradient and first drop scores
        - augmentation_fig_dict (dict): dictionary of augmentations <-> techniques with figures of the performance vs aug severity 1-5
    """

    # assert len(augmentation_list) == len(augmentation_str)

    model = model.to(device)
    augmentation_dict = {}
    augmentation_lab_dict = {}
    for aug_name, aug_class in augmentation_dict.items():
    # for k, (m,d) in zip(augmentation_str, augmentation_list):
    #     d['aug_method'] = m
        pred_probs, labels = augmentation_class_confidences(model, test_loader, device, aug_class, plot_graphs)
        augmentation_dict[aug_name] = pred_probs
        augmentation_lab_dict[aug_name] = labels

    return augmentation_dict, augmentation_lab_dict 


def augmentation_gradient(model, test_loader, device, aug_class, plot_graphs=False):
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
    print(f"Evaluating on severity 0...")
    base_acc, _ , _ = evaluate(model, test_loader, device)
    print(f"Accuracy at severity 0: {base_acc:.4f}")
    severities = [x for x in range(len(aug_class.severities))]
    accuracies = [base_acc]
    for severity in severities[1:]:
        print(f"Evaluating on severity {severity}...")
        corrupted_loader = aug_class.corr_func_dataloader(test_loader, severity_idx=severity)
        # corrupted_loader = get_corrupted_dataloader(test_loader, 
        #                                             corr_func, 
        #                                             severity=severity,
        #                                             corr_kwargs=corr_kwargs)
        acc, _,_ = evaluate(model, corrupted_loader, device)
        accuracies.append(acc)
        print(f"Accuracy at severity {severity}: {acc:.4f}")

    # Plot results
    fig = None
    if plot_graphs is not False:
        fig = plot_accuracy_vs_severity(accuracies, severities, plot_graphs)  
    return best_fit_gradient(severities, accuracies), accuracies, fig


def augmentation_perf_gradient_method(
    model, 
    test_loader, 
    device, 
    augmentation_class_dict,
    plot_graphs=False
):
    """Big header method for determining how model behaves with different augmentation severities

    Iterate through the augmentation iterables for each augmentation technique (e.g. blur, contrast, etc for imagecorr)

    Args:
        model (torch.nn): model (torch model for now)
        test_loader (torch.utils.data.DataLoader): data loader
        device (torch.device): cuda/cpu device
        augmentation_func_wrapper (function): corruption function (e.g. corrupt_func_album)
        augmentation_list (list): list of tuples for augmentation
        augmentation_str (str): list of augmentation techniques for given augmentation method
        plot_graphs (bool, optional): False or the graphing library to plot graph. Defaults to False.

    Returns:
        Tuple: 
        - augmentation_dict (dict): dictionary of augmentations <-> techniques with respective gradient and first drop scores
        - augmentation_fig_dict (dict): dictionary of augmentations <-> techniques with figures of the performance vs aug severity 1-5
    """
    model = model.to(device)
    augmentation_dict = {}
    augmentation_fig_dict = {}
    for aug_name, aug_class in augmentation_class_dict.items():
        #d['aug_method'] = m
        gradient, accuracies, fig = augmentation_gradient(model, test_loader, device, aug_class, plot_graphs)
        first_drop = accuracies[1] - accuracies[0]
        
        print(accuracies, first_drop)
        print(aug_name, 'augmentation method gradient:', gradient)
        augmentation_dict[aug_name] = (gradient, first_drop)
        augmentation_fig_dict[aug_name] = fig
        
        print()
    return augmentation_dict, augmentation_fig_dict 

def eval_conf_dict(conf_dict, label_dict):
    metric_dict = {}
    modes = ['acc', 'acc_byclass', 'top1', 'top1correct', 'top1wrong', 'correctclass', \
    'allclasses_proba', 'allclasses_label', 'nll', 'ece']
    for k,pred_probs in conf_dict.items():
        labels = label_dict[k]
        metric_dict[k] = {}
        for mode in modes:
            metric_dict[k][mode] = {}
            for i in range(len(pred_probs)):
                res = indiv_confidence_eval(pred_probs[i], labels[i], mode)
                metric_dict[k][mode][i] = res
            arr = []
    return metric_dict

def display_metric_dict(metric_dict):
    for k1,v1 in metric_dict.items():
        print("- augmentation method", k1)
        temp_dict = {}
        temp_dict['all_others'] = []
        temp_dict['all_others_names'] = []

        for k2,v2 in v1.items():
            print("-> metric", k2)
            num_keys = len(list(v2.keys()))
            big_list = []; medium_list = []
            if type(v2[0]) == float:
                some_list = [v3 for k3,v3 in v2.items()]
                temp_dict['all_others'].append(some_list)
                temp_dict['all_others_names'].append(k2)
            else:
                big_list2 = []
                assert type(v2[0]) == list
                assert all(isinstance(i,float) for i in v2[0])
                for i in range(len(v2[0])):
                    some_list = [v3[i] for k3,v3 in v2.items()]
                    big_list2.append(some_list)
                temp_dict[k2] = big_list2

        for k,v in temp_dict.items():
            if k == 'all_others_names':
                continue
            elif k == 'all_others':
                plot_grouped_bar(v, temp_dict['all_others_names'], k)
            else:
                plot_grouped_bar(v, [str(x) for x in range(len(v))], k)

def plot_grouped_bar(list_of_lists, list_of_names, title="placeholder"):
    plt.close()
    #TODO some assertion to be done here
    vals= np.array(list_of_lists, dtype=float)
    n_classes, n_metrics = vals.shape 
    assert len(list_of_names) == n_classes

    metric_labels = list(range(n_metrics))
    x = np.arange(n_classes)
    total_group_width = 0.8
    bar_width = total_group_width / n_metrics
    offsets = (np.arange(n_metrics) - (n_metrics - 1) / 2) * bar_width
    fig,ax = plt.subplots(figsize=(8,6))

    for j in range(n_metrics):
        ax.bar(x+offsets[j], vals[:,j], width = bar_width, label = metric_labels[j])

    ax.set_xticks(x); ax.set_xticklabels(list_of_names); ax.set_ylabel("Value"); ax.set_xlabel("Class") 
    #TODO class metrics change idk
    ax.set_title(title); ax.legend()
    plt.tight_layout()
    plt.savefig(f'outputs/grouped_bar_{title}.png') #TODO
    plt.show()

def indiv_confidence_eval(pred_probs, labels, mode='top1'):
    if mode == 'top1':
        l = indiv_confidence_eval_t1(pred_probs, labels)
    elif mode == 'top1correct':
        l = indiv_confidence_eval_t1c(pred_probs, labels)
    elif mode == 'top1wrong':
        l = indiv_confidence_eval_t1w(pred_probs, labels)
    elif mode == 'correctclass':
        l = indiv_confidence_eval_cc(pred_probs, labels)
    elif mode == 'nll':
        return nll(pred_probs, labels)
    elif mode == 'ece':
        return ece(pred_probs, labels)
    elif mode == 'acc':
        return acc(pred_probs, labels, False)
    elif mode == 'acc_byclass':
        return acc(pred_probs, labels, True)
    elif mode == 'allclasses_proba':
        return allclasses(pred_probs, labels, mode='proba')
    elif mode == 'allclasses_label':
        return allclasses(pred_probs, labels, mode='label')
    return float(l.mean())

def allclasses(pred_probs, labels, mode='label'):
    res = []
    for i in range(len(pred_probs[0])):
        if mode == 'label':
            pred_probs_temp = [x for t,x in enumerate(pred_probs) if labels[t] == i]
        if mode == 'proba':
            pred_probs_temp = [x for t,x in enumerate(pred_probs) if list(x).index(max(x)) == i]
        l = [x[i] for x in pred_probs_temp]
        if len(l) == 0:
            res.append(float(0))
        else:
            res.append(float(np.mean(l)))
    return res

def acc(pred_probs, labels, by_class=True):
    if by_class is False:
        corr =  np.array([x for i,x in enumerate(pred_probs) if list(x).index(max(x)) == labels[i]])
        return len(corr)/len(pred_probs)
    assert by_class is True
    res = []
    for i in range(len(pred_probs[0])):
        corr_i =  np.array([x for t,x in enumerate(pred_probs) if list(x).index(max(x)) == labels[t] == i])
        base_i = np.array([x for t,x in enumerate(pred_probs) if labels[t] == i])
        if len(base_i) == 0:
            res.append(float(0))
        else:
            res.append(len(corr_i)/len(base_i))
    return res

def indiv_confidence_eval_t1(pred_probs, labels):
    return np.array([max(x) for x in pred_probs])

def indiv_confidence_eval_t1c(pred_probs, labels):
    return np.array([max(x) for i,x in enumerate(pred_probs) if list(x).index(max(x)) == labels[i]])

def indiv_confidence_eval_t1w(pred_probs, labels):
    return np.array([max(x) for i,x in enumerate(pred_probs) if list(x).index(max(x)) != labels[i]])

def indiv_confidence_eval_cc(pred_probs, labels):
    return np.array([x[labels[i]] for i,x in enumerate(pred_probs)])

def nll(probs, labels, ep=1e-12):
    corr = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(corr + ep)))

def ece(probs, labels, n_bins=20):
    confs = probs.max(axis=1); preds = probs.argmax(axis=1); accs = preds == labels
    bin_boundaries = np.linspace(0,1,n_bins+1)
    ece = 0
    for i in range(n_bins):
        bin_l, bin_u = bin_boundaries[i], bin_boundaries[i+1]
        in_bin = (confs > bin_l) & (confs <= bin_u)
        if in_bin.any():
            bin_acc = accs[in_bin].mean()
            bin_conf = confs[in_bin].mean()
            ece += np.abs(bin_acc - bin_conf) * in_bin.mean()

    return float(ece)

def brittle_image_method(
    model, 
    test_loader, 
    device, 
    augmentation_dict,
    # augmentation_func_wrapper, 
    # augmentation_list,
    # augmentation_str,
):
    # assert len(augmentation_list) == len(augmentation_str)
    DATA_LEN = len(test_loader.dataset)
    model = model.to(device)
    metric_dict = {}; brittle_dict = {}

    # for k, (m,d) in zip(augmentation_str, augmentation_list):
    for k, aug_class in augmentation_dict.items():
        if k == "None":
            continue
        # print('>', k)
        # d['aug_method'] = m
        metric_dict[k] = {}; brittle_dict[k] = {}
        # for i in range(5+1):
        severity_list = ["None"] + aug_class.severities
        for i,severity in enumerate(severity_list):
            preds, pred_probs, labels = get_preds_and_proba_and_labels(
                model, test_loader, device, aug_class, severity, #augmentation_func_wrapper, d
            )
            metric_dict[k][i] = {}
            metric_dict[k][i]['pred_probs'] = pred_probs
            metric_dict[k][i]['preds'] = preds
            metric_dict[k][i]['labels'] = labels


        for i in range(1,1+len(aug_class.severities)):
            brittle_rank, brittle_metrics = brittleness_metrics_method(
                metric_dict[k][0]['pred_probs'], 
                metric_dict[k][i]['pred_probs'],
                metric_dict[k][0]['preds'],
                metric_dict[k][i]['preds']
            )
            brittle_dict[k][i] = {}
            brittle_dict[k][i]['brittle_rank'] = brittle_rank 
            brittle_dict[k][i]['brittle_metrics'] = brittle_metrics 

        # DATA_LEN = len(metric_dict[k][list(metric_dict[k].keys())[0]]['pred_probs'])
        # SEVERITY_LEN = 5
        tbw_list1 = []; tbw_list2 = []
        for j in range(DATA_LEN):
            min_magnitude = None
            SEVERITY_LEN = len(metric_dict[k])
            for i in metric_dict[k]: #range(1,1+SEVERITY_LEN):
                y_true = metric_dict[k][i]['labels']
                y_pred = metric_dict[k][i]['preds']
                if y_pred[j] != y_true[j]:
                    min_magnitude = (i,k)
                    break 
                else:
                    min_magnitude = (None, i)
            if min_magnitude[0] is None:
                tbw_i = (1, min_magnitude[1])
            else:
                tbw_i = (min_magnitude[0]/SEVERITY_LEN , min_magnitude[1])
            tbw_list1.append(tbw_i[0]); tbw_list2.append(tbw_i[1])

        all_brittle_metrics = [brittle_dict[k][i]['brittle_metrics'] for i in brittle_dict[k]]
        ave_brittle_metrics = [sum(i)/len(i) for i in all_brittle_metrics]

    return metric_dict, brittle_dict
        #print(pred_probs_diff.shape)
        # probs_diff_list.append(pred_probs_diff)
        # corr_preds_diff_list = np.concatenate((corr_preds_diff_list, corr_preds_diff), axis=0)
        # all_preds_diff_list = np.concatenate((all_preds_diff_list, all_preds_diff), axis=0)
        # # print(preds_diff_list)
        # print()

def get_preds_and_proba_and_labels(model, test_loader, device, aug_class, i):
    if i != 0 and i != "None":
        corrupted_loader = aug_class.corr_func_dataloader(test_loader, i)
        # corrupted_loader = get_corrupted_dataloader(test_loader, 
        #                                             corr_func, 
        #                                             severity=i,
        #                                             corr_kwargs=corr_kwargs)
    else:
        corrupted_loader = test_loader                                                
    features, labels = get_logits(model, corrupted_loader, device)
    labels = labels.astype(np.int64)
    pred_probs = softmax(features, axis=1)
    preds = pred_probs.argmax(axis=1)

    return preds, pred_probs, labels

def brittleness_metrics_method(p, p_pert, y, y_pert):
    flip_flag = (y != y_pert).astype(int)
    conf_change = np.abs(p[np.arange(len(y)), y] - p_pert[np.arange(len(y)), y_pert])
    js_div = np.array([jensenshannon(p[i] , p_pert[i]) for i in range(len(y))])
    rank_corr = np.array([spearmanr(np.argsort(-p[i]), np.argsort(-p_pert[i])).correlation for i in range(len(y))])
    brittle_score = flip_flag + js_div + conf_change
    brittle_rank = np.argsort(-brittle_score)
    return brittle_rank, [flip_flag , js_div , conf_change]

def combine_ranks_multiple_lists(lists):
    M = len(lists)
    N = len(lists[0])

    rank_maps = []
    for lst in lists:
        rank_map = {num: i+1 for i, num in enumerate(lst)}
        rank_maps.append(rank_map)

    combined_ranks = []
    for num in range(N):
        ranks = [rank_map[num] for rank_map in rank_maps]
        product_rank = 1
        for r in ranks:
            product_rank *= r 
        combined_ranks.append(product_rank)

    return combined_ranks

def get_combined_score(d):
    temp = []
    for k,v in d.items():
        lists = v['brittle_metrics']
        slists = sum(lists)
        temp.append(slists)
    stemp = sum(temp)
    return stemp
#combined_scores = brittle_dict[perturbation_type]

def get_sublist_class(values, indices, cls):
    return values[indices == cls]
#metrics = brittle_dict[perturbation_type][1]['brittle_metrics']
#indices = metric_dict[perturbation_type][1]['labels']
#sublist_class = [values[indices == cls] for values in metrics]

def get_brittle_dict_class(brittle_dict, metric_dict, cls):
    new_dict = {}
    for k1,v1 in brittle_dict.items():
        new_dict[k1] = {}
        for k2,v2 in v1.items():
            metrics = brittle_dict[k1][k2]['brittle_metrics']
            indices = metric_dict[k1][k2]['labels']
            new_dict[k1][k2] = [values[indices == cls] for values in metrics]

    return new_dict

def average_all_reports(reports):
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


# =============================================================================================

if __name__ == "__main__":
    from cvrob_util import SimpleCNN
    from torchvision import datasets
    import torchvision.transforms as transforms

    device = torch.device("cpu")#"cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
    # ===============================================================================================================

    print("noisy indices done! let's go to: robustness evaluation")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load('label_noise_simplecnn.h5', weights_only=True))

    clean_test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
    clean_test_loader = torch.utils.data.DataLoader(clean_test_dataset, batch_size=128, shuffle=False)

    # augmentation_list, augmentation_str, corrupt_func = get_corruption_helpers('nrtk')
    # aug_dict_g, aug_dict_f = augmentation_perf_gradient_method(
    #     model, 
    #     clean_test_loader, 
    #     device, 
    #     corrupt_func, 
    #     augmentation_list,
    #     augmentation_str,
    #     plot_graphs=False
    # )
    
    # print(aug_dict_g)

    augmentation_class_dict = make_augmentation_dict_album()
    aug_dict_g, aug_dict_f = augmentation_perf_gradient_method(
        model, 
        clean_test_loader, 
        device, 
        augmentation_class_dict,
        plot_graphs=False
    )
    
    print(aug_dict_g)