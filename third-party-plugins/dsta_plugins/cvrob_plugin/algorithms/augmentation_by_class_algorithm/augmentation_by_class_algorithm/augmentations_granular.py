from augmentations import *
import pandas as pd 
from pprint import pprint 
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from .cvrob_util import evaluate_1img, get_metric_dict
from torch.utils.data import TensorDataset
from .augmentations_class import *

def plot_corrupted_images(list_of_images, severities, filename=None, graph_lib='matplotlib', pred_label_list=None, class_names=None, label=None):
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
        return plot_corrupted_images_mpl(list_of_images, severities, filename, pred_label_list, class_names, label)
    elif graph_lib == 'plotly':
        return plot_corrupted_images_plotly(list_of_images, severities, filename, pred_label_list, class_names, label)
    else:
        raise ValueError('not valid graphing library')

def plot_corrupted_images_plotly(list_of_images, severities, filename=None, pred_label_list=None, class_names=None, label=None):

    cols = len(severities)
    
    # Create a subplot grid
    fig = make_subplots(rows=1, cols=cols,
                        subplot_titles=severities,
                        vertical_spacing=0.02, horizontal_spacing=0.02)

    for j, severity in enumerate(severities):
        # Apply corruption
        corrupted_img = list_of_images[j]
        # Convert to uint8 if needed
        if corrupted_img.dtype != np.uint8:
            corrupted_img = np.clip(corrupted_img, 0, 255).astype(np.uint8)
        # Add image to subplot
        fig.add_trace(
            go.Image(z=corrupted_img),
            row=1,
            col=j + 1
        )

        # Custom Y-axis label (as annotation)
        if j == 0:
            fig.add_annotation(
                text=severity,
                xref="paper", yref="paper",
                x=0, y=1,
                showarrow=False,
                font=dict(size=12),
                xanchor="right",
                yanchor="middle"
            )

    # Set layout
    fig.update_layout(
        height=150,
        width=200 * cols,
        title_text="Corrupted Image Grid (by Severity)",
        showlegend=False
    )

    # Hide axes
    for j in range(1, cols + 1):
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=j)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=j)

    # Save to file if filename is provided
    if filename:
        fig.write_image(filename)

    fig.show()
    return fig

def plot_corrupted_images_mpl(list_of_images, severities, filename=None, pred_label_list=None, class_names=None, label=None):
    
    fig, axes = plt.subplots(2, len(severities), squeeze=False, figsize=(20*len(severities),32))
    # print('<', len(pred_label_list))
    for j, severity in enumerate(severities):
        
        corrupted_img = list_of_images[j]
        axes[0, j].imshow(corrupted_img)
        # print('<',pred_label_list[j])
        lab_range = list(range(len(pred_label_list[j][0])))
        predicted_class = pred_label_list[j][1]

        if class_names is not None: #dict
            predicted_class = class_names[predicted_class]
            if label is not None and type(label)==int:
                label = class_names[label]
            lab_range = [class_names[i] for i in lab_range]

        if label is None:
            border_color = 'black'
        elif label != predicted_class:
            border_color = 'red'
        else:
            border_color = 'green'

        axes[0, j].tick_params(color=border_color, labelcolor=border_color)
        for spine in axes[0, j].spines.values():
            spine.set_edgecolor(border_color)

        colors = ['black']*len(pred_label_list[j][0])
        if label is not None:
            lab_idx = lab_range.index(label); colors[lab_idx] = 'blue'

        axes[1, j].barh(lab_range, pred_label_list[j][0], color=colors)
        axes[1, j].set_xlim(0,1)
        axes[1, j].tick_params(axis='both', labelsize=30)
        axes[1, j].text(
            0.5, -0.2, 
            f"Predicted Class : {predicted_class}", 
            horizontalalignment='center', verticalalignment='top', 
            transform=axes[0, j].transAxes , fontsize=60, color=border_color
        )
        axes[1, j].text(
            0.5, -0.07, 
            f"Label : {label}", 
            horizontalalignment='center', verticalalignment='top', 
            transform=axes[0, j].transAxes , fontsize=60, color=border_color
        )

        axes[0, j].set_title(f'Severity {severity}', fontsize=75)
    
    plt.tight_layout()
    plt.show()

    if filename is not None:
        fig.savefig(os.path.join('brittleness', filename))
        print(f'saving to {filename}')

    return fig

def augmentation_granular_one_augmentation(
        model, 
        test_loader, 
        metric_dict, #dict of functions
        augmentation,
        old_results={}
    ):
    aug_class, severity = augmentation
    corrupted_loader = aug_class.corr_func_dataloader(test_loader, severity)
    result_list = []

    base_acc, y_pred, y_true = evaluate(model, corrupted_loader, next(model.parameters()).device)
    features, labels = get_logits(model, corrupted_loader, next(model.parameters()).device)
    labels = [int(l) for l in labels]
    pred_probs = softmax(features, axis=1)

    for metric_name, metric_func in metric_dict.items():
        if metric_name in old_results:
            result_list.append(old_results[metric_name])
        else:
            metric = metric_func(base_acc, y_pred, y_true, features, labels, pred_probs)
            metric = round(float(metric), 4)
            result_list.append(metric)
    raw_list = [base_acc, y_pred, y_true, features, labels, pred_probs]

    return result_list, raw_list

def augmentation_granular_method_1img(
    img_array, 
    metric_dict, #dict of functions
    augmentation_dict,
    model,
    device,
    path="",
    graph_lib='matplotlib',
    class_names=None,
    label=None,
):
    for aug_name, aug_class in augmentation_dict.items():
        k1 = aug_name
        if k1 == "None":
            continue
        img_list = [img_array]
        probs, pred = evaluate_1img(model, device, img_array)
        pred_label_list = [(probs, pred, label)]
        for k2 in aug_class.severities:
            corrupted_img = aug_class.corr_func_one_img(img_array, k2)
            probs, pred = evaluate_1img(model, device, corrupted_img)
            ppl = (probs, pred, label)
            img_list.append(corrupted_img)
            pred_label_list.append(ppl)
        severities = ["original"] + list(aug_class.severities) 

        fig = plot_corrupted_images(
            img_list, 
            severities, 
            filename=f'test_visual_{k1}_{path}.png', 
            graph_lib=graph_lib, 
            pred_label_list=pred_label_list,
            class_names=class_names,
            label=label
        )       

def consolidate_metric_per_aug(consolidate_dict):
    SEVERITY_LEN = len(consolidate_dict)
    last_key = list(consolidate_dict.keys())[-1]
    DATA_LEN = len(consolidate_dict[last_key][1])
    
    tbw_list1 = []; tbw_list2 = []
    for j in range(DATA_LEN):
        min_magnitude = None
        for i, (k,v) in enumerate (consolidate_dict.items()):
            base_acc, y_pred, y_true, features, labels, pred_probs = v 
            if y_pred[j] != y_true[j]:
                min_magnitude = (i,k)
                break 
        if min_magnitude is None:
            tbw_i = (1, last_key)
        else:
            tbw_i = (min_magnitude[0]/SEVERITY_LEN, min_magnitude[1])
        tbw_list1.append(tbw_i[0]); tbw_list2.append(tbw_i[1])
    return tbw_list1, tbw_list2, sum(tbw_list1)/len(tbw_list1)

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

def augmentation_granular_method_sklearn(
    model, 
    test_loader, 
    device, 
    metric_dict,
    augmentation_dict,
    target_names=None,
    num_iterations=5
):
    model = model.to(device)
    big_dict = {}; big_dict2 = {}
    for aug_name, aug_class in augmentation_dict.items():
        k1 = aug_name
        big_dict[aug_name] = {}
        big_dict2[aug_name] = {}
        severities = ["None"] + aug_class.severities
        for s,k2 in enumerate(severities):
            K = num_iterations if k2 != "None" else 1 
            all_reports = []; all_cm = []
            for i in range(K):
                seed = 1000*s + i 
                aug_class.set_seed(seed)

                if k2 == "None":
                    corrupted_loader = test_loader
                else:
                    corrupted_loader = aug_class.corr_func_dataloader(test_loader, k2)
                base_acc, y_pred, y_true = evaluate(model, corrupted_loader, next(model.parameters()).device)
                report = classification_report(y_true, y_pred, labels=list(range(len(target_names))), target_names=target_names, output_dict=True, zero_division=np.nan)
                cm = confusion_matrix(y_true, y_pred, labels=list(range(len(target_names))))
                all_reports.append(report); all_cm.append(cm)

            avg_report = average_all_reports(all_reports)
            avg_cm = np.mean(all_cm, axis=0)
            TP = np.diag(avg_cm); FP = avg_cm.sum(axis=0) - TP; FN = avg_cm.sum(axis=1) - TP
            TN = avg_cm.sum() - (TP+FP+FN); N = avg_cm.sum()
            cm_stats = {
                label: {
                    "TP": float(TP[i])/N, "FP": float(FP[i])/N,
                    "FN": float(FN[i])/N, "TN": float(TN[i])/N
                } for i, label in enumerate(target_names)
            }
            big_dict[aug_name][k2] = avg_report
            big_dict2[aug_name][k2] = cm_stats 

    return big_dict, big_dict2

def augmentation_granular_method(
    model, 
    test_loader, 
    device, 
    metric_dict, #dict of functions
    augmentation_dict,
    old_df=None,
    replace=False,
):
    cols = ["perturbation_method", "hyperparam"] + list(metric_dict.keys())
    model = model.to(device)
    big_list = []
    for aug_name, aug_class in augmentation_dict.items():
        k1 = aug_name
        consolidate_dict = {}
        for k2 in aug_class.severities:
            print('-k', k1, k2)
            old_results = {}
            if old_df is not None and replace is False:
                if k1 == "None" and k2 == "None":
                    continue
                any_bool = old_df[(old_df["perturbation_method"]==k1) & (old_df["hyperparam"]==k2)]
                print('-b', any_bool)
                if any_bool:
                    row = old_df[(old_df["perturbation_method"]==k1) & (old_df["hyperparam"]==k2)]
                    print('--', len(row), len(cols))
                    if len(cols)==len(row):
                        print('-- continue')
                        continue
                    else:
                        old_results = {k:row[k] for k,v in metric_dict.items() if k in row}
                        

            header_list = [k1, k2]
            result_list, raw_list = augmentation_granular_one_augmentation(
                            model, 
                            test_loader, 
                            metric_dict, #dict of functions
                            augmentation=(aug_class, k2),
                            old_results=old_results
                        )
            result_list = header_list + result_list
            consolidate_dict[k2] = raw_list
            big_list.append(result_list)

        consolidate_metrics = consolidate_metric_per_aug(consolidate_dict)
        print('='*16)
        print(k1)
        print(consolidate_metrics[0][:15], consolidate_metrics[1][:15], consolidate_metrics[2])
        print('='*16)
    df = pd.DataFrame(big_list, columns=cols)
    if old_df is not None and replace is True:
        merged_df = pd.concat([
            df, old_df[~old_df.set_index(["perturbation_method", "hyperparam"]).index.isin(
                df.set_index(["perturbation_method", "hyperparam"]).index
            )]
        ])
    elif old_df is not None and replace is False:
        merged_df = pd.concat([old_df, df])
    else:
        merged_df = df

    merged_df = merged_df[
        ~((merged_df["perturbation_method"] == "") & (merged_df["hyperparam"] == ""))
    ]
    merged_df = merged_df[
        ~(pd.isna(merged_df["perturbation_method"]) & pd.isna(merged_df["hyperparam"]))
    ]
    return merged_df.reset_index(drop=True)
   
def visualize_metric_dataframe(df, method, change='absolute', appendum=''):
    plt.close()
    sub_df = df[df["perturbation_method"].isin(["None", method])]
    # print( '>', sub_df["perturbation_method"])
    metric_cols = list(sub_df.columns)[2:]
    x_col = sub_df['hyperparam']
    fig,ax = plt.subplots(figsize=(20,16))

    for col in metric_cols:
        if change == 'absolute':
            value_arr = np.array(sub_df[col]) 
        elif change == 'ratio':
            value_arr = np.array(sub_df[col]) / np.array(sub_df[col])[0]
        else:
            raise ValueError("Only valid options for change are absolute and ratio")

        ax.scatter(x_col, value_arr, label=col)
        ax.plot(x_col, value_arr)
    ax.set_title(f"Model Performance Metrics for Perturbation Method {method}")
    ax.set_xlabel("Hyperparameters", fontsize=16)
    ax.set_ylabel("Value", fontsize=16)
    if change == 'absolute':
        ax.set_ylim([0,1])
    else:
        ax.set_ylim([0,2])
    ax.legend()
    plt.savefig(os.path.join('brittleness', f"metric_plot_indiv_{change}_{method}_{appendum}.png"))
    plt.show()

if __name__ == "__main__":
    from util import *
    args = json_to_argparse_args('argparse_config.json')
    DATA_DIR = ''

    img_array = load_data(args.image_file_path, DATA_DIR)
    model = load_data(args.model_file_path, DATA_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    with open('boat_classes.json', 'r') as f:
        class_names = json.load(f)
    class_names = {int(k):v for k,v in class_names.items()}

    test_dataset = load_data(args.test_file_path, DATA_DIR)
    param_dict = load_data(args.params_file_path, DATA_DIR)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=param_dict['batch_size'], 
        shuffle=False
    )

    metric_dict = get_metric_dict()
    metric_dict = {k:v for k,v in metric_dict.items() if k not in ['auc', 'recall']}
    augmentation_dict = make_augmentation_dict_album2()

    augmentation_dict = {'GaussianBlur': augmentation_dict['GaussianBlur']}
    all_aug_methods = ['GaussianBlur']

    report, report2 = augmentation_granular_method_sklearn(
        model, 
        test_loader, 
        device, 
        metric_dict, #dict of functions
        augmentation_dict,
        target_names=[k for k in class_names]
    )
    import json 
    with open("sklearn_report.json", 'w') as f:
        json.dump(report, f, indent=4)
    with open("sklearn_report2.json", 'w') as f:
        json.dump(report2, f, indent=4)
