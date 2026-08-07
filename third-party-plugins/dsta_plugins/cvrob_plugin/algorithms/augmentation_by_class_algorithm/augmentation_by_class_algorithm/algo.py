import logging
from pathlib import Path, PurePath
from typing import Dict, List, Tuple, Union, Callable, Any, Optional
import copy
import shutil

from aiverify_test_engine.interfaces.ialgorithm import IAlgorithm
from aiverify_test_engine.interfaces.idata import IData
from aiverify_test_engine.interfaces.imodel import IModel
from aiverify_test_engine.interfaces.ipipeline import IPipeline
from aiverify_test_engine.interfaces.iserializer import ISerializer
from aiverify_test_engine.plugins.enums.model_type import ModelType
from aiverify_test_engine.plugins.enums.plugin_type import PluginType
from aiverify_test_engine.plugins.metadata.plugin_metadata import PluginMetadata
from aiverify_test_engine.utils.json_utils import load_schema_file, validate_json
from aiverify_test_engine.utils.simple_progress import SimpleProgress

import numpy as np
import torch
from .cvrob_util import evaluate, average_all_reports, handle_class_names_arg
from .augmentations_class import handle_url_algos
from .cvrob_algo_common import BasePlugin
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import pandas as pd 
import json
import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.graph_objects as go
from pprint import pprint
import matplotlib.lines as mlines

# =====================================================================================
# NOTE:
# 1. Check that you have installed the aiverify_test_engine latest package.
# 2. Check that you have run tests/install_core_plugins_requirements.sh to install all the
#    requirements required by the core plugins (serializers, data, models).
#    Alternatively, you may install the plugins that you require by installing the
#    requirements individually.
# 3. Do not modify the class name, else the plugin cannot be read by the system.
# =====================================================================================
class Plugin(BasePlugin):
    """
    # TODO: Update the plugin description below
    The Plugin(Augmentation by Class Algorithm) class specifies methods in generating results for algorithm
    """

    # Some information on plugin
    _name: str = "Augmentation by Class Algorithm"
    _description: str = "This algorithm shows the relationship between certain augmentations and the class statistics produced by the model"
    _version: str = "0.1.0"
    _metadata: PluginMetadata = PluginMetadata(_name, _description, _version)
    _plugin_type: PluginType = PluginType.ALGORITHM
    _requires_ground_truth: bool = True
    _supported_algorithm_model_type: List = [ModelType.CLASSIFICATION]

    # Column groups for the per-class plots, shared by both plotting backends.
    _METRIC_COLS = ['precision', 'recall', 'f1-score']
    _CM_COLS = ['TP', 'FP', 'FN', 'TN']
    _POP_COLS = ['preds_population', 'actual_population']

    def generate(self) -> None:
        """
        A method to generate the algorithm results with the provided data, model, ground truth information.
        """
        # Retrieve data information
        self._data = self._data_instance.get_data()

        file_names = [Path(i).name for i in self._data_instance.get_data()["image_directory"]]
        df: pd.DataFrame = self._ground_truth_instance.get_data()
        self._file_name_label = "file_name" #self._input_arguments["file_name_label"]
        self._ordered_ground_truth_df = df.set_index(self._file_name_label).reindex(file_names) 

        # Initialise main image directory
        if self._save_folder.exists():
            shutil.rmtree(self._save_folder)
        self._save_folder.mkdir(parents=True, exist_ok=True)

        aug_dict = self._resolve_aug_dict()
        self._augmentation_bc_method(aug_dict)
        # Update progress (For 100% completion)
        # Progress is now advanced per-augmentation inside _augmentation_method,
        # so this final tick is no longer needed (it would push completed past total).
        # self._progress_inst.update(1)

    def _augmentation_bc_method(self, aug_dict):
        """
        Evaluate every selected augmentation per class and assemble the results.

        Loads the images once, evaluates the shared clean baseline once, then
        delegates each augmentation to ``_process_augmentation`` and packages the
        combined per-class results, augmentation names, class names and size.

        Args:
            aug_dict (Dict[str, Any]): Mapping of augmentation name to instance.
        """
        image_paths: list[str] = self._data_instance.get_data()["image_directory"].tolist()
        ground_truths = self._ordered_ground_truth_df[self._ground_truth_label].tolist()
        test_dataset, test_loader = self._load_images(image_paths, ground_truths)
        # KIV: set a random seed here manually; if we want to manually set it then we'll need to change this
        seed = self._input_arguments.get('random_seed', 42)   # configurable, 42 default
        display_rng = np.random.default_rng(seed)
        display_idx = display_rng.integers(len(image_paths))

        model = self._unwrap_model()

        aug_methods = self._get_aug_methods()
        print("Augmentation methods:", aug_methods)
        if 'url' in aug_dict and 'http' in aug_dict['url']:
            handle_url_algos(aug_dict, aug_methods)

        class_names_arg = self._input_arguments.get('class_names') or None
        class_names = handle_class_names_arg(class_names_arg, model)
        print("Class names:", class_names)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Register total work units (one per augmentation that will actually run)
        # so progress advances incrementally instead of jumping 0 -> 100%.
        num_augs_to_run = sum(
            1 for aug_name in aug_dict if self._should_run_aug(aug_name, aug_methods)
        )
        self._progress_inst.add_total(num_augs_to_run)

        target_names = [class_names[k] for k in class_names]

        # The clean-severity ("None") baseline is identical for every augmentation:
        # an untouched test_loader yields the same predictions regardless of
        # aug_class. Evaluate it once here and reuse it below, saving num_augs - 1
        # full evaluate() passes over the dataset.
        baseline = self._evaluate_baseline(model, test_loader, test_dataset, target_names, device)

        combined_results = []
        for aug_name, aug_class in aug_dict.items():
            if not self._should_run_aug(aug_name, aug_methods):
                continue

            combined_results.append(self._process_augmentation(
                aug_name, aug_class, model, test_loader,
                image_paths, ground_truths, class_names, target_names,
                display_idx, device, baseline,
            ))

            # One augmentation finished; advance the progress bar.
            self._progress_inst.update(1)
            print()

        self._results = {
            "results": combined_results,
            "augmentation_names": [x["Augmentation"] for x in combined_results],
            "class_names": class_names,
            "dataset_size": len(image_paths),
        }

    def _process_augmentation(self, aug_name, aug_class, model, test_loader,
                              image_paths, ground_truths, class_names, target_names,
                              display_idx, device, baseline):
        """
        Evaluate one augmentation across its severities and assemble its results.

        Runs each severity (reusing the clean baseline for ``"None"``), saves the
        averaged confusion matrix per severity, builds the display samples, and
        renders the per-class plots via ``_sklearn_method``.

        Args:
            aug_name (str): Name of the augmentation.
            aug_class: The ``Augmentation`` instance to evaluate.
            model: Underlying model (or API URL string).
            test_loader (DataLoader): Loader over the (uncorrupted) test images.
            image_paths (List[str]): All image file paths.
            ground_truths (List): Ground-truth label per image.
            class_names (Dict[str, str]): Mapping from class index to display name.
            target_names (List[str]): Class display names aligned with matrix axes.
            display_idx (int): Index of the sample image to display.
            device (torch.device): Device the model runs on.
            baseline (tuple): ``(avg_report, avg_cm, cm_stats, dataset, y_pred)``
                for the clean pass, shared across augmentations.

        Returns:
            Dict[str, Any]: The per-augmentation results dict.
        """
        base_avg_report, base_avg_cm, base_cm_stats, base_dataset, base_y_pred = baseline

        crs = []; cms = []; cm_dict = dict(); display_scored = dict()
        severities = ["None"] + aug_class.severities
        if aug_name == "None":
            severities = ["None"]

        for severity_idx, severity_name in enumerate(severities):
            print('severity idx', severity_idx, 'severity_name', severity_name)
            if severity_name == "None":
                # Reuse the baseline computed once above; the clean pass is
                # identical across augmentations, so don't re-evaluate it.
                avg_report, avg_cm, cm_stats = base_avg_report, base_avg_cm, base_cm_stats
                scored_dataset, scored_y_pred = base_dataset, base_y_pred
            else:
                avg_report, avg_cm, scored_dataset, scored_y_pred = self._evaluate_severity(
                    model, test_loader, aug_class, severity_name, severity_idx, target_names, device
                )
                cm_stats = self._compute_cm_stats(avg_cm, target_names)

            corrupted_dir = Path(aug_name) / f"severity_{severity_name}"

            # The display image/prediction come from the same scored pass
            # (shuffle=False, so display_idx aligns), avoiding a second
            # corrupt+inference pass per severity.
            display_scored[str(severity_name)] = (scored_dataset, scored_y_pred[display_idx])

            cm_path, cm_path1 = self._save_cm_path(avg_cm, target_names, corrupted_dir)
            cm_dict[str(severity_name)] = [
                str(Path(cm_path).relative_to(self._output_folder)),
                str(Path(cm_path1).relative_to(self._output_folder)),
            ]
            crs.append(avg_report); cms.append(cm_stats)

        display_info = self._build_display_info(
            aug_name, aug_class, image_paths, ground_truths,
            class_names, display_idx, display_scored,
        )
        path_dict = self._sklearn_method(crs, cms, severities, class_names, Path(aug_name), aug_name)
        return {
            "Augmentation": aug_name,
            "display_info": display_info,
            "classification_report": crs,
            "conf_matrix": cms,
            "plot_paths": path_dict,
            "confusion_matrix": cm_dict,
        }

    def _evaluate_baseline(self, model, test_loader, test_dataset, target_names, device):
        """
        Evaluate the clean (uncorrupted) pass once for reuse across augmentations.

        Args:
            model: Underlying model (or API URL string).
            test_loader (DataLoader): Loader over the uncorrupted test images.
            test_dataset (Dataset): The dataset behind ``test_loader`` (for display).
            target_names (List[str]): Class display names aligned with matrix axes.
            device (torch.device): Device the model runs on.

        Returns:
            tuple: ``(avg_report, avg_cm, cm_stats, dataset, y_pred)`` for the
                clean pass, ready to reuse as the ``"None"`` severity.
        """
        _, y_pred, y_true = evaluate(model, test_loader, device)
        report = classification_report(
            y_true, y_pred,
            labels=list(range(len(target_names))),
            target_names=target_names,
            output_dict=True,
            zero_division=np.nan,
        )
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(target_names))))
        avg_report = average_all_reports([report])
        avg_cm = np.mean([cm], axis=0)
        cm_stats = self._compute_cm_stats(avg_cm, target_names)
        return avg_report, avg_cm, cm_stats, test_dataset, y_pred

    def _evaluate_severity(self, model, test_loader, aug_class, severity_name,
                           severity_idx, target_names, device):
        """
        Evaluate one non-clean severity, averaging over epochs.

        Corrupts the loader at ``severity_name`` for each epoch (seeded so runs are
        reproducible), scores the model, and averages the per-epoch report and
        confusion matrix. The last epoch's dataset and predictions are returned for
        building the display sample without a second corrupt+inference pass.

        Args:
            model: Underlying model (or API URL string).
            test_loader (DataLoader): Loader over the uncorrupted test images.
            aug_class: The ``Augmentation`` instance providing the corruption.
            severity_name (str): Severity label to apply.
            severity_idx (int): Position of this severity (used to seed epochs).
            target_names (List[str]): Class display names aligned with matrix axes.
            device (torch.device): Device the model runs on.

        Returns:
            tuple: ``(avg_report, avg_cm, dataset, y_pred)`` — averaged report and
                confusion matrix, plus the last epoch's scored dataset and preds.
        """
        num_epochs = self._input_arguments.get('num_epochs') or 1
        num_epochs = 1 if num_epochs is None else num_epochs
        num_epochs = 1 if aug_class.deterministic else num_epochs

        all_reports = []; all_cm = []
        corrupted_loader = None; y_pred = None
        for i in range(num_epochs):
            print("NUMEPOCHS", num_epochs)
            seed = 1000 * severity_idx + i
            aug_class.set_seed(seed)

            corrupted_loader = aug_class.corr_func_dataloader(test_loader, severity_name)
            _, y_pred, y_true = evaluate(model, corrupted_loader, device)
            report = classification_report(
                y_true, y_pred,
                labels=list(range(len(target_names))),
                target_names=target_names,
                output_dict=True,
                zero_division=np.nan,
            )
            cm = confusion_matrix(y_true, y_pred, labels=list(range(len(target_names))))
            all_reports.append(report); all_cm.append(cm)

        avg_report = average_all_reports(all_reports)
        avg_cm = np.mean(all_cm, axis=0)
        # Display the last epoch's realization: its corrupted dataset and the
        # predictions the model actually produced on it.
        return avg_report, avg_cm, corrupted_loader.dataset, y_pred

    def _build_display_info(self, aug_name, aug_class, image_paths, ground_truths,
                            class_names, display_idx, display_scored):
        """
        Build the per-severity display samples for one augmentation.

        For the chosen sample image at each severity (plus ``"None"``), saves the
        exact corrupted image the model scored and records the saved path with the
        ground-truth and predicted class names, reusing the scored predictions
        from ``display_scored`` rather than re-running inference.

        Args:
            aug_name (str): Name of the augmentation.
            aug_class: The ``Augmentation`` instance providing the severity list.
            image_paths (List[str]): All image file paths.
            ground_truths (List): Ground-truth label per image.
            class_names (Dict[str, str]): Mapping from class index to display name.
            display_idx (int): Index of the sample image to display.
            display_scored (dict): Maps severity label to
                ``(dataset, prediction_at_display_idx)`` from the scored pass.

        Returns:
            Dict[str, List[str]]: Mapping of severity (as string) to
                ``[relative_image_path, ground_truth_name, predicted_name]``.
        """
        display_info = dict()
        severities = ["None"] + aug_class.severities
        if aug_name == "None":
            severities = ["None"]
        for severity in severities:
            corrupted_dir = Path(aug_name) / f"severity_{severity}"
            dataset, prediction = display_scored[str(severity)]
            display_info[str(severity)] = self._display_row_from_scored_batch(
                dataset,
                display_idx,
                prediction,
                corrupted_dir,
                image_paths[display_idx],
                ground_truths[display_idx],
                class_names,
            )
        return display_info

    def _compute_cm_stats(self, avg_cm, target_names):
        """
        Derive per-class TP/FP/FN/TN from an averaged confusion matrix.

        Computes per-class counts from ``avg_cm``; an all-zero matrix (no
        samples) yields zeroed stats for every class rather than division noise.

        Args:
            avg_cm (np.ndarray): Square confusion matrix (possibly averaged over
                epochs), rows = true labels, columns = predicted labels.
            target_names (list[str]): Class names aligned with the matrix axes.

        Returns:
            Dict[str, Dict[str, float]]: Mapping of class name to its TP, FP, FN
                and TN counts as floats.
        """
        TP = np.diag(avg_cm); FP = avg_cm.sum(axis=0) - TP; FN = avg_cm.sum(axis=1) - TP
        TN = avg_cm.sum() - (TP + FP + FN)  # ; N = avg_cm.sum()
        if avg_cm.sum() == 0:
            return {
                label: {
                    "TP": 0.0, "FP": 0.0,
                    "FN": 0.0, "TN": 0.0
                } for label in target_names
            }
        return {
            label: {
                "TP": float(TP[i]), "FP": float(FP[i]),
                "FN": float(FN[i]), "TN": float(TN[i])
            } for i, label in enumerate(target_names)
        }

    def _sklearn_method(self, data, data2, severities, class_names, subfolder_name, aug_name):
        """
        Bulk of the visualization plotting for metrics, confusion matrix, and population stats.

        Args:
            data: data of classification reports
            data2: data of confusion matrices
            severities: severities
            class_names: class names
            subfolder_name: destination for saved artifacts
            aug_name: augmentation name
        
        Returns:
            path_dict: dictionary of paths of saved artifacts
        """
        plt.rcParams.update({'font.size': 18})

        save_dir0 = self._save_folder / subfolder_name
        save_dir0.mkdir(parents=True, exist_ok=True)
        save_dir = save_dir0 / "figures"
        save_dir.mkdir(parents=True, exist_ok=True)
        big_df = None; rows = []

        for severity, cr, cm in zip(severities, data, data2):

            df = pd.DataFrame.from_dict(cr).T
            # print(df.index.value_counts())
            df['severity'] = severity
            df['class'] = df.index

            for cl,stats in cm.items():
                row = {"severity": severity, "class": cl}
                row.update(stats); rows.append(row)
            if big_df is None:
                big_df = df 
            else:
                big_df = pd.concat([big_df, df])

        cm_df = pd.DataFrame(rows)
        combined_df = pd.merge(big_df, cm_df, on=['class', 'severity'])
        combined_df['preds_population'] = combined_df['TP'] + combined_df['FP']
        combined_df['actual_population'] = combined_df['TP'] + combined_df['FN']
        print("combined_df")
        print(combined_df.head())

        path_dict = {}
        temp = {}

        for i in class_names:
            ax_path, ax_path1, ax_path2 = self._matplotlib_class_images(combined_df, class_names, i, save_dir, aug_name)
            html_path, html_path1, html_path2 = self._plotly_class_images(combined_df, class_names, i, save_dir, aug_name)

            temp[class_names[i]] = [
                str(ax_path.relative_to(self._output_folder)), 
                str(ax_path1.relative_to(self._output_folder)), 
                str(ax_path2.relative_to(self._output_folder)),
                str(html_path.relative_to(self._output_folder)), 
                str(html_path1.relative_to(self._output_folder)), 
                str(html_path2.relative_to(self._output_folder))
            ]

        path_dict['class_plot'] = temp

        plot_df = combined_df.pivot(index='severity', columns='class', values='preds_population')
        plt.figure(figsize=(16,9))

        colors = plt.cm.jet(np.linspace(0,1,len(class_names)))

        bottom = None 
        for c in class_names:
            if bottom is None:
                plt.bar(plot_df.index, plot_df[class_names[c]], label=class_names[c], color=colors[int(c)])
                bottom = plot_df[class_names[c]].values
            else:
                plt.bar(plot_df.index, plot_df[class_names[c]], label=class_names[c], color=colors[int(c)], bottom=bottom)
                bottom = bottom + plot_df[class_names[c]].values 

        plt.xlabel('severity', fontsize=20); plt.ylabel('fraction of all samples predicted', fontsize=20)
        plt.xticks(fontsize=18 , rotation=45); plt.yticks(fontsize=18)
        plt.legend(title='class', bbox_to_anchor=(1.02,1), loc='upper left', fontsize=18)
        plt.title(f'{aug_name}: Predictions and Label Proportions per class vs Augmentation Severity', fontsize=24, pad=30)
        plt.subplots_adjust(left=0.15, right=0.8, top=0.88, bottom=0.3)
        plt_path = save_dir / "all_classes_proportions_barchart.png"
        plt.savefig(plt_path, bbox_inches="tight")
        path_dict['matplotlib_image_path'] = str(plt_path.relative_to(self._output_folder))

        long_df = plot_df.reset_index().melt(id_vars='severity', var_name='class', value_name='pred_frac')

        print(long_df.head())
        fx = px.bar(
            long_df, x='severity', y='pred_frac', color='class', 
            title=f'{aug_name}: Predictions and Label Proportions per class vs Augmentation Severity', 
            color_discrete_sequence=px.colors.sample_colorscale("Jet", [i/(len(class_names)-1) for i in range(len(class_names))])
        )
        fx.update_layout(barmode='stack', yaxis_title='fraction of all samples predicted', xaxis_title='severity')
        fx_path = save_dir / "all_classes_proportions_plotly_barchart.html"
        fx.write_html(fx_path)
        path_dict['plotly_image_path'] = str(fx_path.relative_to(self._output_folder))
        return path_dict

    def _prep_class_df(self, combined_df, class_names, i):
        """
        Slice the combined dataframe to one class and prepare it for plotting.

        Returns the raw per-class slice (with NaNs intact for axis-range and
        marker decisions), a copy with plotting columns NaN-filled to 0 for line
        stability, and the per-group NaN masks keyed by ``metrics``/``cm``/``pop``.

        Args:
            combined_df (pd.DataFrame): Merged metrics + CM-stats + populations.
            class_names (Dict[str, str]): Mapping from class index to display name.
            i (str): Class index key into ``class_names``.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]: The raw
                slice, the NaN-filled plotting slice, and the NaN masks.
        """
        sub_df = combined_df[combined_df['class'] == class_names[i]].copy().reset_index(drop=True)
        nan_masks = {
            'metrics': sub_df[self._METRIC_COLS].isna(),
            'cm': sub_df[self._CM_COLS].isna(),
            'pop': sub_df[self._POP_COLS].isna(),
        }
        plot_df = sub_df.copy()
        plot_df[self._METRIC_COLS] = plot_df[self._METRIC_COLS].fillna(0)
        plot_df[self._CM_COLS] = plot_df[self._CM_COLS].fillna(0)
        plot_df[self._POP_COLS] = plot_df[self._POP_COLS].fillna(0)
        return sub_df, plot_df, nan_masks

    def _class_plot_specs(self, aug_name, class_name):
        """
        Describe the three per-class plots (metrics / CM-stats / populations).

        Each spec drives both backends: which columns to plot, axis label, title,
        y-axis strategy (``fixed`` [-0.1, 1.1] for metrics vs ``auto`` for the
        others), whether to draw NaN markers, and the output filename per backend.

        Args:
            aug_name (str): Name of the augmentation (for titles).
            class_name (str): Display name of the class (for titles/filenames).

        Returns:
            List[Dict[str, Any]]: One spec dict per plot, in output order.
        """
        return [
            {
                'key': 'metrics', 'cols': self._METRIC_COLS, 'ylabel': 'metric',
                'ylim': 'fixed', 'nan_markers': True,
                'title': f"Sklearn report statistics for {aug_name}, class = {class_name}",
                'mpl_file': f"sklearn_figure_class_{class_name}.png",
                'html_file': f"sklearn_figure_class_{class_name}.html",
            },
            {
                'key': 'cm', 'cols': self._CM_COLS, 'ylabel': 'metric',
                'ylim': 'auto', 'nan_markers': False,
                'title': f"Confusion matrix stats for {aug_name}, class = {class_name}",
                'mpl_file': f"cm_figure_class_{class_name}.png",
                'html_file': f"cm_figure_class_{class_name}.html",
            },
            {
                'key': 'pop', 'cols': self._POP_COLS, 'ylabel': 'populations',
                'ylim': 'auto', 'nan_markers': True,
                'title': f"Raw class populations for {aug_name}, class = {class_name}",
                'mpl_file': f"sklearn_figure_class_{class_name}_popns.png",
                'html_file': f"sklearn_figure_class_{class_name}_popns.html",
            },
        ]

    def _matplotlib_class_images(self, combined_df, class_names, i, save_dir, aug_name):
        """
        Render the three per-class matplotlib plots for one class.

        Args:
            combined_df (pd.DataFrame): dataframe for combined stuff
            class_names (Dict[str, str]): Mapping from class index to display name.
            i (str): class name key
            save_dir (Path): Sub-path under the save folder to write the image into.
            aug_name (str): augmentation method name
        
        Returns:
            Tuple[Path, Path, Path]: Saved metrics, CM-stats and population paths.
        """
        sub_df, plot_df, nan_masks = self._prep_class_df(combined_df, class_names, i)
        specs = self._class_plot_specs(aug_name, class_names[i])
        paths = []
        for spec in specs:
            ax = plot_df.plot(x='severity', y=spec['cols'])
            line_colors = [line.get_color() for line in ax.get_lines()]

            ax.set_xlabel('severity', fontsize=20)
            ax.set_ylabel(spec['ylabel'], fontsize=20)
            ax.figure.set_size_inches(16, 9)
            ax.set_title(spec['title'], fontsize=24)
            ax.tick_params(axis='x', labelsize=20, labelrotation=45)
            ax.tick_params(axis='y', labelsize=20)

            if spec['ylim'] == 'fixed':
                ax.set_ylim(-0.1, 1.1)
            else:
                y_min = sub_df[spec['cols']].min().min()
                y_max = sub_df[spec['cols']].max().max()
                if np.isfinite(y_min) and np.isfinite(y_max) and y_min < y_max:
                    ax.set_ylim(-5, y_max + 5)

            if spec['nan_markers']:
                mask_df = nan_masks[spec['key']]
                severity_order = plot_df['severity'].tolist()  # ['None', 'sigma_1.50', ...]
                pos_map = {v: idx for idx, v in enumerate(severity_order)}

                # Mark where NaNs existed (a small 'x' at y=0) per column.
                for col_idx, col in enumerate(spec['cols']):
                    mask = mask_df[col].values
                    x_pos = [pos_map[label] for label in sub_df.loc[mask, 'severity']]
                    ax.scatter(x_pos, np.zeros(mask.sum()), marker='x',
                               color=line_colors[col_idx], alpha=0.8)

                nan_handles = [
                    mlines.Line2D([], [], color=line_colors[col_idx], marker='x',
                                  linestyle='None', markersize=10, label=f"x {col} (NaN)")
                    for col_idx, col in enumerate(spec['cols'])
                    if mask_df[col].values.sum() > 0
                ]
                plt.tight_layout()
                handles, _ = ax.get_legend_handles_labels()
                ax.legend(handles=handles + nan_handles, fontsize=16)
            else:
                plt.tight_layout()

            path = save_dir / spec['mpl_file']
            ax.figure.savefig(path)
            plt.close(ax.figure)
            paths.append(path)

        return tuple(paths)

    def _plotly_class_images(self, combined_df, class_names, i, save_dir, aug_name):
        """
        Render the three per-class plotly plots for one class.

        Args:
            combined_df (pd.DataFrame): dataframe for combined stuff
            class_names (Dict[str, str]): Mapping from class index to display name.
            i (str): class name key
            save_dir (Path): Sub-path under the save folder to write the image into.
            aug_name (str): augmentation method name
        
        Returns:
            Tuple[Path, Path, Path]: Saved metrics, CM-stats and population paths.
        """
        sub_df, plot_df, nan_masks = self._prep_class_df(combined_df, class_names, i)
        specs = self._class_plot_specs(aug_name, class_names[i])
        paths = []
        for spec in specs:
            fig = px.line(plot_df, x="severity", y=spec['cols'], markers=True,
                          title=spec['title'])

            if spec['nan_markers']:
                mask_df = nan_masks[spec['key']]
                trace_colors = [trace.line.color for trace in fig.data]
                for idx, col in enumerate(spec['cols']):
                    mask = mask_df[col].values
                    if mask.sum() == 0:
                        continue
                    fig.add_scatter(
                        x=plot_df.loc[mask, "severity"],
                        y=plot_df.loc[mask, col],
                        mode="markers",
                        marker=dict(symbol="x", size=10, color=trace_colors[idx]),
                        name=f"x {col} (NaN)",
                        showlegend=True,
                    )

            fig.update_layout(
                width=1600, height=900,
                xaxis_title="severity", yaxis_title=spec['ylabel'],
                font=dict(size=20), title_font_size=24,
            )

            if spec['ylim'] == 'fixed':
                fig.update_yaxes(range=[-0.1, 1.1])
            else:
                y_min = sub_df[spec['cols']].min().min()
                y_max = sub_df[spec['cols']].max().max()
                if np.isfinite(y_min) and np.isfinite(y_max) and y_min < y_max:
                    fig.update_yaxes(range=[-5, y_max + 5])

            path = save_dir / spec['html_file']
            fig.write_html(str(path))
            paths.append(path)

        return tuple(paths)

    def _save_cm_path(self, avg_cm, target_names,  corrupted_dir):
        """
        Save the confusion matrix as matplotlib and plotly images.

        Args:
            avg_cm (np.ndarray): Confusion Matrix (averaged over epochs) to be done
            target_names (List[str]): Class display names aligned with matrix axes.
            corrupted_dir (Path): Sub-path under the save folder to write the image into.

        Returns:
            tuple:
                save_path (Path): matplotlib png path
                save_path1 (Path): plotly html path
        """
        n_classes = len(target_names)
        fig_size = max(8, n_classes * 1.5)

        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=avg_cm,
            display_labels=target_names
        )

        disp.plot(
            ax=ax,
            cmap="Blues",
            colorbar=True,
            values_format=".2f"  # since it's averaged (float)
        )
        for text in disp.text_.ravel():
            text.set_fontsize(8)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        ax.set_title("Average Confusion Matrix")

        plt.tight_layout()
        save_dir = self._save_folder / corrupted_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "avg_confusion_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved to {save_path} [matplotlib]")

        fig = go.Figure(
            data=go.Heatmap(
                z=avg_cm,
                x=target_names,
                y=target_names,
                colorscale="Blues",
                colorbar=dict(title="Value"),
                text=np.round(avg_cm, 2),
                texttemplate="%{text}",
                textfont={"size": 12}
            )
        )
        fig.update_layout(
            title="Average Confusion Matrix",
            width=max(600, n_classes * 80),
            height=max(600, n_classes * 80),
            xaxis_title="Predicted label",
            yaxis_title="True label",
        )
        # Rotate x-axis labels
        fig.update_xaxes(tickangle=45)

        save_dir = self._save_folder / corrupted_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path1 = save_dir / "avg_confusion_matrix.html"
        fig.write_html(str(save_path1))

        print(f"Saved to {save_path1} [plotly]")
        return save_path , save_path1
