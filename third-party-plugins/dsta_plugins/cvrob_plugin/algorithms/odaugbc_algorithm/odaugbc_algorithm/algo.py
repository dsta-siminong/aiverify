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

# from . import augmentations
import numpy as np
from PIL import Image
import inspect
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, TensorDataset
from .cvrob_util import *
from .augmentations_class import make_augmentation_dict, custom_parameter_change
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import pandas as pd 
import json
import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.graph_objects as go
from pprint import pprint

from .pycocotools_fdet.coco import COCO
from .pycocotools_fdet.cocoeval import COCOeval
# =====================================================================================
# NOTE:
# 1. Check that you have installed the aiverify_test_engine latest package.
# 2. Check that you have run tests/install_core_plugins_requirements.sh to install all the
#    requirements required by the core plugins (serializers, data, models).
#    Alternatively, you may install the plugins that you require by installing the
#    requirements individually.
# 3. Do not modify the class name, else the plugin cannot be read by the system.
# =====================================================================================
class Plugin(IAlgorithm):
    """
    # TODO: Update the plugin description below
    The Plugin(Augmentation v Metric Algorithm) class specifies methods in generating results for algorithm
    """

    # Some information on plugin
    _name: str = "Augmentation v Metric Algorithm"
    _description: str = "This algorithm shows the relationship between certain augmentations and the model's performance metrics"
    _version: str = "0.1.0"
    _metadata: PluginMetadata = PluginMetadata(_name, _description, _version)
    _plugin_type: PluginType = PluginType.ALGORITHM
    _requires_ground_truth: bool = True
    _supported_algorithm_model_type: List = [ModelType.CLASSIFICATION]

    @staticmethod
    def get_metadata() -> PluginMetadata:
        """
        A method to return the metadata for this plugin

        Returns:
            PluginMetadata: Metadata of this plugin
        """
        return Plugin._metadata

    @staticmethod
    def get_plugin_type() -> PluginType:
        """
        A method to return the type for this plugin

        Returns:
            PluginType: Type of this plugin
        """
        return Plugin._plugin_type

    def __init__(
        self,
        data_instance_and_serializer: Tuple[IData, ISerializer],
        model_instance_and_serializer: Tuple[IModel, ISerializer],
        ground_truth_instance_and_serializer: Tuple[IData, ISerializer],
        initial_data_instance: Union[IData, None],
        initial_model_instance: Union[IModel, IPipeline, None],
        **kwargs,
    ):

        self._initial_data_instance = initial_data_instance
        self._initial_model_instance = initial_model_instance

        # Look for kwargs values for log_instance, progress_callback and base path
        self._logger = kwargs.get("logger", None)
        self._progress_inst = SimpleProgress(
            1, 0, kwargs.get("progress_callback", None)
        )

        # Check if data and model are tuples and if the tuples contain 2 items
        if (
            not isinstance(data_instance_and_serializer, Tuple)
            or len(data_instance_and_serializer) != 2
        ):
            self.add_to_log(
                logging.ERROR,
                f"The algorithm has failed data validation: {data_instance_and_serializer}",
            )
            raise RuntimeError("The algorithm has failed data validation")

        if (
            not isinstance(model_instance_and_serializer, Tuple)
            or len(model_instance_and_serializer) != 2
        ):
            self.add_to_log(
                logging.ERROR,
                f"The algorithm has failed model validation: {model_instance_and_serializer}",
            )
            raise RuntimeError("The algorithm has failed model validation")

        self._data_instance = data_instance_and_serializer[0]
        self._model_instance = model_instance_and_serializer[0]
        self._model_type = kwargs.get("model_type")
        self._ground_truth_label = kwargs.get("ground_truth")

        if Plugin._requires_ground_truth:
            # Check if ground truth instance is tuple and if the tuple contains 2 items
            if (
                not isinstance(ground_truth_instance_and_serializer, Tuple)
                or len(ground_truth_instance_and_serializer) != 2
            ):
                self.add_to_log(
                    logging.ERROR,
                    f"The algorithm has failed ground truth data validation: \
                        {ground_truth_instance_and_serializer}",
                )
                raise RuntimeError(
                    "The algorithm has failed ground truth data validation"
                )
            self._requires_ground_truth = True
            self._ground_truth_instance = ground_truth_instance_and_serializer[0]
            self._ground_truth_serializer = ground_truth_instance_and_serializer[1]
            self._ground_truth = kwargs.get("ground_truth")

        else:
            self._ground_truth_instance = None
            self._ground_truth = ""

        self._base_path = kwargs.get("project_base_path", Path().absolute())

        # Other variables
        self._data = None
        self._results = {"results": [0]}

        # Perform setup for this plug-in
        self.setup()

        # Write all output to the output folder
        self._output_folder = Path.cwd() / "output"
        self._output_folder.mkdir(parents=True, exist_ok=True)
        self._save_folder = self._output_folder / "images"

        # TODO: Update the input json schema in input.schema.json
        # Algorithm input schema defined in input.schema.json
        # By defining the input schema, it allows the front-end to know what algorithm input params is
        # required by this plugin. This allows this algorithm plug-in to receive the arguments values it requires.
        
        current_file_dir = Path(__file__).parent
        
        self._input_schema = load_schema_file(
            str(current_file_dir / "input.schema.json")
        )

        # TODO: Update the output json schema in output.schema.json
        # Algorithm output schema defined in output.schema.json
        # By defining the output schema, this plug-in validates the result with the output schema.
        # This allows the result to be validated against the schema before passing it to the front-end for display.
        self._output_schema = load_schema_file(
            str(current_file_dir / "output.schema.json")
        )

        # Retrieve the input parameters defined in the input schema and store them
        self._input_arguments = dict()
        for key in self._input_schema.get("properties").keys():
            self._input_arguments.update({key: kwargs.get(key)})

        # Perform validation on input argument schema
        if not validate_json(self._input_arguments, self._input_schema):
            self.add_to_log(
                logging.ERROR,
                f"The algorithm has failed input schema validation. \
                    The input must adhere to the schema in input.schema.json: {self._input_arguments}",
            )
            raise RuntimeError("The algorithm has failed input schema validation. \
                               The input must adhere to the schema in input.schema.json")

    def add_to_log(self, log_level: int, log_message: str) -> None:
        """
        A helper method to log messages to store events occurred

        Args:
            log_level (int): The logging level
            log_message (str): The logging message
        """
        if self._logger is not None:
            if not isinstance(log_level, int) or not isinstance(log_message, str):
                raise RuntimeError(
                    "The algorithm has invalid log level or message. The log level should be a \
                        logging level(i.e. logging.DEBUG) and log message should be in String format"
                )        
        if self._logger is not None:
            if log_level is logging.DEBUG:
                self._logger.debug(log_message)
            elif log_level is logging.INFO:
                self._logger.info(log_message)
            elif log_level is logging.WARNING:
                self._logger.warning(log_message)
            elif log_level is logging.ERROR:
                self._logger.error(log_message)
            elif log_level is logging.CRITICAL:
                self._logger.critical(log_message)
            else:
                pass  # Invalid log level
        else:
            pass  # No log instance

    def setup(self) -> None:
        """
        A method to perform setup for this algorithm plugin
        """
        # Perform validation on logger
        if self._logger:
            if not isinstance(self._logger, logging.Logger):
                raise RuntimeError(
                    "The algorithm has failed to set up logger. The logger type is invalid"
                )

        # Perform validation on model type
        if self._model_type not in Plugin._supported_algorithm_model_type:
            self.add_to_log(
                logging.ERROR,
                f"The algorithm has failed validation for model type: {self._model_type}",
            )
            raise RuntimeError("The algorithm has failed validation for model type")

        # Perform validation on data instance
        if not isinstance(self._data_instance, IData):
            self.add_to_log(
                logging.ERROR,
                f"The algorithm has failed data validation: {self._data_instance}",
            )
            raise RuntimeError("The algorithm has failed data validation")

        # Perform validation on model instance
        if not isinstance(self._model_instance, IModel) and not isinstance(
            self._model_instance, IPipeline
        ):
            self.add_to_log(
                logging.ERROR,
                f"The algorithm has failed model validation: {self._model_instance}",
            )
            raise RuntimeError("The algorithm has failed model validation")

        # Perform validation on ground truth instance
        if self._requires_ground_truth:
            if not isinstance(self._ground_truth_instance, IData):
                self.add_to_log(
                    logging.ERROR,
                    f"The algorithm has failed ground truth data validation: {self._ground_truth_instance}",
                )
                raise RuntimeError(
                    "The algorithm has failed ground truth data validation"
                )

            # Perform validation on ground truth header
            if not isinstance(self._ground_truth, str):
                self.add_to_log(
                    logging.ERROR,
                    "The algorithm has failed ground truth header validation. \
                    Header must be in String and must be present in the dataset: {self._ground_truth}",
                )
                raise RuntimeError(
                    "The algorithm has failed ground truth header validation. \
                    Header must be in String and must be present in the dataset"
                )

        # Perform validation on progress_inst
        if self._progress_inst:
            if not isinstance(self._progress_inst, SimpleProgress):
                raise RuntimeError(
                    "The algorithm has failed validation for the progress bar"
                )

        # Perform validation on project_base_path
        if not isinstance(self._base_path, PurePath):
            self.add_to_log(
                logging.ERROR,
                "The algorithm has failed validation for the project path. \
                Ensure that the project path is a valid path: {self._base_path}",
            )
            raise RuntimeError(
                "The algorithm has failed validation for the project path. \
                Ensure that the project path is a valid path"
            )

        # Perform validation on metadata
        if not isinstance(self._metadata, PluginMetadata):
            self.add_to_log(
                logging.ERROR,
                f"The algorithm has failed validation for its metadata: {Plugin._metadata}",
            )
            raise RuntimeError("The algorithm has failed validation for its metadata")

        # Perform validation on plugin type
        if not isinstance(self._plugin_type, PluginType):
            self.add_to_log(
                logging.ERROR,
                "The algorithm has failed validation for its plugin type. \
                Ensure that PluginType is PluginType.ALGORITHM: {Plugin._plugin_type}",
            )
            raise RuntimeError(
                "The algorithm has failed validation for its plugin type. \
                Ensure that PluginType is PluginType.ALGORITHM"
            )
        # Perform logging
        self.add_to_log(logging.INFO, "Setup completed")

    def get_progress(self) -> int:
        """
        A method to return the current progress for this plugin

        Returns:
            int: Completion Progress
        """
        return self._progress_inst.get_progress()

    def get_results(self) -> Dict:
        """
        A method to return generated results for this plugin

        Returns:
            Dict: The results to be returned for display
        """
        return self._results

    def generate(self) -> None:
        """
        A method to generate the algorithm results with the provided data, model, ground truth information.
        """
        # Retrieve data information
        self._data = self._data_instance.get_data()
        #make ground truth
        file_names = [Path(i).name for i in self._data_instance.get_data()["image_directory"]]
        df: pd.DataFrame = self._ground_truth_instance.get_data()
        self._df = df
        self._gt_dict = self._build_detection_gt(df)

        self._ordered_ground_truth = [
            self._gt_dict.get(fname, []) for fname in file_names
        ]

        # Initialise main image directory
        if self._save_folder.exists():
            shutil.rmtree(self._save_folder)
        self._save_folder.mkdir(parents=True, exist_ok=True)

        # Apply user defined parameters to default parameters
        aug_library = self._input_arguments.get('aug_library') or "albumentations"
        aug_dict = make_augmentation_dict(aug_library)
        self._iou_thres = self._input_arguments.get('iou_thres') or 0.5
        self._score_thres = self._input_arguments.get('score_thres') or 0.5

        custom_parameters = None
        try:
            custom_parameters = self._input_arguments.get('custom_parameters') or None
            custom_parameters = triplets(custom_parameters)
            for sublist in custom_parameters:
                aug_name, param_name, parameters_in_string = sublist
                aug_dict = custom_parameter_change(aug_dict, aug_name, param_name, parameters_in_string)
        except Exception as e:
            print("No custom parameter_change")
            print(f"Custom parameters exception: {e} , {custom_parameters}")
            print()
        
        self._augmentation_bc_method(aug_dict)
        # Update progress (For 100% completion)
        self._progress_inst.update(1)

    def _resolve_model(self):
        """Return the underlying torch model from the model instance."""
        if "_model" in dir(self._model_instance):
            return self._model_instance._model
        elif "_pipeline" in dir(self._model_instance):
            return self._model_instance._pipeline
        raise ValueError(
            f"Unknown model instance type {type(self._model_instance)}: {dir(self._model_instance)}"
        )

    def _resolve_aug_methods(self):
        """Return the list of augmentation method names to run (or ['all'])."""
        aug_methods = self._input_arguments.get('aug_methods') or 'all'
        return [x.strip() for x in aug_methods.split(",") if x.strip()]

    def _resolve_class_names(self, model):
        """Return the class_names dict from input arguments or inferred from the model."""
        class_names_arg = self._input_arguments.get('class_names') or None
        return handle_class_names_arg(class_names_arg, model)

    # def _run_severity_epochs(
    #     self,
    #     model,
    #     test_loader,
    #     aug_class,
    #     severity_name: str,
    #     severity_idx: int,
    #     num_epochs: int,
    #     class_names: dict,
    # ):
    #     """
    #     Run ``num_epochs`` evaluation passes for one (aug, severity) combination
    #     and return averaged detection statistics.

    #     Returns:
    #         avg_stats  - per-class metrics averaged over epochs
    #         avg_matrix - detection-matching matrix averaged over epochs
    #         avg_map  - scalar mAP@50 averaged over valid epochs (None if none valid)
    #     """
    #     all_stats = []; all_matrices = []; all_maps = []

    #     for i in range(num_epochs):
    #         print("NUMEPOCHS", num_epochs)
    #         seed = 1000 * severity_idx + i
    #         aug_class.set_seed(seed)

    #         if severity_name == "None":
    #             corrupted_loader = test_loader
    #         else:
    #             corrupted_loader = aug_class.corr_func_dataloader(test_loader, severity_name)

    #         det_stats = evaluate_detection_detailed(
    #             model,
    #             corrupted_loader,
    #             None,
    #             class_names,
    #             iou_thresh=self._iou_thres,
    #             score_thresh=self._score_thres,
    #         )

    #         all_stats.append(det_stats["per_class"])
    #         all_matrices.append(det_stats["matrix"])
    #         all_maps.append(det_stats["map"])

    #     avg_stats = average_detection_stats(all_stats)
    #     avg_matrix = np.mean(all_matrices, axis=0)
    #     avg_map = (
    #         float(np.mean([x for x in all_maps if x >= 0]))
    #         if any(x >= 0 for x in all_maps)
    #         else None
    #     )
    #     return avg_stats, avg_matrix, avg_map

    def _get_display_info_for_severity(
        self,
        model,
        image_paths: list,
        ground_truths: list,
        aug_class,
        severity_name: str,
        display_idx: int,
        class_names: dict,
        corrupted_dir: Path,
    ):
        """
        Obtain the display image for one severity, run inference on it, save both
        the plain image and the annotated overlay, and return the display-info list.

        Returns:
            list: [rel_image_path, ground_truth, prediction_dict, rel_overlay_path]
        """
        display_image = self._get_one_corrupted_image_direct(
            image_paths, ground_truths, aug_class, severity_name, display_idx
        )

        image_path = self._save_one_image(display_image, str(corrupted_dir), display_idx)

        image_tensor = torch.tensor(display_image).unsqueeze(0).float()
        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor)
        output = outputs[0]

        prediction = {
            "boxes":  output["boxes"].cpu().numpy().tolist(),
            "labels": output["labels"].cpu().numpy().tolist(),
            "scores": output["scores"].cpu().numpy().tolist(),
        }

        ground_truth = ground_truths[display_idx]

        image_path2 = self._save_image_with_predictions(
            image=display_image,
            pred_boxes=prediction['boxes'],
            pred_labels=prediction['labels'],
            pred_scores=prediction['scores'],
            gt_boxes=ground_truth,
            gt_labels=[obj["label"] for obj in ground_truth],
            class_names=class_names,
            subfolder_name=str(corrupted_dir),
            idx=display_idx,
            score_threshold=self._score_thres
        )

        return [
            str(Path(image_path).relative_to(self._output_folder)),
            ground_truth,
            prediction,
            str(Path(image_path2).relative_to(self._output_folder)),
        ]

    def _augmentation_bc_method(self, aug_dict):
        image_paths: list[str] = self._data_instance.get_data()["image_directory"].tolist()
        ground_truths = self._ordered_ground_truth
        test_dataset, test_loader = self._load_images_objdet(image_paths, ground_truths)

        # KIV: set a random seed here manually
        np.random.seed(42)
        display_idx = np.random.choice(len(image_paths))

        model = self._resolve_model()
        aug_methods = self._resolve_aug_methods()
        class_names = self._resolve_class_names(model)

        print("Augmentation methods:", aug_methods)
        print("Class names:", class_names)

        combined_results = []

        gt_json = 'ground_truths.json'
        create_coco_gt(image_paths, self._df, class_names, gt_json)

        for aug_name, aug_class in aug_dict.items():
            if aug_name not in aug_methods and aug_methods != ["all"]:
                continue

            individual_results = {"Augmentation": aug_name}
            display_info = {}
            cm_dict = {}
            coco_dict = {}
            coco_graph_dict = {}
            crs = []
            cms = []

            severities = ["None"] + aug_class.severities
            if aug_name == "None":
                severities = ["None"]

            for severity_idx, severity_name in enumerate(severities):
                print('severity idx', severity_idx, 'severity_name', severity_name)
                corrupted_dir = Path(aug_name) / f"severity_{severity_name}"

                num_epochs = self._input_arguments.get('num_epochs') or 1
                num_epochs = num_epochs if severity_name != "None" else 1
                num_epochs = 1 if num_epochs is None else num_epochs
                num_epochs = 1 if aug_class.deterministic else num_epochs

                save_dir_coco = self._save_folder / corrupted_dir
                save_dir_coco.mkdir(parents=True, exist_ok=True)
                avg_stats, avg_matrix, avg_map, avg_summary, coco_imgs = self._run_severity_epochs_coco(
                    model, test_loader, aug_class, severity_name, severity_idx,
                    num_epochs, class_names, 
                    gt_json, image_paths, self._iou_thres, self._score_thres, save_dir_coco
                )

                display_info[str(severity_name)] = self._get_display_info_for_severity(
                    model, image_paths, ground_truths, aug_class, severity_name,
                    display_idx, class_names, corrupted_dir,
                )

                save_path, save_path_html = self._save_detection_matrix_path(
                    avg_matrix, class_names, corrupted_dir
                )
                cm_dict[str(severity_name)] = [
                    str(Path(save_path).relative_to(self._output_folder)),
                    str(Path(save_path_html).relative_to(self._output_folder)),
                ]

                coco_graph_dict[str(severity_name)] = [str(x.relative_to(self._output_folder)) for x in coco_imgs]
                coco_dict[str(severity_name)] = avg_summary

                crs.append({"map": avg_map, **avg_stats})
                cms.append(avg_matrix)

            path_dict = self._detection_method(crs, severities, class_names, Path(aug_name), aug_name)
            individual_results.update({
                "display_info": display_info,
                "classification_report": crs,
                "conf_matrix": cms,
                "plot_paths": path_dict,
                "confusion_matrix": cm_dict,
                "coco_graphs": coco_graph_dict,
                "coco_summary": coco_dict
            })

            combined_results.append(individual_results)
            print()

        self._results = {
            "results": combined_results,
            "augmentation_names": [x["Augmentation"] for x in combined_results],
            "class_names": class_names,
            "dataset_size": len(image_paths),
        }

    def _build_combined_df(self, data: list, severities: list) -> "pd.DataFrame":
        """
        Convert per-severity detection stats into a flat DataFrame with columns:
        severity, class, precision, recall, f1_score, TP, FP, FN, support, map.
        """
        rows = []
        for severity, stats in zip(severities, data):
            map = stats.get("map", None)
            if "per_class" in stats:
                per_class = stats["per_class"]
            else:
                per_class = {k: v for k, v in stats.items() if k != "map"}
            per_class = normalize_per_class(per_class)

            for class_name, metrics in per_class.items():
                rows.append({
                    "severity":  severity,
                    "class":     class_name,
                    "precision": metrics["precision"],
                    "recall":    metrics["recall"],
                    "f1_score":  metrics["f1_score"],
                    "TP":        metrics["TP"],
                    "FP":        metrics["FP"],
                    "FN":        metrics["FN"],
                    "support":   metrics["support"],
                    "map":    map,
                })
        df = pd.DataFrame(rows)
        df['pred_population']   = df['TP'] + df['FP']
        df['actual_population'] = df['TP'] + df['FN']
        return df

    def _plot_class_metrics(
        self,
        class_df: "pd.DataFrame",
        plot_df: "pd.DataFrame",
        nan_masks: dict,
        save_dir: Path,
        aug_name: str,
        class_name: str,
    ):
        """
        Plot precision / recall / f1 over severity for one class (matplotlib + plotly).

        Returns:
            (png_path, html_path)
        """
        metric_cols = ['precision', 'recall', 'f1_score', 'map']
        severity_order = plot_df['severity'].tolist()
        pos_map = {v: i for i, v in enumerate(severity_order)}

        # --- matplotlib ---
        fig, ax = plt.subplots(figsize=(10, 6))
        line_colors = {}
        for col, label in zip(metric_cols, ['precision', 'recall', 'f1', 'map']):
            line, = ax.plot(plot_df['severity'], plot_df[col], label=label)
            line_colors[col] = line.get_color()

        for col, label in zip(metric_cols, ['precision', 'recall', 'f1', 'map']):
            mask = nan_masks[col]
            if mask.any():
                x_pos = [pos_map[s] for s in plot_df.loc[mask, 'severity']]
                ax.scatter(x_pos, [0] * len(x_pos), marker='x',
                           color=line_colors[col], alpha=0.8, zorder=5,
                           label=f'{label} (NaN\u21920)')

        ax.set_title(f"{aug_name} - {class_name}")
        ax.set_xlabel("severity")
        ax.set_ylabel("score")
        ax.legend()
        ax.set_ylim(-0.1, 1.1)
        plt.xticks(rotation=45)

        png_path = save_dir / f"{class_name}_metrics.png"
        plt.savefig(png_path, bbox_inches="tight")
        plt.close()

        # --- plotly ---
        fig_html = px.line(
            plot_df, x='severity', y=metric_cols,
            title=f"{aug_name} - {class_name}",
            labels={'value': 'metric', 'variable': 'metric'}
        )
        plotly_colors = px.colors.qualitative.Plotly
        col_color_map = {col: plotly_colors[i] for i, col in enumerate(metric_cols)}

        for col, label in zip(metric_cols, ['precision', 'recall', 'f1', 'map']):
            mask = nan_masks[col]
            if mask.any():
                nan_severities = plot_df.loc[mask, 'severity'].tolist()
                fig_html.add_trace(go.Scatter(
                    x=nan_severities, y=[0] * len(nan_severities),
                    mode='markers',
                    marker=dict(symbol='x', size=12, color=col_color_map[col],
                                line=dict(width=2)),
                    name=f'{label} (NaN\u21920)', showlegend=True
                ))

        fig_html.update_layout(
            width=1600, height=900, xaxis_title="severity", yaxis_title="metric",
            font=dict(size=20), title_font_size=24
        )
        fig_html.update_yaxes(range=[-0.1, 1.1])
        html_path = save_dir / f"{class_name}_metrics.html"
        fig_html.write_html(str(html_path))

        return png_path, html_path

    def _plot_class_counts(
        self,
        class_df: "pd.DataFrame",
        save_dir: Path,
        aug_name: str,
        class_name: str,
    ):
        """
        Plot TP / FP / FN counts over severity for one class (matplotlib + plotly).
        TN is undefined for object detection and is omitted.

        Returns:
            (png_path_cm, html_path_cm)
        """
        count_cols = ['TP', 'FP', 'FN']
        count_colors = {'TP': '#2ca02c', 'FP': '#ff7f0e', 'FN': '#d62728'}
        count_plot_df = class_df[['severity'] + count_cols].fillna(0)

        # --- matplotlib ---
        fig, ax = plt.subplots(figsize=(10, 6))
        for col in count_cols:
            ax.plot(count_plot_df['severity'], count_plot_df[col],
                    label=col, color=count_colors[col], marker='o')

        ax.set_title(f"{aug_name} - {class_name} (TP / FP / FN counts)")
        ax.set_xlabel("severity")
        ax.set_ylabel("count")
        ax.legend()
        plt.xticks(rotation=45)

        png_path_cm = save_dir / f"{class_name}_counts.png"
        plt.savefig(png_path_cm, bbox_inches="tight")
        plt.close()

        # --- plotly ---
        fig_counts = go.Figure()
        for col in count_cols:
            fig_counts.add_trace(go.Scatter(
                x=count_plot_df['severity'], y=count_plot_df[col],
                mode='lines+markers', name=col,
                line=dict(color=count_colors[col])
            ))
        fig_counts.update_layout(
            title=f"{aug_name} - {class_name} (TP / FP / FN counts)",
            width=1600, height=900, xaxis_title="severity", yaxis_title="count",
            font=dict(size=20), title_font_size=24
        )
        html_path_cm = save_dir / f"{class_name}_counts.html"
        fig_counts.write_html(str(html_path_cm))

        return png_path_cm, html_path_cm

    def _plot_class_population(
        self,
        class_df: "pd.DataFrame",
        save_dir: Path,
        aug_name: str,
        class_name: str,
    ):
        """
        Plot actual_population (TP+FN = GT boxes) vs pred_population (TP+FP = predicted
        boxes) over severity for one class (matplotlib + plotly).

        Returns:
            (png_path_pop, html_path_pop)
        """
        pop_df = class_df[['severity', 'TP', 'FP', 'FN']].fillna(0).copy()
        pop_df['actual_population'] = pop_df['TP'] + pop_df['FN']
        pop_df['pred_population']   = pop_df['TP'] + pop_df['FP']
        y_max = pop_df[['actual_population', 'pred_population']].max().max()

        # --- matplotlib ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(pop_df['severity'], pop_df['actual_population'],
                label='actual population (GT boxes)', color='steelblue', marker='o')
        ax.plot(pop_df['severity'], pop_df['pred_population'],
                label='pred population (pred boxes)', color='darkorange',
                marker='s', linestyle='--')
        ax.set_title(f"{aug_name} - {class_name} (population)")
        ax.set_xlabel("severity")
        ax.set_ylabel("count")
        ax.legend()
        plt.xticks(rotation=45)
        ax.set_ylim(-5, y_max + 5)

        png_path_pop = save_dir / f"{class_name}_population.png"
        plt.savefig(png_path_pop, bbox_inches="tight")
        plt.close()

        # --- plotly ---
        fig_pop = go.Figure()
        fig_pop.add_trace(go.Scatter(
            x=pop_df['severity'], y=pop_df['actual_population'],
            mode='lines+markers', name='actual population (GT boxes)',
            line=dict(color='steelblue')
        ))
        fig_pop.add_trace(go.Scatter(
            x=pop_df['severity'], y=pop_df['pred_population'],
            mode='lines+markers', name='pred population (pred boxes)',
            line=dict(color='darkorange', dash='dash')
        ))
        fig_pop.update_layout(
            title=f"{aug_name} - {class_name} (population)",
            width=1600, height=900, xaxis_title="severity", yaxis_title="count",
            font=dict(size=20), title_font_size=24
        )
        fig_pop.update_yaxes(range=[5, y_max + 5])
        html_path_pop = save_dir / f"{class_name}_population.html"
        fig_pop.write_html(str(html_path_pop))

        return png_path_pop, html_path_pop

    def _plot_map(
        self,
        combined_df: "pd.DataFrame",
        save_dir: Path,
        aug_name: str,
    ):
        """
        Plot mAP@50 over severity (matplotlib + plotly).

        Returns:
            (map_png, map_html)
        """
        map_df = combined_df[["severity", "map"]].drop_duplicates().reset_index(drop=True)
        map_nan_mask = map_df['map'].isna()
        plot_map_df = map_df.copy()
        plot_map_df['map'] = plot_map_df['map'].fillna(0)

        map_severity_order = plot_map_df['severity'].tolist()
        map_pos_map = {v: i for i, v in enumerate(map_severity_order)}

        # --- matplotlib ---
        fig, ax = plt.subplots(figsize=(10, 6))
        map_line, = ax.plot(plot_map_df['severity'], plot_map_df['map'])
        map_color = map_line.get_color()

        if map_nan_mask.any():
            x_pos = [map_pos_map[s] for s in plot_map_df.loc[map_nan_mask, 'severity']]
            ax.scatter(x_pos, [0] * len(x_pos), marker='x', color=map_color,
                       alpha=0.8, zorder=5, label='map (NaN\u21920)')

        ax.set_title(f"{aug_name} mAP@50")
        ax.set_xlabel("severity")
        ax.set_ylabel("mAP@50")
        ax.set_ylim(-0.1, 1.1)
        ax.legend()
        plt.xticks(rotation=45)

        map_png = save_dir / "map.png"
        plt.savefig(map_png, bbox_inches="tight")
        plt.close()

        # --- plotly ---
        fig_html = px.line(plot_map_df, x='severity', y='map',
                           title=f"{aug_name} mAP@50")
        if map_nan_mask.any():
            nan_severities = plot_map_df.loc[map_nan_mask, 'severity'].tolist()
            fig_html.add_trace(go.Scatter(
                x=nan_severities, y=[0] * len(nan_severities),
                mode='markers',
                marker=dict(symbol='x', size=12,
                            color=px.colors.qualitative.Plotly[0],
                            line=dict(width=2)),
                name='map (NaN\u21920)', showlegend=True
            ))

        fig_html.update_layout(
            width=1600, height=900, xaxis_title="severity", yaxis_title="map",
            font=dict(size=20), title_font_size=24
        )
        map_html = save_dir / "map.html"
        fig_html.write_html(str(map_html))

        return map_png, map_html

    def _plot_all_classes_population_barchart(
        self,
        combined_df: "pd.DataFrame",
        class_names: dict,
        save_dir: Path,
        aug_name: str,
    ):
        """
        Stacked bar chart showing pred_population (TP+FP) per foreground class across
        severities — the object-detection equivalent of the classification
        all_classes_proportions_barchart.

        Background (class_id == '0') is excluded.
        Both matplotlib (.png) and plotly (.html) versions are produced.

        Returns:
            (plt_path, fx_path)
        """
        foreground_names = [v for k, v in class_names.items() if str(k) != '0']

        # pred_population is already in combined_df (TP + FP computed upstream).
        # We need one row per (severity, class), so drop any map duplicates first.
        pop_df = (
            combined_df[combined_df['class'].isin(foreground_names)]
            [['severity', 'class', 'pred_population']]
            .drop_duplicates()
        )

        plot_df = pop_df.pivot(index='severity', columns='class', values='pred_population').fillna(0)

        # --- matplotlib ---
        fig, ax = plt.subplots(figsize=(16, 9))
        colors = plt.cm.jet(np.linspace(0, 1, len(foreground_names)))

        bottom = None
        for idx, class_name in enumerate(foreground_names):
            if class_name not in plot_df.columns:
                continue
            vals = plot_df[class_name].values
            if bottom is None:
                ax.bar(plot_df.index, vals, label=class_name, color=colors[idx])
                bottom = vals.copy()
            else:
                ax.bar(plot_df.index, vals, label=class_name, color=colors[idx], bottom=bottom)
                bottom = bottom + vals

        ax.set_xlabel('severity', fontsize=20)
        ax.set_ylabel('predicted box count', fontsize=20)
        ax.tick_params(axis='x', labelsize=18, rotation=45)
        ax.tick_params(axis='y', labelsize=18)
        ax.legend(title='class', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=18)
        ax.set_title(
            f'{aug_name}: Predicted Box Counts per Class vs Augmentation Severity',
            fontsize=24, pad=30
        )
        plt.subplots_adjust(left=0.15, right=0.8, top=0.88, bottom=0.3)

        plt_path = save_dir / "all_classes_proportions_barchart.png"
        plt.savefig(plt_path, bbox_inches="tight")
        plt.close()

        # --- plotly ---
        long_df = plot_df.reset_index().melt(
            id_vars='severity', var_name='class', value_name='pred_count'
        )
        n = len(foreground_names)
        color_scale = (
            px.colors.sample_colorscale("Jet", [i / max(n - 1, 1) for i in range(n)])
        )
        fx = px.bar(
            long_df, x='severity', y='pred_count', color='class',
            title=f'{aug_name}: Predicted Box Counts per Class vs Augmentation Severity',
            color_discrete_sequence=color_scale,
        )
        fx.update_layout(
            barmode='stack',
            yaxis_title='predicted box count',
            xaxis_title='severity',
        )
        fx_path = save_dir / "all_classes_proportions_plotly_barchart.html"
        fx.write_html(str(fx_path))

        return plt_path, fx_path

    def _detection_method(
        self,
        data,
        severities,
        class_names,
        subfolder_name,
        aug_name
    ):
        plt.rcParams.update({'font.size': 18})

        save_dir0 = self._save_folder / subfolder_name
        save_dir0.mkdir(parents=True, exist_ok=True)
        save_dir = save_dir0 / "figures"
        save_dir.mkdir(parents=True, exist_ok=True)

        combined_df = self._build_combined_df(data, severities)
        print(combined_df.head())

        metric_cols = ['precision', 'recall', 'f1_score', 'map']
        path_dict = {}
        temp = {}

        background_names = {v for k, v in class_names.items() if str(k) == '0'}

        for class_name in class_names.values():
            if class_name in background_names:
                continue

            class_df = combined_df[combined_df["class"] == class_name].reset_index(drop=True)

            nan_masks = {col: class_df[col].isna() for col in metric_cols}
            plot_df = class_df.copy()
            plot_df[metric_cols] = plot_df[metric_cols].fillna(0)

            png_path,     html_path     = self._plot_class_metrics(
                class_df, plot_df, nan_masks, save_dir, aug_name, class_name)
            png_path_cm,  html_path_cm  = self._plot_class_counts(
                class_df, save_dir, aug_name, class_name)
            png_path_pop, html_path_pop = self._plot_class_population(
                class_df, save_dir, aug_name, class_name)

            temp[class_name] = [
                str(png_path.relative_to(self._output_folder)),
                str(png_path_cm.relative_to(self._output_folder)),
                str(png_path_pop.relative_to(self._output_folder)),
                str(html_path.relative_to(self._output_folder)),
                str(html_path_cm.relative_to(self._output_folder)),
                str(html_path_pop.relative_to(self._output_folder)),
            ]

        path_dict["class_plot"] = temp

        plt_path, fx_path = self._plot_all_classes_population_barchart(
            combined_df, class_names, save_dir, aug_name
        )
        path_dict['matplotlib_image_path'] = str(plt_path.relative_to(self._output_folder))
        path_dict['plotly_image_path']     = str(fx_path.relative_to(self._output_folder))

        map_png, map_html = self._plot_map(combined_df, save_dir, aug_name)
        path_dict["map_plot"] = [
            str(map_png.relative_to(self._output_folder)),
            str(map_html.relative_to(self._output_folder)),
        ]

        return path_dict

    def _save_one_image(self, image: np.ndarray, subfolder_name: str, idx: int) -> str:

        save_dir = self._save_folder / subfolder_name
        save_dir.mkdir(parents=True, exist_ok=True)
        image_path = save_dir / f"{idx}_without_prediction.png"

        # CHW -> HWC
        image = image.transpose(1, 2, 0)
        image = image.astype(np.float32)
        # normalize safely
        if image.max() <= 1.5:
            image *= 255.0

        image = np.clip(image, 0, 255).astype(np.uint8)
        Image.fromarray(image).save(image_path)
        return str(image_path)

    # def _get_one_corrupted_image(
    #     self,
    #     testloader,
    #     aug_class,
    #     severity,
    #     target_idx
    # ):
    #     current_idx = 0

    #     for images, targets in testloader:
    #         batch_size = len(images)
    #         # target image inside this batch
    #         if current_idx + batch_size > target_idx:
    #             local_idx = target_idx - current_idx
    #             image = images[local_idx]
    #             target = targets[local_idx]

    #             image_np = (
    #                 image.mul(255)
    #                 .byte()
    #                 .cpu()
    #                 .numpy()
    #                 .transpose(1, 2, 0)
    #             )

    #             if aug_class.name == "None" or severity == "None":
    #                 corrupted_image = image_np
    #             else:
    #                 corrupted_image, _ = aug_class.corr_func_sample(
    #                     image_np,
    #                     target,
    #                     severity
    #                 )

    #             corrupted_image = (
    #                 corrupted_image
    #                 .transpose(2, 0, 1)
    #                 .astype(np.float32)
    #                 / 255.0
    #             )

    #             return corrupted_image

    #         current_idx += batch_size

    def _save_detection_matrix_path(
        self,
        det_matrix,
        class_names_dir,
        corrupted_dir
    ):
        class_names = list(class_names_dir.values())
        n_classes = len(class_names)
        fig_size = max(8, n_classes * 1.5)

        # =========================
        # MATPLOTLIB
        # =========================

        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        im = ax.imshow(det_matrix, cmap="Blues")
        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

        plt.setp(
            ax.get_xticklabels(),
            rotation=45,
            ha="right"
        )
        # text values
        for i in range(n_classes):
            for j in range(n_classes):

                ax.text(
                    j,
                    i,
                    f"{det_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8
                )

        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Ground truth class")
        ax.set_title("Detection Matching Matrix")
        fig.colorbar(im)
        plt.tight_layout()
        save_dir = self._save_folder / corrupted_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "detection_matrix.png"
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )
        plt.close(fig)
        print(f"Saved to {save_path} [matplotlib]")

        # =========================
        # PLOTLY
        # =========================

        fig = go.Figure(
            data=go.Heatmap(
                z=det_matrix,
                x=class_names,
                y=class_names,
                colorscale="Blues",
                colorbar=dict(title="Matches"),
                text=np.round(det_matrix, 2),
                texttemplate="%{text}",
                textfont={"size": 12}
            )
        )
        fig.update_layout(
            title="Detection Matching Matrix",
            width=max(600, n_classes * 80),
            height=max(600, n_classes * 80),
            xaxis_title="Predicted class",
            yaxis_title="Ground truth class",
        )
        fig.update_xaxes(tickangle=45)
        save_path_html = save_dir / "detection_matrix.html"
        fig.write_html(str(save_path_html))
        print(f"Saved to {save_path_html} [plotly]")
        return save_path, save_path_html

    def _build_detection_gt(self, df):
        gt_dict = {}

        for _, row in df.iterrows():
            fname = row["file_name"]
            bbox = [row["x_min"], row["y_min"], row["x_max"], row["y_max"]]
            label = row["class_id"]

            if fname not in gt_dict:
                gt_dict[fname] = []

            gt_dict[fname].append({
                "bbox": bbox,
                "label": label
            })

        return gt_dict

    def _load_images_objdet(self, image_paths, targets):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        dataset = DetectionDataset(image_paths, targets, transform)

        loader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
            collate_fn=self._collate_fn  # IMPORTANT
        )

        return dataset, loader

    def _collate_fn(self, batch):
        return tuple(zip(*batch))

    def _save_image_with_predictions(
        self,
        image: np.ndarray,
        pred_boxes,
        pred_labels,
        pred_scores,
        gt_boxes,
        gt_labels,
        class_names: dict,
        subfolder_name: str,
        idx: int,
        score_threshold: float = 0.5,
    ) -> str:
        """
        Overlay ground-truth boxes (green) and predicted boxes (red) on the image,
        then save it as ``{idx}_with_prediction.png`` in the same subfolder structure
        used by _save_one_image.

        Args:
            image (np.ndarray): CHW float image (values in [0, 1] or [0, 255]).
            pred_boxes: Tensor or array of shape (N, 4) with [x1, y1, x2, y2] predictions.
            pred_labels: Array of predicted label ids (length N).
            pred_scores: Array of prediction confidence scores (length N).
            gt_boxes: List of [x_min, y_min, x_max, y_max] from ground-truth objects.
            gt_labels: List of ground-truth label ids.
            class_names (dict): Mapping from str(label_id) -> class name string.
            subfolder_name (str): Sub-folder name (mirrors the one used by _save_one_image).
            idx (int): Image index, used in the filename.
            score_threshold (float): Predictions below this confidence are skipped.

        Returns:
            str: Absolute path to the saved overlay image.
        """
        from PIL import ImageDraw, ImageFont

        save_dir = self._save_folder / subfolder_name
        save_dir.mkdir(parents=True, exist_ok=True)
        image_path = save_dir / f"{idx}_with_prediction.png"

        # --- Normalise to HWC uint8 ---
        img_hwc = image.transpose(1, 2, 0).astype(np.float32)
        if img_hwc.max() <= 1.5:
            img_hwc = img_hwc * 255.0
        img_hwc = np.clip(img_hwc, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_hwc).convert("RGB")
        draw = ImageDraw.Draw(pil_img)

        # Try to load a small font; fall back to the default if unavailable.
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        except Exception:
            font = ImageFont.load_default()

        # --- Ground-truth boxes (green) ---
        for obj in (gt_boxes or []):
            bbox = obj if isinstance(obj, (list, tuple)) else obj.get("bbox", [])
            label = obj.get("label") if isinstance(obj, dict) else None
            if len(bbox) == 4:
                x1, y1, x2, y2 = [float(v) for v in bbox]
                draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=2)
                if label is not None:
                    name = class_names.get(str(label), str(label))
                    draw.text((x1, max(0, y1 - 13)), f"GT:{name}", fill=(0, 200, 0), font=font)

        # --- Predicted boxes (red) ---
        if pred_boxes is not None and len(pred_boxes) > 0:
            # Convert tensors to numpy if needed
            boxes_np = pred_boxes.cpu().numpy() if hasattr(pred_boxes, "cpu") else np.asarray(pred_boxes)
            for i, (box, label, score) in enumerate(zip(boxes_np, pred_labels, pred_scores)):
                if float(score) < score_threshold:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box]
                draw.rectangle([x1, y1, x2, y2], outline=(220, 30, 30), width=2)
                name = class_names.get(str(label), str(label))
                draw.text((x1, max(0, y1 - 13)), f"{name} {score:.2f}", fill=(220, 30, 30), font=font)

        pil_img.save(image_path)
        return str(image_path)

    def _get_one_corrupted_image_direct(self, image_paths, ground_truths, aug_class, severity, target_idx):
        image = Image.open(image_paths[target_idx]).convert("RGB")
        image_np = np.array(image).astype(np.uint8)  # HWC uint8, no full loader needed

        if aug_class.name == "None" or severity == "None":
            return image_np.transpose(2, 0, 1).astype(np.float32) / 255.0

        corrupted_image, _ = aug_class.corr_func_sample(
            image_np, ground_truths[target_idx], severity
        )
        corrupted_image = corrupted_image.astype(np.float32)
        if corrupted_image.max() > 1.5:
            corrupted_image /= 255.0
        return np.clip(corrupted_image, 0, 1).transpose(2, 0, 1)

    def _run_severity_epochs_coco(
        self,
        model,
        test_loader,
        aug_class,
        severity_name: str,
        severity_idx: int,
        num_epochs: int,
        class_names: dict,
        gt_json: str,
        image_paths: list,
        iou_threshold: float,
        score_threshold: float,
        save_dir_coco: Path,
    ):
        """
        Run ``num_epochs`` evaluation passes for one (aug, severity) combination
        and return averaged detection statistics.

        Returns:
            avg_stats  - per-class metrics averaged over epochs
            avg_matrix - detection-matching matrix averaged over epochs
            avg_map  - scalar mAP@50 averaged over valid epochs (None if none valid)
        """
        pr_json = 'predictions.json'
        all_stats = []; all_matrices = []; all_maps = []; all_summaries = []

        for i in range(num_epochs):
            print("NUMEPOCHS", num_epochs)
            seed = 1000 * severity_idx + i
            aug_class.set_seed(seed)

            if severity_name == "None":
                corrupted_loader = test_loader
            else:
                corrupted_loader = aug_class.corr_func_dataloader(test_loader, severity_name)

            det_stats = evaluate_detection_and_create_coco_predictions(
                model, corrupted_loader, None, class_names, image_paths, pr_json,
                iou_thresh=iou_threshold, score_thresh=score_threshold,
                coco_score_threshold=0.0,  # match original create_coco_predictions default
            )
            
            cocoGt = COCO(gt_json)
            cocoDt = cocoGt.loadRes(pr_json)  # initialize COCO prediction api
            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')  # initialize COCO evaluation api
            cocoEval.evaluate()
            cocoEval.accumulateFBeta()
            cocoEval.accumulate()
            summary = cocoEval.collectSummaryResults(fbeta_betas=(1, 2), fbeta_iou_thrs=(iou_threshold,))

            fbeta_filename = save_dir_coco / "fbeta_curve.png"
            cocoEval.plotFBetaCurve(fbeta_filename, betas=[1,2], iouThr=iou_threshold, average='macro')
            pr_filename = save_dir_coco / "pr_curve.png"
            cocoEval.plotPRCurve(pr_filename, average='macro')
            cocopr_filename = save_dir_coco / "cocopr_curve.png" #TODO: KIV doing this by class
            cocoEval.plotCocoPRCurve(cocopr_filename)
            per_class_report = cocoEval.generateReport()

            for k,v in det_stats['per_class'].items():
                assert k in per_class_report
                class_report = per_class_report[k]
                for k1,v1 in class_report.items():
                    det_stats['per_class'][k][k1] = v1

            all_stats.append(det_stats["per_class"])
            all_matrices.append(det_stats["matrix"])
            all_maps.append(det_stats["map"])
            all_summaries.append(summary)

        avg_stats = average_detection_stats(all_stats)
        avg_matrix = np.mean(all_matrices, axis=0)
        avg_map = (
            float(np.mean([x for x in all_maps if x >= 0]))
            if any(x >= 0 for x in all_maps)
            else None
        )
        avg_summary = average_summaries(all_summaries)
        return avg_stats, avg_matrix, avg_map, avg_summary, [fbeta_filename, pr_filename, cocopr_filename]