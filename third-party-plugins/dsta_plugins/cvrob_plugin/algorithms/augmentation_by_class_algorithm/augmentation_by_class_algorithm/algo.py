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
from .cvrob_util import evaluate, triplets, average_all_reports, handle_class_names_arg
from .augmentations_class import make_augmentation_dict, custom_parameter_change
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
        print('data')
        print(self._data_instance.get_data())
        file_names = [Path(i).name for i in self._data_instance.get_data()["image_directory"]]
        df: pd.DataFrame = self._ground_truth_instance.get_data()

        self._file_name_label = "file_name" #self._input_arguments["file_name_label"]
        self._ordered_ground_truth_df = df.set_index(self._file_name_label).reindex(file_names) 

        # Initialise main image directory
        if self._save_folder.exists():
            shutil.rmtree(self._save_folder)
        self._save_folder.mkdir(parents=True, exist_ok=True)

        # Apply user defined parameters to default parameters
        aug_library = self._input_arguments.get('aug_library') or "albumentations"
        aug_dict = make_augmentation_dict(aug_library)

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

    def _augmentation_bc_method(self, aug_dict):
        image_paths : list[str] = self._data_instance.get_data()["image_directory"].tolist()
        ground_truths = self._ordered_ground_truth_df[self._ground_truth_label].tolist()
        test_dataset, test_loader = self._load_images(image_paths, ground_truths)
        #KIV: set a random seed here manually; if we want to manually set it then we'll need to change this
        np.random.seed(42) 
        display_idx = np.random.choice(len(image_paths))
        output_results = dict()

        if "_model" in dir(self._model_instance):
            model = self._model_instance._model
        elif "_pipeline" in dir(self._model_instance):
            model = self._model_instance._pipeline
        else:
            raise ValueError("idk what the", type(self._model_instance),"model instance is supposed to be ", dir(self._model_instance))

        combined_results = []; combined_results2 = []

        aug_methods = self._input_arguments.get('aug_methods') or 'all'
        aug_methods = [x.strip() for x in aug_methods.split(",") if x.strip()]
        print("Augmentation methods:", aug_methods)

        class_names_arg = self._input_arguments.get('class_names') or None 
        class_names = handle_class_names_arg(class_names_arg, model)
        print("Class names:", class_names)

        labels = [k for k in class_names]
        target_names = [class_names[k] for k in class_names]

        for aug_name, aug_class in aug_dict.items():
            if aug_name not in aug_methods and aug_methods != ["all"]:
                continue

            individual_results = dict() ; display_info = dict(); cm_dict = dict() 
            crs = []; cms = []
            individual_results.update({"Augmentation": aug_name})
            severities = ["None"] + aug_class.severities
            if aug_name == "None":
                severities = ["None"]
            for severity_idx, severity_name in enumerate(severities):
                print('severity idx', severity_idx, 'severity_name', severity_name)
                num_epochs = self._input_arguments.get('num_epochs') or 1
                num_epochs = num_epochs if severity_name != "None" else 1 
                num_epochs = 1 if num_epochs is None else num_epochs
                num_epochs = 1 if aug_class.deterministic else num_epochs
                all_reports = []; all_cm = []
                for i in range(num_epochs):

                    print("NUMEPOCHS", num_epochs)
                    seed = 1000*severity_idx + i 
                    aug_class.set_seed(seed)

                    if severity_name == "None":
                        corrupted_loader = test_loader
                    else:
                        corrupted_loader = aug_class.corr_func_dataloader(test_loader, severity_name)
                    base_acc, y_pred, y_true = evaluate(model, corrupted_loader, next(model.parameters()).device)
                    report = classification_report(y_true, 
                        y_pred, 
                        labels=list(range(len(target_names))),  
                        target_names=target_names, 
                        output_dict=True, 
                        zero_division=np.nan
                    )

                    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(target_names))))
                    all_reports.append(report); all_cm.append(cm)

                avg_report = average_all_reports(all_reports)
                avg_cm = np.mean(all_cm, axis=0)
                TP = np.diag(avg_cm); FP = avg_cm.sum(axis=0) - TP; FN = avg_cm.sum(axis=1) - TP
                TN = avg_cm.sum() - (TP+FP+FN); N = avg_cm.sum()
                if avg_cm.sum() == 0:
                    cm_stats = {
                        label: {
                            "TP": 0.0, "FP": 0.0, 
                            "FN": 0.0, "TN": 0.0
                        } for label in target_names
                    }
                else:
                    cm_stats = {
                        label: {
                            "TP": float(TP[i]), "FP": float(FP[i]),
                            "FN": float(FN[i]), "TN": float(TN[i])
                        } for i, label in enumerate(target_names)
                    }

                corrupted_dir = Path(aug_name) / f"severity_{severity_name}"

                display_image = self._get_one_corrupted_image_direct(
                    image_paths[display_idx],
                    aug_class,
                    severity_name,
                )

                image_path = self._save_one_image(display_image, str(corrupted_dir), Path(str(image_paths[display_idx])).name)
                image = torch.tensor(display_image).unsqueeze(0).float()
                model = model.float()

                model.eval()
                with torch.no_grad():
                    outputs = model(image)
                    _, prediction = torch.max(outputs, 1)
                prediction = prediction.item()
                ground_truth = ground_truths[display_idx]

                random_display = [
                    str(Path(image_path).relative_to(self._output_folder)),
                    class_names[str(ground_truth)],
                    class_names[str(prediction)],
                ]
                display_info.update({str(severity_name): random_display})

                cm_path, cm_path1 = self._save_cm_path(avg_cm, target_names,  corrupted_dir)
                cm_dict.update({str(severity_name): [
                    str(Path(cm_path).relative_to(self._output_folder)),
                    str(Path(cm_path1).relative_to(self._output_folder))
                ]})
                crs.append(avg_report) ; cms.append(cm_stats)

            path_dict = self._sklearn_method(crs, cms, severities, class_names, Path(aug_name), aug_name)
            # print()
            # print("CRS")
            # print(crs)
            # print()
            individual_results.update(
                {
                    "display_info": display_info, 
                    "classification_report": crs ,
                    "conf_matrix": cms,
                    "plot_paths": path_dict,
                    "confusion_matrix": cm_dict
                }
            )

            combined_results.append(individual_results)

            print()

        output_results.update({
            "results": combined_results,
            "augmentation_names": [x["Augmentation"] for x in combined_results],
            "class_names": class_names,
            "dataset_size": len(image_paths)
        })

        self._results = output_results

    def _load_images(self, image_paths: list[str], labels) -> list[np.ndarray]:
        """
        Load a list of numpy images from file paths.

        Args:
            image_paths (list[str]): A list of image file paths

        Returns:
            np.ndarray: A list of numpy images
        """
        transform = transforms.Compose([
            transforms.Resize((240, 320)),  # H, W
            transforms.ToTensor()
        ])

        # Load all images into a tensor
        image_tensors = torch.stack([transform(Image.open(p).convert("RGB")) for p in image_paths])

        # Convert labels to tensor
        label_tensors = torch.tensor(labels, dtype=torch.long)

        # Create TensorDataset
        dataset = TensorDataset(image_tensors, label_tensors)

        # Create DataLoader
        loader = DataLoader(dataset, batch_size=128, shuffle=False)

        return dataset, loader

    def _sklearn_method(self, data, data2, severities, class_names, subfolder_name, aug_name):
        plt.rcParams.update({'font.size': 18})

        save_dir0 = self._save_folder / subfolder_name
        save_dir0.mkdir(parents=True, exist_ok=True)
        save_dir = save_dir0 / "figures"
        save_dir.mkdir(parents=True, exist_ok=True)
        big_df = None; rows = []

        for severity, cr, cm in zip(severities, data, data2):
            # print("CR")
            # pprint(cr)
            # print("CM")
            # pprint(cm)
            # print(f"Severity: {severity}")
            # print("CR keys:", list(cr.keys()))

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

    def _matplotlib_class_images(self, combined_df, class_names, i, save_dir, aug_name):
        sub_df = combined_df[combined_df['class'] == class_names[i]].copy()
        sub_df = sub_df.reset_index(drop=True)
        # =========================
        # 1. Define columns first
        # =========================
        metric_cols = ['precision', 'recall', 'f1-score']
        cm_cols = ['TP', 'FP', 'FN', 'TN']
        pop_cols = ['preds_population', 'actual_population']

        # =========================
        # 2. Preserve NaN masks (important for semantics)
        # =========================
        nan_mask_metrics = sub_df[metric_cols].isna()
        nan_mask_cm = sub_df[cm_cols].isna()
        nan_mask_pop = sub_df[pop_cols].isna()

        # =========================
        # 3. Fill NaNs ONLY for plotting stability
        # =========================
        plot_df = sub_df.copy()
        plot_df[metric_cols] = plot_df[metric_cols].fillna(0)
        plot_df[cm_cols] = plot_df[cm_cols].fillna(0)
        plot_df[pop_cols] = plot_df[pop_cols].fillna(0)

        # ==========================================================
        # 4. METRICS PLOT (precision / recall / f1-score)
        # ==========================================================
        ax = plot_df.plot(x='severity', y=metric_cols)
        line_colors = [line.get_color() for line in ax.get_lines()]

        y_min = plot_df[metric_cols].min().min()
        y_max = plot_df[metric_cols].max().max()

        ax.set_xlabel('severity', fontsize=20)
        ax.set_ylabel('metric', fontsize=20)
        ax.figure.set_size_inches(16, 9)
        ax.set_title(
            f"Sklearn report statistics for {aug_name}, class = {class_names[i]}",
            fontsize=24
        )

        ax.set_ylim(-0.1, 1.1)

        # if np.isfinite(y_min) and np.isfinite(y_max) and y_min < y_max:
        #     ax.set_ylim(y_min - 0.1, y_max + 0.1)

        # Optional: mark where NaNs existed (tiny visual cue)

        severity_order = plot_df['severity'].tolist()  # ['None', 'sigma_1.50', ...]
        pos_map = {v: i for i, v in enumerate(severity_order)}

        for col_idx, col in enumerate(metric_cols):
            mask = nan_mask_metrics[col].values
            x_labels = sub_df.loc[mask, 'severity']
            x_pos = [pos_map[label] for label in x_labels]  # integer positions
            y = np.zeros(mask.sum())

            ax.scatter(
                x_pos,
                y,
                marker='x',
                color=line_colors[col_idx],
                alpha=0.8
            )

        nan_handles = []
        for col_idx, col in enumerate(metric_cols):
            mask = nan_mask_metrics[col].values

            if mask.sum() == 0:
                continue

            handle = mlines.Line2D(
                [],
                [],
                color=line_colors[col_idx],
                marker='x',
                linestyle='None',
                markersize=10,
                label=f"x {col} (NaN)"
            )

            nan_handles.append(handle)

        ax_path = save_dir / f"sklearn_figure_class_{class_names[i]}.png"
        ax.tick_params(axis='x', labelsize=20, labelrotation=45)
        ax.tick_params(axis='y', labelsize=20)
        plt.tight_layout()

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles + nan_handles, fontsize=16)
        ax.figure.savefig(ax_path)
        plt.close(ax.figure)

        # ==========================================================
        # 5. CONFUSION MATRIX PLOT (TP / FP / FN / TN)
        # ==========================================================
        ax1 = plot_df.plot(x='severity', y=cm_cols)

        y_min = sub_df[cm_cols].min().min()
        y_max = sub_df[cm_cols].max().max()

        ax1.set_xlabel('severity', fontsize=20)
        ax1.set_ylabel('metric', fontsize=20)
        ax1.figure.set_size_inches(16, 9)
        ax1.set_title(
            f"Confusion matrix stats for {aug_name}, class = {class_names[i]}",
            fontsize=24
        )

        if np.isfinite(y_min) and np.isfinite(y_max) and y_min < y_max:
            ax1.set_ylim(-5, y_max + 5)

        ax_path1 = save_dir / f"cm_figure_class_{class_names[i]}.png"
        ax1.tick_params(axis='x', labelsize=20, labelrotation=45)
        ax1.tick_params(axis='y', labelsize=20)
        plt.tight_layout()
        ax1.figure.savefig(ax_path1)
        plt.close(ax1.figure)

        # ==========================================================
        # 6. POPULATION PLOT (safe, usually no NaNs here)
        # ==========================================================
        ax2 = plot_df.plot(x='severity', y=pop_cols)

        y_min = sub_df[pop_cols].min().min()
        y_max = sub_df[pop_cols].max().max()

        ax2.set_xlabel('severity', fontsize=20)
        ax2.set_ylabel('populations', fontsize=20)
        ax2.figure.set_size_inches(16, 9)
        ax2.set_title(
            f"Raw class populations for {aug_name}, class = {class_names[i]}",
            fontsize=24
        )

        if np.isfinite(y_min) and np.isfinite(y_max) and y_min < y_max:
            ax2.set_ylim(-5, y_max + 5)

        severity_order = plot_df['severity'].tolist()  # ['None', 'sigma_1.50', ...]
        pos_map = {v: i for i, v in enumerate(severity_order)}

        for col_idx, col in enumerate(pop_cols):
            mask = nan_mask_pop[col].values
            x_labels = sub_df.loc[mask, 'severity']
            x_pos = [pos_map[label] for label in x_labels]  # integer positions
            y = np.zeros(mask.sum())

            ax2.scatter(
                x_pos,
                y,
                marker='x',
                color=line_colors[col_idx],
                alpha=0.8
            )
            
        nan_handles = []
        for col_idx, col in enumerate(pop_cols):
            mask = nan_mask_pop[col].values

            if mask.sum() == 0:
                continue

            handle = mlines.Line2D(
                [],
                [],
                color=line_colors[col_idx],
                marker='x',
                linestyle='None',
                markersize=10,
                label=f"x {col} (NaN)"
            )

            nan_handles.append(handle)

        ax_path2 = save_dir / f"sklearn_figure_class_{class_names[i]}_popns.png"
        ax2.tick_params(axis='x', labelsize=20, labelrotation=45)
        ax2.tick_params(axis='y', labelsize=20)
        plt.tight_layout()
        handles, labels = ax2.get_legend_handles_labels()
        ax2.legend(handles=handles + nan_handles, fontsize=16)
        ax2.figure.savefig(ax_path2)
        plt.close(ax2.figure)

        return ax_path, ax_path1, ax_path2

    def _plotly_class_images(self, combined_df, class_names, i, save_dir, aug_name):
        sub_df = combined_df[combined_df['class'] == class_names[i]].copy()

        # =========================
        # 1. Define columns FIRST
        # =========================
        metric_cols = ['precision', 'recall', 'f1-score']
        cm_cols = ['TP', 'FP', 'FN', 'TN']
        pop_cols = ['preds_population', 'actual_population']

        # =========================
        # 2. Fill ONLY for plotting stability
        # =========================
        nan_mask_metrics = sub_df[metric_cols].isna()
        nan_mask_pop = sub_df[pop_cols].isna()
        plot_df = sub_df.copy()
        plot_df[metric_cols] = plot_df[metric_cols].fillna(0)
        plot_df[cm_cols] = plot_df[cm_cols].fillna(0)
        plot_df[pop_cols] = plot_df[pop_cols].fillna(0)

        # =========================
        # 3. Metrics plot
        # =========================

        fig = px.line(
            plot_df,
            x="severity",
            y=metric_cols,
            markers=True,
            title=f"Sklearn report statistics for {aug_name}, class = {class_names[i]}"
        )

        # capture trace colors for reuse
        trace_colors = [trace.line.color for trace in fig.data]

        for idx, col in enumerate(metric_cols):
            mask = nan_mask_metrics[col].values

            if mask.sum() == 0:
                continue

            fig.add_scatter(
                x=plot_df.loc[mask, "severity"],
                y=plot_df.loc[mask, col],
                mode="markers",
                marker=dict(
                    symbol="x",
                    size=10,
                    color=trace_colors[idx]
                ),
                name=f"x {col} (NaN)",
                showlegend=True
            )


        fig.update_layout(
            width=1600,
            height=900,
            xaxis_title="severity",
            yaxis_title="metric",
            font=dict(size=20),
            title_font_size=24
        )

        y_min = plot_df[metric_cols].min().min()
        y_max = plot_df[metric_cols].max().max()

        fig.update_yaxes(range=[-0.1, 1.1])
        # if np.isfinite(y_min) and np.isfinite(y_max) and y_min < y_max:
        #     fig.update_yaxes(range=[y_min - 0.1, y_max + 0.1])

        html_path = save_dir / f"sklearn_figure_class_{class_names[i]}.html"
        fig.write_html(str(html_path))

        # =========================
        # 4. Confusion matrix plot
        # =========================
        fig = px.line(
            plot_df,
            x="severity",
            y=cm_cols,
            markers=True,
            title=f"Confusion matrix stats for {aug_name}, class = {class_names[i]}"
        )

        fig.update_layout(
            width=1600,
            height=900,
            xaxis_title="severity",
            yaxis_title="metric",
            font=dict(size=20),
            title_font_size=24
        )

        y_min = sub_df[cm_cols].min().min()
        y_max = sub_df[cm_cols].max().max()

        if np.isfinite(y_min) and np.isfinite(y_max) and y_min < y_max:
            fig.update_yaxes(range=[-5, y_max + 5])

        html_path1 = save_dir / f"cm_figure_class_{class_names[i]}.html"
        fig.write_html(str(html_path1))

        # =========================
        # 5. Population plot
        # =========================
        fig = px.line(
            plot_df,
            x="severity",
            y=pop_cols,
            markers=True,
            title=f"Raw class populations for {aug_name}, class = {class_names[i]}"
        )

        # capture trace colors for reuse
        trace_colors = [trace.line.color for trace in fig.data]

        for idx, col in enumerate(pop_cols):
            mask = nan_mask_pop[col].values

            if mask.sum() == 0:
                continue

            fig.add_scatter(
                x=plot_df.loc[mask, "severity"],
                y=plot_df.loc[mask, col],
                mode="markers",
                marker=dict(
                    symbol="x",
                    size=10,
                    color=trace_colors[idx]
                ),
                name=f"x {col} (NaN)",
                showlegend=True
            )

        fig.update_layout(
            width=1600,
            height=900,
            xaxis_title="severity",
            yaxis_title="populations",
            font=dict(size=20),
            title_font_size=24
        )

        y_min = sub_df[pop_cols].min().min()
        y_max = sub_df[pop_cols].max().max()

        if np.isfinite(y_min) and np.isfinite(y_max) and y_min < y_max:
            fig.update_yaxes(range=[-5, y_max + 5])

        html_path2 = save_dir / f"sklearn_figure_class_{class_names[i]}_popns.html"
        fig.write_html(str(html_path2))

        return html_path, html_path1, html_path2

    def _save_cm_path(self, avg_cm, target_names,  corrupted_dir):

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

    def _save_one_image(self, image: np.ndarray, subfolder_name, image_path_original):
        save_dir = self._save_folder / subfolder_name
        save_dir.mkdir(parents=True, exist_ok=True)

        image_path = save_dir / image_path_original
        image = np.transpose(image, (1, 2, 0))
        Image.fromarray((image * 255.0).astype(np.uint8)).save(image_path)

        return str(image_path)

    # def _get_one_corrupted_image(self, testloader, aug_class, severity, target_idx):
    #     current_idx = 0

    #     for images, labels in testloader:
    #         batch_size = images.shape[0]

    #         # Check if target is inside this batch
    #         if current_idx + batch_size > target_idx:
    #             local_idx = target_idx - current_idx

    #             images_np = (images * 255).byte().numpy().transpose(0, 2, 3, 1)

    #             if aug_class.name == "None" or severity == "None":
    #                 corrupted = images_np
    #             else:
    #                 corrupted = aug_class.corr_func_arr(images_np, severity)

    #             corrupted = corrupted.transpose(0, 3, 1, 2) / 255.0

    #             return corrupted[local_idx]  # <-- only one image

    #         current_idx += batch_size

    def _get_one_corrupted_image_direct(
        self,
        image_path: str,
        aug_class,         # Augmentation instance
        severity: str,     # e.g. "severity_1" or "None"
        resize: tuple[int, int] = (240, 320),  # (H, W) — match _load_images
    ) -> np.ndarray:
        """
        Fetch and corrupt a single image directly from disk.
        Matches the pipeline of _load_images + _get_one_corrupted_image exactly,
        but without loading any other images.

        Returns: CHW float32 numpy array in [0, 1]
        """
        # 1. Load and resize — identical to _load_images transform
        image = Image.open(image_path).convert("RGB")
        if resize is not None:
            image = image.resize((resize[1], resize[0]), Image.BILINEAR)  # PIL takes (W, H)

        # 2. To uint8 HWC numpy — skip the float tensor round-trip entirely
        image_np = np.array(image, dtype=np.uint8)  # HWC uint8

        # 3. Corrupt
        if aug_class.name == "None" or severity == "None":
            corrupted = image_np  # HWC uint8
        else:
            corrupted = aug_class.corr_func_arr(
                image_np[None],   # needs batch dim: (1, H, W, C)
                severity
            )[0]                  # back to (H, W, C)

        # 4. Normalise to CHW float32 [0, 1]
        return corrupted.transpose(2, 0, 1).astype(np.float32) / 255.0