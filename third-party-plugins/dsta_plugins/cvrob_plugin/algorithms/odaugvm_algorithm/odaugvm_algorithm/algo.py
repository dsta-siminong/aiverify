import logging
from pathlib import Path, PurePath
from typing import Dict, List, Tuple, Union, Callable, Any, Optional
import copy
import shutil
import os

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
from PIL import Image
import inspect
import numpy as np
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, TensorDataset
from .cvrob_util import evaluate_detection, triplets, augmentation_gradient_det, handle_class_names_arg, DetectionDataset
from .augmentations_class import make_augmentation_dict, custom_parameter_change
from pathlib import Path
import pandas as pd
from PIL import ImageDraw, ImageFont
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
    _supported_algorithm_model_type: List = [ModelType.CLASSIFICATION, ModelType.DETECTION]

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
        df = self._normalize_columns(df)
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

        self._augmentation_method(aug_dict)

        # Update progress (For 100% completion)
        self._progress_inst.update(1)

    def _augmentation_method(self, aug_dict):
        if "_model" in dir(self._model_instance):
            model = self._model_instance._model
        elif "_pipeline" in dir(self._model_instance):
            model = self._model_instance._pipeline
        else:
            raise ValueError("idk what the", type(self._model_instance),"model instance is supposed to be ", dir(self._model_instance))

        combined_results = []; gradients = []; first_drops = []

        aug_methods = self._input_arguments.get('aug_methods') or 'all'
        aug_methods = [x.strip() for x in aug_methods.split(",") if x.strip()]
        print("Augmentation methods:", aug_methods)

        class_names_arg = self._input_arguments['class_names'] or None 
        class_names = handle_class_names_arg(class_names_arg, model)
        print("Class names:", class_names)
        self._ordered_ground_truth = self._resolve_class_ids(self._ordered_ground_truth, class_names)

        image_paths: list[str] = self._data_instance.get_data()["image_directory"].tolist()
        ground_truths = self._ordered_ground_truth
        test_dataset, test_loader = self._load_images_objdet(image_paths, ground_truths)
        #KIV: set a random seed here manually; if we want to manually set it then we'll need to change this
        np.random.seed(42)
        display_idx = np.random.choice(len(image_paths))
        output_results = dict()

        for aug_name, aug_class in aug_dict.items():
            
            if aug_name not in aug_methods and aug_methods != ["all"]:
                continue
            individual_results = dict() 
            individual_results.update({"Augmentation": aug_name})

            display_info = dict()
            aug_dir =  self._output_folder / aug_name
            os.makedirs(aug_dir, exist_ok=True)
            num_epochs = self._input_arguments.get('num_epochs') or 1

            #Main mAP/performance evaluation method
            gradient, maps, fig_path = augmentation_gradient_det(
                model, test_loader, None, aug_class, 'matplotlib', aug_dir, num_epochs, self._iou_thres
            )
            first_drop = maps[0] - maps[1]
            severities = ["None"] + aug_class.severities

            for severity_idx, severity in enumerate(severities):

                corrupted_dir = Path(aug_name) / f"severity{severity}"
                display_image = self._get_one_corrupted_image_direct(image_paths, ground_truths, aug_class, severity, display_idx)
                image_path = self._save_one_image(display_image, str(corrupted_dir), display_idx)
                image = torch.tensor(display_image).unsqueeze(0).float()

                model.eval()
                with torch.no_grad():
                    outputs = model(image)
                pred = outputs[0]

                prediction = {
                    "boxes": pred["boxes"].cpu().numpy().tolist(),
                    "labels": pred["labels"].cpu().numpy().tolist(),
                    "scores": pred["scores"].cpu().numpy().tolist(),
                }
                ground_truth = ground_truths[display_idx]

                image_path2, drawn_prediction = self._save_image_with_predictions(
                    image=display_image,
                    prediction=prediction,
                    gt_boxes=ground_truth,        # list of {"bbox": [...], "label": ...}
                    class_names=class_names,
                    subfolder_name=str(corrupted_dir),
                    idx=display_idx,
                    score_threshold=self._score_thres
                )

                random_display = [
                    str(Path(image_path).relative_to(self._output_folder)),
                    ground_truth,#gt_classes,
                    drawn_prediction,#pred_classes,
                    str(Path(image_path2).relative_to(self._output_folder)),
                ]
                display_info.update({str(severity): random_display})
            print(maps, first_drop)
            maps_dict = {k:v for k,v in zip(severities, maps)}
            print(aug_name, 'augmentation method gradient:', gradient)
            individual_results.update(
                {"display_info": display_info, 
                "maps": maps_dict, 
                "fig_img": str(fig_path.relative_to(self._output_folder))}
            )
            combined_results.append(individual_results)
            gradients.append(gradient)
            first_drops.append(first_drop)

            print()

        output_results.update({
            "results": combined_results,
            "gradients": gradients,
            "first_drops": first_drops,
            "augmentation_names": [x["Augmentation"] for x in combined_results],
            "dataset_size": len(image_paths),
            "class_names": class_names
        })

        self._results = output_results

    def _normalize_columns(self, df):
        """
        Normalize supported ground-truth CSV schemas to a common internal
        schema: file_name, x_min, y_min, x_max, y_max, class_raw

        `class_raw` is left untouched here — it may hold numeric ids (AI
        Verify schema) or string class names (TF Object Detection schema).
        Resolving it to a numeric class_id is handled separately by
        _resolve_class_ids.

        Supports:
        - AI Verify schema: file_name, x_min, y_min, x_max, y_max, class_id
        - TF Object Detection API schema: filename, width, height, class,
            xmin, ymin, xmax, ymax
        """
        cols = {c.lower().strip(): c for c in df.columns}
        present = set(cols.keys())

        schema_aiv = {"file_name", "x_min", "y_min", "x_max", "y_max", "class_id"}
        schema_tfod = {"filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"}

        if schema_aiv.issubset(present):
            rename_map = {
                cols["file_name"]: "file_name",
                cols["x_min"]: "x_min",
                cols["y_min"]: "y_min",
                cols["x_max"]: "x_max",
                cols["y_max"]: "y_max",
                cols["class_id"]: "class_raw",
            }
            return df.rename(columns=rename_map)

        if schema_tfod.issubset(present):
            rename_map = {
                cols["filename"]: "file_name",
                cols["xmin"]: "x_min",
                cols["ymin"]: "y_min",
                cols["xmax"]: "x_max",
                cols["ymax"]: "y_max",
                cols["class"]: "class_raw",
            }
            return df.rename(columns=rename_map)

        raise ValueError(
            "Unrecognized ground-truth CSV schema. Expected columns matching "
            f"either {sorted(schema_aiv)} or {sorted(schema_tfod)}, got "
            f"{sorted(present)}."
        )

    def _build_detection_gt(self, df):
        gt_dict = {}

        for _, row in df.iterrows():
            fname = row["file_name"]
            bbox = [row["x_min"], row["y_min"], row["x_max"], row["y_max"]]
            label = row["class_raw"]  # not yet resolved to a numeric id

            gt_dict.setdefault(fname, []).append({
                "bbox": bbox,
                "label": label,
            })

        return gt_dict

    def _resolve_class_ids(self, ordered_ground_truth, class_names: dict):
        """
        Resolve each annotation's `label` (class_raw) to a numeric class_id,
        in place across the per-image ground truth structure.

        ordered_ground_truth: list (one entry per image) of lists of
            {"bbox": [...], "label": <raw class value>} dicts.
        class_names: dict mapping class id as a string to class name, e.g.
            {'0': 'class_0', '1': 'class_1', '2': 'class_2', '3': 'class_3'}

        If a label is already numeric (or numeric-as-string, e.g. '2'), it's
        used directly as class_id — no lookup needed. Otherwise it's treated
        as a class name and mapped back to its id via class_names.
        """
        name_to_id = {name: int(id_str) for id_str, name in class_names.items()}

        def resolve(value):
            try:
                return int(value)
            except (ValueError, TypeError):
                pass
            if value not in name_to_id:
                raise ValueError(
                    f"Class value '{value}' is neither a numeric class_id nor "
                    f"a known class name. Known class names: {sorted(name_to_id)}"
                )
            return name_to_id[value]

        return [
            [{"bbox": ann["bbox"], "label": resolve(ann["label"])} for ann in per_image]
            for per_image in ordered_ground_truth
        ]

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
        image,
        prediction,
        gt_boxes,
        #gt_labels,
        class_names: dict,
        subfolder_name: str,
        idx: int,
        score_threshold: float = 0.5,
    ) -> tuple[str, dict]:          # <-- now returns (path, drawn_prediction)

        pred_boxes=prediction['boxes']
        pred_labels=prediction['labels']
        pred_scores=prediction['scores']
        
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
        # Build this in lockstep with the drawing loop so it can never drift
        # out of sync with what's actually rendered.
        drawn_boxes, drawn_labels, drawn_scores = [], [], []

        if pred_boxes is not None and len(pred_boxes) > 0:
            boxes_np = pred_boxes.cpu().numpy() if hasattr(pred_boxes, "cpu") else np.asarray(pred_boxes)
            for i, (box, label, score) in enumerate(zip(boxes_np, pred_labels, pred_scores)):
                if float(score) < score_threshold:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box]
                draw.rectangle([x1, y1, x2, y2], outline=(220, 30, 30), width=2)
                name = class_names.get(str(label), str(label))
                draw.text((x1, max(0, y1 - 13)), f"{name} {score:.2f}", fill=(220, 30, 30), font=font)

                drawn_boxes.append([x1, y1, x2, y2])
                drawn_labels.append(label)
                drawn_scores.append(float(score))

        pil_img.save(image_path)

        drawn_prediction = {
            "boxes": drawn_boxes,
            "labels": drawn_labels,
            "scores": drawn_scores,
        }
        return str(image_path), drawn_prediction

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