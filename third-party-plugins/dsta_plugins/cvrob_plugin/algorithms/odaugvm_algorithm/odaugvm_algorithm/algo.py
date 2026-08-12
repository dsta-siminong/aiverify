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
import torch
from .cvrob_util import get_prediction_from_image, augmentation_gradient_det, handle_class_names_arg
from .augmentations_class import handle_url_algos
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
from . import cvrob_algo_common

@dataclass
class _OdCtx:
    """
    Everything ``_augmentation_method`` resolves once up front and threads to its
    phase helpers, so no single method has to re-derive the setup.

    Attributes:
        model: The unwrapped torch detection model (or API URL string).
        device (torch.device): Device the model runs on.
        class_names (dict): Class index (as str) -> display name.
        image_paths (list): All image file paths.
        ground_truths (list): Per-image ground-truth annotations (class ids resolved).
        test_loader: Clean detection data loader (shuffle off).
        display_idx (int): Index of the single sample shown per severity.
        aug_methods (list): Augmentation names requested (or ``["all"]``).
        num_epochs (int): Repeats per severity for non-deterministic augmentations.
    """
    model: object
    device: object
    class_names: dict
    image_paths: list
    ground_truths: list
    test_loader: object
    display_idx: int
    aug_methods: list
    num_epochs: int

# =====================================================================================
# NOTE:
# 1. Check that you have installed the aiverify_test_engine latest package.
# 2. Check that you have run tests/install_core_plugins_requirements.sh to install all the
#    requirements required by the core plugins (serializers, data, models).
#    Alternatively, you may install the plugins that you require by installing the
#    requirements individually.
# 3. Do not modify the class name, else the plugin cannot be read by the system.
# =====================================================================================
class Plugin(cvrob_algo_common.BasePlugin):
    """
    # TODO: Update the plugin description below
    The Plugin(OD Augmentation v Metric Algorithm) class specifies methods in generating results for algorithm
    """

    # Some information on plugin
    _name: str = "OD Augmentation v Metric Algorithm"
    _description: str = "This algorithm shows the relationship between certain augmentations and the model's performance metrics"
    _version: str = "0.1.0"
    _metadata: PluginMetadata = PluginMetadata(_name, _description, _version)
    _plugin_type: PluginType = PluginType.ALGORITHM
    _requires_ground_truth: bool = True
    _supported_algorithm_model_type: List = [ModelType.CLASSIFICATION, ModelType.DETECTION]

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
        self._iou_thres = self._input_arguments.get('iou_thres') or 0.5
        self._score_thres = self._input_arguments.get('score_thres') or 0.5

        aug_dict = self._resolve_aug_dict()
        self._augmentation_method(aug_dict)

        # Update progress (For 100% completion)
        # self._progress_inst.update(1)

    def _augmentation_method(self, aug_dict):
        """
        Orchestrate the mAP-vs-severity evaluation across all requested augmentations.

        Resolves setup once, then evaluates each in-scope augmentation (mAP per
        severity, gradient, and per-severity display samples) and assembles the
        aggregate results dict into ``self._results``. Progress advances once per
        augmentation that actually runs.

        Args:
            aug_dict (Dict[str, Any]): Mapping of augmentation name to instance.
        """
        ctx = self._setup_odaugvm(aug_dict)

        # One work unit per augmentation that will actually run, so progress
        # advances incrementally instead of jumping 0 -> 100%.
        num_augs_to_run = sum(
            1 for aug_name in aug_dict if self._should_run_aug(aug_name, ctx.aug_methods)
        )
        self._progress_inst.add_total(num_augs_to_run)

        combined_results = []; gradients = []; first_drops = []
        for aug_name, aug_class in aug_dict.items():
            if not self._should_run_aug(aug_name, ctx.aug_methods):
                continue

            individual_results, gradient, first_drop = self._run_one_aug(
                ctx, aug_name, aug_class
            )
            combined_results.append(individual_results)
            gradients.append(gradient)
            first_drops.append(first_drop)

            self._progress_inst.update(1)
            print()

        self._results = {
            "results": combined_results,
            "gradients": gradients,
            "first_drops": first_drops,
            "augmentation_names": [x["Augmentation"] for x in combined_results],
            "dataset_size": len(ctx.image_paths),
            "class_names": ctx.class_names,
        }

    def _setup_odaugvm(self, aug_dict):
        """
        Resolve all inputs the evaluation needs before any augmentation runs.

        Unwraps the model, resolves class names and the URL-backed augmentations,
        picks the device, resolves ground-truth class ids, loads the clean loader,
        and chooses the single display sample.

        Args:
            aug_dict (Dict[str, Any]): Mapping of augmentation name to instance.

        Returns:
            _OdCtx: The resolved context threaded through the phase helpers.
        """
        model = self._unwrap_model()

        aug_methods = self._get_aug_methods()
        print("Augmentation methods:", aug_methods)
        if 'url' in aug_dict and 'http' in aug_dict['url']:
            handle_url_algos(aug_dict, aug_methods)

        class_names_arg = self._input_arguments['class_names'] or None
        class_names = handle_class_names_arg(class_names_arg, model)
        print("Class names:", class_names)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ground_truths = self._resolve_class_ids(self._ordered_ground_truth, class_names)
        self._ordered_ground_truth = ground_truths

        image_paths: list[str] = self._data_instance.get_data()["image_directory"].tolist()
        _, test_loader = self._load_images_objdet(image_paths, ground_truths)
        # KIV: set a random seed here manually; if we want to make it configurable
        # later we'll change this.
        np.random.seed(42)
        display_idx = np.random.choice(len(image_paths))
        num_epochs = self._input_arguments.get('num_epochs') or 1

        return _OdCtx(
            model=model,
            device=device,
            class_names=class_names,
            image_paths=image_paths,
            ground_truths=ground_truths,
            test_loader=test_loader,
            display_idx=display_idx,
            aug_methods=aug_methods,
            num_epochs=num_epochs,
        )

    def _run_one_aug(self, ctx, aug_name, aug_class):
        """
        Evaluate a single augmentation end to end.

        Runs the mAP-vs-severity pass (gradient + per-severity maps + saved
        figure), then builds the per-severity display samples.

        Args:
            ctx (_OdCtx): Resolved run context.
            aug_name (str): Name of this augmentation.
            aug_class: The ``Augmentation`` instance.

        Returns:
            Tuple[dict, float, float]: The per-augmentation results dict, the
                best-fit gradient, and the first-severity confidence drop.
        """
        aug_dir = self._output_folder / aug_name
        os.makedirs(aug_dir, exist_ok=True)

        # Main mAP/performance evaluation method
        gradient, maps, fig_path = augmentation_gradient_det(
            ctx.model, ctx.test_loader, ctx.device, aug_class,
            'matplotlib', aug_dir, ctx.num_epochs, self._iou_thres,
        )
        first_drop = maps[0] - maps[1]
        severities = ["None"] + aug_class.severities

        display_info = self._build_severity_display(ctx, aug_name, aug_class, severities)

        print(maps, first_drop)
        maps_dict = {k: v for k, v in zip(severities, maps)}
        print(aug_name, 'augmentation method gradient:', gradient)
        individual_results = {
            "Augmentation": aug_name,
            "display_info": display_info,
            "maps": maps_dict,
            "fig_img": str(fig_path.relative_to(self._output_folder)),
        }
        return individual_results, gradient, first_drop

    def _build_severity_display(self, ctx, aug_name, aug_class, severities):
        """
        Save the display sample (clean + corrupted, with/without predictions) per severity.

        For the chosen display index, corrupts the image at each severity, saves
        the plain image and a prediction-overlaid copy, and records their paths
        alongside the ground truth and drawn predictions.

        Args:
            ctx (_OdCtx): Resolved run context.
            aug_name (str): Name of this augmentation (for the save sub-path).
            aug_class: The ``Augmentation`` instance.
            severities (list): ``["None", *aug_class.severities]``.

        Returns:
            Dict[str, list]: Maps each severity (as str) to
                ``[plain_path, ground_truth, drawn_prediction, overlaid_path]``.
        """
        display_info = dict()
        display_idx = ctx.display_idx
        ground_truth = ctx.ground_truths[display_idx]

        for severity in severities:
            corrupted_dir = Path(aug_name) / f"severity{severity}"
            display_image = self._get_one_corrupted_image_direct(
                ctx.image_paths, ctx.ground_truths, aug_class, severity, display_idx
            )
            image_path = self._save_one_image(display_image, str(corrupted_dir), display_idx)
            prediction = get_prediction_from_image(ctx.model, display_image, ctx.device)

            image_path2, drawn_prediction = self._save_image_with_predictions(
                image=display_image,
                prediction=prediction,
                gt_boxes=ground_truth,        # list of {"bbox": [...], "label": ...}
                class_names=ctx.class_names,
                subfolder_name=str(corrupted_dir),
                idx=display_idx,
                score_threshold=self._score_thres,
            )

            display_info[str(severity)] = [
                str(Path(image_path).relative_to(self._output_folder)),
                ground_truth,        # gt_classes
                drawn_prediction,    # pred_classes
                str(Path(image_path2).relative_to(self._output_folder)),
            ]
        return display_info

