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
from .cvrob_util import augmentation_gradient, handle_class_names_arg, mem_mb
from .augmentations_class import handle_url_algos
from . import cvrob_algo_common
from pathlib import Path
import pandas as pd

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

    def generate(self) -> None:
        """
        A method to generate the algorithm results with the provided data, model, ground truth information.
        """
        print(f"[mem at the start of generate] {mem_mb():.1f} MB")
        # Retrieve data information
        self._data = self._data_instance.get_data()
        #make ground truth
        file_names = [Path(i).name for i in self._data_instance.get_data()["image_directory"]]
        df: pd.DataFrame = self._ground_truth_instance.get_data()
        self._file_name_label = "file_name" #self._input_arguments["file_name_label"]
        self._ordered_ground_truth_df = df.set_index(self._file_name_label).reindex(file_names) 

        # Initialise main image directory
        if self._save_folder.exists():
            shutil.rmtree(self._save_folder)
        self._save_folder.mkdir(parents=True, exist_ok=True)

        aug_dict = self._resolve_aug_dict()
        self._augmentation_method(aug_dict)

        # Update progress (For 100% completion)
        # Progress is now advanced per-augmentation inside _augmentation_method,
        # so this final tick is no longer needed (it would push completed past total).
        # self._progress_inst.update(1)

    def _augmentation_method(self, aug_dict):
        """
        Evaluate the model against every selected augmentation.

        Loads images, unwraps the model, resolves which augmentations to run,
        then accumulates per-augmentation results, gradients, and first-drop
        values into ``self._results``.

        Args:
            aug_dict (Dict[str, Any]): Mapping of augmentation name to its
                ``Augmentation``/``AugmentationUrl`` instance (may also contain a
                ``"url"`` entry for remote backends).

        Returns:
            None
        """
        print(f"[mem at the start of _augmentation_method] {mem_mb():.1f} MB")
        image_paths: list[str] = self._data_instance.get_data()["image_directory"].tolist()
        ground_truths = self._ordered_ground_truth_df[self._ground_truth_label].tolist()
        test_dataset, test_loader = self._load_images(image_paths, ground_truths)
        #KIV: set a random seed here manually; if we want to manually set it then we'll need to change this
        seed = self._input_arguments.get('random_seed', 42)   # configurable, 42 default
        display_rng = np.random.default_rng(seed)
        display_idx = display_rng.integers(len(image_paths))

        model = self._unwrap_model()

        aug_methods = self._get_aug_methods()
        print("Augmentation methods:", aug_methods)
        if 'url' in aug_dict and 'http' in aug_dict['url']:
            handle_url_algos(aug_dict, aug_methods)

        class_names_arg = self._input_arguments['class_names'] or None
        class_names = handle_class_names_arg(class_names_arg, model)
        print("Class names:", class_names)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Register total work units (one per augmentation that will actually run)
        # so progress advances incrementally instead of jumping 0 -> 100%.
        num_augs_to_run = sum(
            1 for aug_name in aug_dict if self._should_run_aug(aug_name, aug_methods)
        )
        self._progress_inst.add_total(num_augs_to_run)

        combined_results = []; gradients = []; first_drops = []
        for aug_name, aug_class in aug_dict.items():
            if not self._should_run_aug(aug_name, aug_methods):
                continue

            individual_results, gradient, first_drop = self._process_augmentation(
                aug_name, aug_class, model, test_loader,
                image_paths, ground_truths, class_names, display_idx, device,
            )
            combined_results.append(individual_results)
            gradients.append(gradient)
            first_drops.append(first_drop)

            # One augmentation finished; advance the progress bar.
            self._progress_inst.update(1)
            print()

        self._results = {
            "results": combined_results,
            "gradients": gradients,
            "first_drops": first_drops,
            "augmentation_names": [x["Augmentation"] for x in combined_results],
            "dataset_size": len(image_paths),
            "class_names": class_names,
        }

    def _process_augmentation(self, aug_name, aug_class, model, test_loader,
                              image_paths, ground_truths, class_names, display_idx, device):
        """
        Evaluate one augmentation across its severities and assemble its results.

        Computes the accuracy-vs-severity gradient and first-severity drop, builds
        the per-severity display samples, and packages everything (including the
        saved plot path) into a single results dict.

        Args:
            aug_name (str): Name of the augmentation.
            aug_class: The ``Augmentation`` instance to evaluate.
            model: Underlying model (or API URL string).
            test_loader (DataLoader): Loader over the (uncorrupted) test images.
            image_paths (List[str]): All image file paths.
            ground_truths (List): Ground-truth label per image.
            class_names (Dict[str, str]): Mapping from class index to display name.
            display_idx (int): Index of the sample image to display.
            device (torch.device): Device the model runs on.

        Returns:
            Tuple[Dict[str, Any], float, float]: The per-augmentation results dict,
                the best-fit gradient, and the first-severity accuracy drop.
        """
        print(f"{aug_name}: [mem at the start of aug_dict.items stuff] {mem_mb():.1f} MB")
        aug_dir = self._output_folder / aug_name
        os.makedirs(aug_dir, exist_ok=True)

        num_epochs = self._input_arguments.get('num_epochs') or 1
        gradient, accuracies, fig_path, display_scored = augmentation_gradient(
            model, test_loader, device, aug_class, 'matplotlib', aug_dir, num_epochs,
            display_idx=display_idx,
        )
        first_drop = accuracies[1] - accuracies[0]
        severities = ["None"] + aug_class.severities

        display_info = self._build_display_info(
            aug_name, aug_class, image_paths, ground_truths,
            class_names, display_idx, display_scored,
        )

        print(accuracies, first_drop)
        print(aug_name, 'augmentation method gradient:', gradient)
        individual_results = {
            "Augmentation": aug_name,
            "display_info": display_info,
            "accuracies": {k: v for k, v in zip(severities, accuracies)},
            "fig_img": str(fig_path.relative_to(self._output_folder)),
        }
        return individual_results, gradient, first_drop

    def _build_display_info(self, aug_name, aug_class, image_paths,
                            ground_truths, class_names, display_idx, display_scored):
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
        for severity in severities:
            corrupted_dir = Path(aug_name) / f"severity{severity}"
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