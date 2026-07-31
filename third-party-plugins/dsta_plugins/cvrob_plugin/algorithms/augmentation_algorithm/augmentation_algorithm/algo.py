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
from .cvrob_util import get_prediction_from_image, triplets, augmentation_gradient, handle_class_names_arg, mem_mb, evaluate
from .augmentations_class import make_augmentation_dict, custom_parameter_change, handle_url_algos
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
                    f"The algorithm has failed ground truth header validation. \
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
                f"The algorithm has failed validation for the project path. \
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
                f"The algorithm has failed validation for its plugin type. \
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

        print('aug dict', aug_dict)
        self._augmentation_method(aug_dict)

        # Update progress (For 100% completion)
        self._progress_inst.update(1)

    def _augmentation_method(self, aug_dict):
        print(f"[mem at the start of _augmentation_method] {mem_mb():.1f} MB")
        image_paths: list[str] = self._data_instance.get_data()["image_directory"].tolist()
        ground_truths = self._ordered_ground_truth_df[self._ground_truth_label].tolist()
        test_dataset, test_loader = self._load_images(image_paths, ground_truths)
        #KIV: set a random seed here manually; if we want to manually set it then we'll need to change this
        np.random.seed(42) 
        display_idx = np.random.choice(len(image_paths))
        output_results = dict()

        # model_api_url = self._input_arguments.get("model_api_url") or None
        # if model_api_url:
        #     model = model_api_url
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
        if 'http' in aug_dict['url']:
            handle_url_algos(aug_dict, aug_methods)

        class_names_arg = self._input_arguments['class_names'] or None 
        # if model_api_url and not class_names_arg:
        #     raise ValueError("class_names must be explicitly provided when using model_api_url, since class count can't be inferred from a remote API")

        class_names = handle_class_names_arg(class_names_arg, model)
        print("Class names:", class_names)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for aug_name, aug_class in aug_dict.items():
            print(f"{aug_name}: [mem at the start of aug_dict.items stuff] {mem_mb():.1f} MB")
                      
            if aug_name not in aug_methods and aug_methods != ["all"]:
                continue
            if aug_name == 'url':
                continue

            print(f"{aug_name}: [mem at the start of aug_dict.items stuff] {mem_mb():.1f} MB")
            individual_results = dict() 
            individual_results.update({"Augmentation": aug_name})

            display_info = dict()
            aug_dir =  self._output_folder / aug_name
            os.makedirs(aug_dir, exist_ok=True)

            num_epochs = self._input_arguments.get('num_epochs') or 1
            # loader1 = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
            # print(f"[mem before 1-image batch] {mem_mb():.1f} MB")
            # acc, _, _ = evaluate(model, loader1, None)  # will run 100 forward passes but each on a single image
            # print(f"[mem after 1-image batch loop] {mem_mb():.1f} MB")    
            gradient, accuracies, fig_path = augmentation_gradient(model, test_loader, device, aug_class, 'matplotlib', aug_dir, num_epochs)
            first_drop = accuracies[1] - accuracies[0]
            severities = ["None"] + aug_class.severities
            for severity_idx, severity in enumerate(severities):
                corrupted_dir = Path(aug_name) / f"severity{severity}"
                display_image = self._get_one_corrupted_image_direct(
                    image_paths[display_idx],
                    aug_class,
                    severity,
                )
                image_path = self._save_one_image(display_image, str(corrupted_dir), Path(str(image_paths[display_idx])).name)
                prediction = get_prediction_from_image(model, display_image, device)
                ground_truth = ground_truths[display_idx]

                random_display = [
                    str(Path(image_path).relative_to(self._output_folder)),
                    class_names[str(ground_truth)],
                    class_names[str(prediction)],
                ]
                display_info.update({str(severity): random_display})
                
            print(accuracies, first_drop)
            accuracies_dict = {k:v for k,v in zip(severities, accuracies)}
            print(aug_name, 'augmentation method gradient:', gradient)
            individual_results.update(
                {"display_info": display_info, 
                "accuracies": accuracies_dict, 
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

    def _load_images(self, image_paths: list[str], labels) -> list[np.ndarray]:
        """
        Load a list of numpy images from file paths.

        Args:
            image_paths (list[str]): A list of image file paths

        Returns:
            np.ndarray: A list of numpy images
        """
        # from .cvrob_util import ImageDataset  # or wherever it's imported from
        # dataset = ImageDataset(image_paths, labels)
        # dataset.transform = transforms.Compose([
        #     transforms.Resize((500, 700)),  # H, W
        #     transforms.ToTensor()
        # ])
        transform = transforms.Compose([
            transforms.Resize((500, 700)),  # H, W
            transforms.ToTensor()
        ])
        image_tensors = torch.stack([transform(Image.open(p).convert("RGB")) for p in image_paths])
        label_tensors = torch.tensor(labels, dtype=torch.long)
        dataset = TensorDataset(image_tensors, label_tensors)
        loader = DataLoader(dataset, batch_size=16, shuffle=False)

        return dataset, loader

    def _save_one_image(self, image: np.ndarray, subfolder_name, image_path_original):

        save_dir = self._save_folder / subfolder_name
        save_dir.mkdir(parents=True, exist_ok=True)

        image_path = save_dir / image_path_original

        # CHW -> HWC
        image = image.transpose(1, 2, 0)
        image = image.astype(np.float32)
        # normalize safely
        if image.max() <= 1.5:
            image *= 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)

        Image.fromarray(image).save(image_path)
        return str(image_path)

    def _get_one_corrupted_image_direct(
        self,
        image_path: str,
        aug_class,         # Augmentation instance
        severity: str,     # e.g. "severity_1" or "None"
        resize: tuple[int, int] = (500, 700),  # (H, W) — match _load_images
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