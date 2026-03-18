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

from . import augmentations
import numpy as np
from PIL import Image
import inspect
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, TensorDataset
from .cvrob_util import evaluate, collect_probs, triplets
from .augmentations_class import make_augmentation_dict, custom_parameter_change
from .augmentations_brittle import (
    visualize_topk_matplotlib, 
    visualize_topk_plotly, 
    visualize_in_html, 
    BrittlenessResult, 
    BrittlenessResultIndiv,
    brittle_res_indiv_to_dict, 
    brittle_res_to_dict
)
import pandas as pd 
import json
import matplotlib.pyplot as plt
import plotly.express as px 
from pprint import pprint


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
        file_names = [Path(i).name for i in self._data_instance.get_data()["image_directory"]]
        df: pd.DataFrame = self._ground_truth_instance.get_data()
        # print("~~~~ df ~~~~")
        # print(df)
        # print("filenames")
        # print(file_names)
        self._file_name_label = "file_name" #self._input_arguments["file_name_label"]
        self._ordered_ground_truth_df = df.set_index(self._file_name_label).reindex(file_names) 

        # num_epochs = self._input_arguments['num_epochs']
        # Initialise main image directory
        if self._save_folder.exists():
            shutil.rmtree(self._save_folder)
        self._save_folder.mkdir(parents=True, exist_ok=True)

        # Apply user defined parameters to default parameters
        aug_dict = make_augmentation_dict(self._input_arguments['aug_library'])
        custom_parameters = None
        try:
            custom_parameters = self._input_arguments['custom_parameters']
            custom_parameters = triplets(custom_parameters)
            for sublist in custom_parameters:
                aug_name, param_name, parameters_in_string = sublist
                aug_dict = custom_parameter_change(aug_dict, aug_name, param_name, parameters_in_string)
        except Exception as e:
            print("No custom parameter_change")
            print(f"Custom parameters exception: {e} , {custom_parameters}")
            print()

        self._brittle_method(aug_dict)
        # Update progress (For 100% completion)
        self._progress_inst.update(1)

    def _brittle_method(self, aug_dict):
        print("brittle stage 1")
        image_paths : list[str] = self._data_instance.get_data()["image_directory"].tolist()
        ground_truths = self._ordered_ground_truth_df[self._ground_truth_label].tolist()
        test_dataset, test_loader = self._load_images(image_paths, ground_truths)
        np.random.seed(42) #to be set manually next
        display_idx = np.random.choice(len(image_paths))

        if "_model" in dir(self._model_instance):
            model = self._model_instance._model
        elif "_pipeline" in dir(self._model_instance):
            model = self._model_instance._pipeline
        else:
            raise ValueError("idk what the", type(self._model_instance),"model instance is supposed to be ", dir(self._model_instance))
        import json 
        current_file_dir = Path(__file__).parent

        with open(current_file_dir / 'boat_classes.json', 'r') as f:
            class_names = json.load(f)

        # labels = [k for k in class_names]
        class_names = {int(k): v for k, v in class_names.items()}
        target_names = [class_names[k] for k in class_names]

        aug_name = self._input_arguments['aug_method']
        if aug_name not in aug_dict:
            raise ValueError("aug method not valid")
        aug_class = aug_dict[aug_name]

        print("brittle stage 2")
        severity0 = self._input_arguments['severity_before']
        severity1 = self._input_arguments['severity_after']
        if severity0 in ["", " ", "INTEGER", 'integer', 'int', "Integer"]:
            assert severity1 in ["", " ", "INTEGER", 'integer', 'int', "Integer"]
            severity0 = self._input_arguments['severity_before_idx']
            severity1 = self._input_arguments['severity_after_idx']
            all_severities = ["None"] + aug_class.severities
            severity0 =  all_severities[severity0]
            severity1 =  all_severities[severity1]

        severities = (severity0, severity1)
        print("SEVERITIES:", severities)
        if severities[0] == "None":
            loader_A = test_loader
        else: 
            loader_A = aug_class.corr_func_dataloader(test_loader, severity_idx = severities[0])
        loader_B = aug_class.corr_func_dataloader(test_loader, severity_idx = severities[1])

        imgs_A, probs_A, labels = collect_probs(model, loader_A, None)
        imgs_B, probs_B, _      = collect_probs(model, loader_B, None)

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
        
        display_info = []      
        output_results = brittle_res_to_dict(b_result)
        output_results = {k:v for k,v in output_results.items() if k not in ["imgsA", "imgsB"]}
        results_list = output_results['results']
        top_k = sorted(results_list, key=lambda x: x["brittleness"], reverse=True)[:min(15, len(results_list))]
        top_k_indices = [item["index"] for item in top_k]
        
        _, predictions, _ = evaluate(model, test_loader, None)
        for s in severities:
            corrupted_images = self._get_corrupted_images(test_loader, aug_class, s)
            corrupted_dir = Path(aug_name) / f"severity_{s}"
            corrupted_image_paths = self._save_images(corrupted_images, str(corrupted_dir))

            image = torch.tensor(corrupted_images[display_idx]).unsqueeze(0)  # shape [1, C, H, W]

            model.eval()
            with torch.no_grad():
                outputs = model(image)
                _, prediction = torch.max(outputs, 1)
            prediction = prediction.item()

            for i,idx in enumerate(top_k_indices):
                random_display = [
                    str(Path(corrupted_image_paths[idx]).relative_to(self._output_folder)),
                    ground_truths[idx],
                    predictions[idx],
                ]
                display_info.append({f"severity_{s}_number_{i+1}": random_display})

        output_results.update(
            {"display_info": display_info}
        )

        print("OUTPUT RESULTS")
        # print(output_results.keys())
        # for k in output_results:
        #     print(k)
        #     print(np.array(output_results[k]).shape)
        # pprint(output_results)
        results = [
            r for r in b_result.results
            if r.predA == r.label and r.predB != r.label
        ]
        aug_dir =  self._output_folder / aug_name
        mpl_dir = aug_dir / f"matplotlib"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        plotly_dir = aug_dir / f"plotly"
        plotly_dir.mkdir(parents=True, exist_ok=True)
        mpl_path = visualize_topk_matplotlib(
            results, 
            b_result.imgsA, 
            b_result.imgsB, 
            b_result.probs_A, 
            b_result.probs_B,  
            K=min(10, len(results)),
            class_names=class_names, 
            transform=None,
            directory=mpl_dir,
            image_paths=image_paths
        )
        plotly_path = visualize_topk_plotly(
            results, 
            b_result.imgsA, 
            b_result.imgsB, 
            b_result.probs_A, 
            b_result.probs_B, 
            K=min(10, len(results)),
            class_names=class_names, 
            transform=None,
            directory = plotly_dir,
            image_paths=image_paths
        )
        html_path = visualize_in_html(
            results, 
            b_result.imgsA, 
            b_result.imgsB, 
            b_result.probs_A, 
            b_result.probs_B, 
            b_result.labels, 
            class_names=class_names, 
            transform=None,
            directory = plotly_dir,
            image_paths=image_paths
        )
        output_results.update(
            {
                "matplotlib_image_path": str(mpl_path.relative_to(self._output_folder)),
                "plotly_image_path": str(plotly_path.relative_to(self._output_folder)),
                "html_carousel_path": str(html_path.relative_to(self._output_folder)),
            }
        )

        self._results = output_results

    def _get_corrupted_images(self, testloader, aug_class, _severity):
        corrupted_images = []
        for images, labels in testloader:   
            images_np = (images * 255).byte().numpy().transpose(0, 2, 3, 1)  # Convert to HWC format and uint8
            
            # Apply corruption function with provided parameters
            if aug_class.name == "None" or _severity == "None":
                corrupted = images_np
            else:
                corrupted = aug_class.corr_func_arr(images_np, _severity)
            
            corrupted = (torch.tensor(corrupted.transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0).numpy()  #as opposed to torch.tensor
            corrupted_images.append(corrupted)
        return np.concatenate(corrupted_images, axis=0)

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

    def _save_images(self, images: list[np.ndarray], subfolder_name: str) -> list[str]:
        """
        Save a list of numpy arrays as images in a subfolder.

        Args:
            images (list[np.ndarray]): A list of numpy images
            subfolder_name (str): The name of the subfolder to save images

        Returns:
            list[str]: A list of saved image paths
        """
        image_paths = []
        save_dir = self._save_folder / subfolder_name
        save_dir.mkdir(parents=True, exist_ok=True)

        for idx, image in enumerate(images):
            image_path = save_dir / f"{idx}.png"
            # print("image shape", image.shape)
            image = np.transpose(image, (1, 2, 0))
            Image.fromarray((image * 255.0).astype(np.uint8)).save(image_path)
            image_paths.append(str(image_path))
        return image_paths
