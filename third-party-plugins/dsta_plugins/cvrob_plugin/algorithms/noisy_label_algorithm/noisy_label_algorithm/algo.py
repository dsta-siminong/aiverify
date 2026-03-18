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
from PIL import Image
from . import noisy_labels

import inspect
import numpy as np
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, TensorDataset
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
    The Plugin(Noisy Label Algorithm) class specifies methods in generating results for algorithm
    """

    # Some information on plugin
    _name: str = "Noisy Label Algorithm"
    _description: str = "This algorithm looks for the noisy labels within a dataset"
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
        print("~~~~ df ~~~~")
        print(df)
        print("filenames")
        print(file_names)
        self._file_name_label = "file_name" #self._input_arguments["file_name_label"]
        self._ordered_ground_truth_df = df.set_index(self._file_name_label).reindex(file_names) 

        num_epochs = self._input_arguments['num_epochs']
        # Initialise main image directory
        if self._save_folder.exists():
            shutil.rmtree(self._save_folder)
        self._save_folder.mkdir(parents=True, exist_ok=True)

        # Apply user defined parameters to default parameters
        method_fn = noisy_labels.get_all_label_noise_methods_dict()#{name: noisy_labels.METHOD_FN[name] for name in self._input_arguments["corruptions"]}
        DEFAULT_PARAMS = noisy_labels.get_default_params()
        user_params = {k: v for k, v in self._input_arguments.items() if k in DEFAULT_PARAMS and v}
        parameters = copy.deepcopy(DEFAULT_PARAMS)
        parameters['num_epochs'] = num_epochs
        parameters.update(user_params)

        self._noisy_label_assessment(method_fn, parameters)

        # Get the values of the feature name and convert to a list.
        # self._results = {
        #     "my_expected_results": list(self._data[my_user_defined_feature_name].values)
        # }

        # Update progress (For 100% completion)
        self._progress_inst.update(1)

    def _noisy_label_assessment(self, method_fn: dict[str, Callable], param_dict: dict[str, list]) -> None:
        """
        A method to get the accuracy results at different severity levels and formatted in the desired output schema

        Parameters:
            corruption_fn (dict): Mapping of corruption name to its corresponding function object
            parameters (dict): Dict of parameter values at different severity levels
        """
        true_noisy_indices = None
        image_paths: list[str] = self._data_instance.get_data()["image_directory"].tolist()
        ground_truths = self._ordered_ground_truth_df[self._ground_truth_label].tolist()
        test_dataset, test_loader = self._load_images(image_paths, ground_truths)
        combined_results = []
        output_results = dict()
        # np.random.seed(self._set_seed)
        np.random.seed(42) #to be set manually next
        display_idx = np.random.choice(len(image_paths))

        if "_model" in dir(self._model_instance):
            model = self._model_instance._model
        elif "_pipeline" in dir(self._model_instance):
            model = self._model_instance._pipeline
        else:
            raise ValueError("idk what the", type(self._model_instance),"model instance is supposed to be ", dir(self._model_instance))

        self._progress_inst.add_total(len(method_fn))
        evaluate = False #idk if want to change another time

        if evaluate:
            print("changing the indices for noisy index searching")
            class_labels = np.unique(test_dataset.targets)
            num_test_samples = len(test_dataset.targets)
            true_noisy_indices = random.sample(range(num_test_samples), int(noise_ratio * num_test_samples))
            ground_truth_labels = np.array(test_dataset.targets).copy() # Store true labels
            noisy_test_labels = ground_truth_labels.copy()

            for idx in true_noisy_indices:
                while(noisy_test_labels[idx] == ground_truth_labels[idx]):
                    noisy_test_labels[idx] = random.choice(class_labels)

            test_dataset.targets = noisy_test_labels
        
        # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        param_dict['test_loader'] = test_loader
        param_dict['model'] = model
        # param_dict['device'] = device

        noisy_indices_all = []; weight_scores_all = []
        for k,v in method_fn.items():
            individual_results = dict()
            individual_results.update({"method": str(k)})
            print('method:', k)
            sig = noisy_labels.inspect_signature(v)
            accepted_params = {
                name: param_dict[name]
                for name in sig.parameters
                if name in param_dict
            } 
            noisy_indices = v(**accepted_params)
            if type(noisy_indices) == tuple and len(noisy_indices) == 2:
                noisy_indices, weight_scores = noisy_indices
            else:
                weight_scores =  np.array([1 for i in range(len(noisy_indices))])
            # noisy_indices = method(test_loader, model, device)
            if evaluate:
                evaluate_noisy_indices(noisy_indices, true_noisy_indices)
                print('and that was for method:', method)
            noisy_indices_all.append(noisy_indices)
            weight_scores_all.append(weight_scores)
            # random_display = [
            #     str(Path(corrupted_image_paths[display_idx]).relative_to(self._output_folder)),
            #     ground_truths[display_idx],
            #     predictions[display_idx],
            # ]
            # display_info.update({"severity" + str(severity): random_display})
            accepted_params2 = {k:v for k,v in accepted_params.items() if k not in ['device', 'test_loader', 'model']}
            weight_scores_fix = [float(x) for x in weight_scores]
            individual_results.update(
                {"noisy_indices": list(noisy_indices), "weight_scores": weight_scores_fix, "parameters": accepted_params2 }# "display_info": display_info}
            )
            combined_results.append(individual_results)

            self._progress_inst.update(1)
            print()

            break
        
        # return noisy_indices_all, weight_scores_all, true_noisy_indices, test_loader  

        print('time to ensemble the indices!')
        final_noisy_indices = noisy_labels.ensemble_method_general(noisy_indices_all, weight_scores)
        evaluate_stats = [0] #noisy_labels.evaluate_noisy_indices(final_noisy_indices, true_noisy_indices)
        filenames = list(self._ordered_ground_truth_df.index.values)#[self._file_name_label].tolist()
        correct_dict = noisy_labels.label_noise_correction(model, None, final_noisy_indices, test_dataset, filenames)
        # evaluate_stats = correct_dict
        correctd_filenames = correct_dict['filenames']
        correctd_noisy_labels = correct_dict['noisy_labels']
        correctd_corrected_labels = correct_dict['corrected_labels']

        output_results.update({
            "final_noisy_indices": list(final_noisy_indices), 
            "evaluate_stats": evaluate_stats, 
            "correctd_filenames": correctd_filenames,
            "correctd_noisy_labels": correctd_noisy_labels,
            "correctd_corrected_labels": correctd_corrected_labels,
            "combined_results": combined_results,
            "method_names": [str(k) for k in method_fn]
        })
        import pprint
        print("OUTPUT RESULTS")
        pprint.pprint(output_results)

        # Assign output results
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