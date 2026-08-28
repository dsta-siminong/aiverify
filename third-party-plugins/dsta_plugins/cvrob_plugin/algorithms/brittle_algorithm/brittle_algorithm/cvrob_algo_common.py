"""
Shared plugin scaffolding for the cvrob augmentation algorithms.

This module factors out the code that every cvrob augmentation plugin shares:
the ``IAlgorithm`` boilerplate (construction, validation, logging, progress and
results accessors) plus the augmentation-setup and image helpers. Each plugin's
``algo.py`` subclasses :class:`BasePlugin`, keeps its own class-level metadata,
and only implements ``generate`` and its algorithm-specific logic.

It also hosts :class:`ImageDataset`, the lazy image dataset used by
``_load_images``, so every plugin loads images the same way.

NOTE: this file is co-located in each plugin package (like ``cvrob_util.py`` and
``augmentations_class.py``), so ``Path(__file__).parent`` inside ``__init__``
resolves to the package directory and finds that plugin's schema files.
"""

import logging
from pathlib import Path, PurePath
from typing import Dict, Tuple, Union

import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from aiverify_test_engine.interfaces.ialgorithm import IAlgorithm
from aiverify_test_engine.interfaces.idata import IData
from aiverify_test_engine.interfaces.imodel import IModel
from aiverify_test_engine.interfaces.ipipeline import IPipeline
from aiverify_test_engine.interfaces.iserializer import ISerializer
from aiverify_test_engine.plugins.enums.plugin_type import PluginType
from aiverify_test_engine.plugins.metadata.plugin_metadata import PluginMetadata
from aiverify_test_engine.utils.json_utils import load_schema_file, validate_json
from aiverify_test_engine.utils.simple_progress import SimpleProgress

from .cvrob_util import triplets
from .augmentations_class import make_augmentation_dict, custom_parameter_change
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class ImageDataset(Dataset):
    """
    Lazy image dataset that decodes each image from disk on access.

    Stores only file paths and labels, decoding and transforming an image only
    when indexed, so memory stays proportional to the batch rather than the set.
    """

    def __init__(self, image_paths, labels):
        """
        Store the image paths and labels for lazy loading.

        Args:
            image_paths (list[str]): Image file paths.
            labels (list): Label per image, aligned with ``image_paths``.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transforms.ToTensor()

    def __len__(self):
        """
        Return the number of images in the dataset.

        Returns:
            int: Count of image paths.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Load, transform, and return the image and label at ``idx``.

        Opens the image as RGB and applies ``self.transform`` to produce a
        (3, H, W) tensor.

        Args:
            idx (int): Index of the item to fetch.

        Returns:
            Tuple[torch.Tensor, Any]: The transformed image tensor and its label.
        """
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)  # (3, H, W)
        label = self.labels[idx]
        return image, label


class BasePlugin(IAlgorithm):
    """
    Shared base for the cvrob augmentation plugins.

    Implements the parts of the ``IAlgorithm`` contract common to every cvrob
    augmentation plugin (construction/validation, logging, progress, results)
    plus the augmentation-setup and image helpers. Subclasses supply the
    class-level plugin metadata and implement ``generate`` (and their own
    algorithm-specific methods).

    Subclasses MUST define these class attributes (used by ``__init__``/``setup``
    and the metadata accessors): ``_metadata``, ``_plugin_type``,
    ``_requires_ground_truth``, and ``_supported_algorithm_model_type``.
    """

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        """
        A method to return the metadata for this plugin

        Returns:
            PluginMetadata: Metadata of this plugin
        """
        return cls._metadata

    @classmethod
    def get_plugin_type(cls) -> PluginType:
        """
        A method to return the type for this plugin

        Returns:
            PluginType: Type of this plugin
        """
        return cls._plugin_type

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
            0, 0, kwargs.get("progress_callback", None)
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

        if self._requires_ground_truth:
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
        if self._model_type not in self._supported_algorithm_model_type:
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
                f"The algorithm has failed validation for its metadata: {self._metadata}",
            )
            raise RuntimeError("The algorithm has failed validation for its metadata")

        # Perform validation on plugin type
        if not isinstance(self._plugin_type, PluginType):
            self.add_to_log(
                logging.ERROR,
                f"The algorithm has failed validation for its plugin type. \
                Ensure that PluginType is PluginType.ALGORITHM: {self._plugin_type}",
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

    def _resolve_aug_dict(self):
        """
        Build the augmentation dict and apply any custom parameter overrides.

        Uses the ``aug_library`` input (default ``"albumentations"``) for the
        defaults, then applies any ``custom_parameters`` parsed as
        ``(aug_name, param_name, value)`` triplets; absent input keeps defaults.

        Returns:
            Dict[str, Any]: Mapping of augmentation name to its augmentation
                instance, with any overrides applied.

        Raises:
            RuntimeError: If ``custom_parameters`` are provided but malformed
                (e.g. wrong token count, unknown augmentation/parameter name, or
                an unparseable value).
        """
        # Apply user defined parameters to default parameters
        aug_library = self._input_arguments.get('aug_library') or "albumentations"
        aug_dict = make_augmentation_dict(aug_library)

        # Empty/absent input is the normal case: nothing to override, carry on.
        custom_parameters = self._input_arguments.get('custom_parameters') or None
        if custom_parameters:
            # The user explicitly asked for overrides, so a malformed value is a
            # real error: fail loudly rather than silently running with defaults.
            try:
                for aug_name, param_name, parameters_in_string in triplets(custom_parameters):
                    aug_dict = custom_parameter_change(
                        aug_dict, aug_name, param_name, parameters_in_string
                    )
            except Exception as e:
                self.add_to_log(
                    logging.ERROR,
                    f"Failed to apply custom_parameters '{custom_parameters}': {e}",
                )
                raise RuntimeError(
                    f"Invalid custom_parameters '{custom_parameters}': {e}"
                ) from e
        else:
            print("~~ No custom parameters provided, let's use the default ones! n_n ~~")

        print('aug dict', aug_dict)
        return aug_dict

    def _unwrap_model(self):
        """
        Return the underlying torch model/pipeline from the wrapped instance.

        Returns the ``_model`` attribute for a plain model or ``_pipeline`` for a
        pipeline, whichever the wrapped instance exposes.

        Returns:
            Any: The underlying model or pipeline object.

        Raises:
            ValueError: If the wrapped instance exposes neither ``_model`` nor
                ``_pipeline``.
        """
        if "_model" in dir(self._model_instance):
            return self._model_instance._model
        elif "_pipeline" in dir(self._model_instance):
            return self._model_instance._pipeline
        raise ValueError(
            "idk what the", type(self._model_instance),
            "model instance is supposed to be ", dir(self._model_instance),
        )

    def _get_aug_methods(self):
        """
        Parse the ``aug_methods`` input into a list of augmentation names.

        Splits the comma-separated input, stripping whitespace and dropping empty
        tokens; defaults to ``["all"]`` (select every augmentation) when unset.

        Returns:
            List[str]: The requested augmentation names, or ``["all"]`` for all.
        """
        aug_methods = self._input_arguments.get('aug_methods') or 'all'
        return [x.strip() for x in aug_methods.split(",") if x.strip()]

    def _should_run_aug(self, aug_name, aug_methods):
        """
        Decide whether the named augmentation is in scope for this run.

        The ``"url"`` entry is always skipped; any other name runs when it is
        listed in ``aug_methods`` or when ``aug_methods`` is exactly ``["all"]``.

        Args:
            aug_name (str): Name of the augmentation being considered.
            aug_methods (List[str]): Selected augmentation names, or ``["all"]``.

        Returns:
            bool: True if the augmentation should be evaluated, else False.
        """
        if aug_name == 'url':
            return False
        return aug_name in aug_methods or aug_methods == ["all"]

    def _load_images(self, image_paths: list[str], labels) -> list[np.ndarray]:
        """
        Wrap the image paths in a lazy dataset and batching loader.

        Builds an ``ImageDataset`` that decodes and resizes images to (500, 700)
        on access, so only one batch is held in memory at a time, and returns it
        together with a non-shuffling ``DataLoader``.

        Args:
            image_paths (list[str]): Image file paths to load lazily.
            labels: Ground-truth label per image, aligned with ``image_paths``.

        Returns:
            Tuple[ImageDataset, DataLoader]: The dataset and its batching loader.
        """
        dataset = ImageDataset(image_paths, labels)
        dataset.transform = transforms.Compose([
            transforms.Resize((500, 700)),  # H, W
            transforms.ToTensor()
        ])
        # transform = transforms.Compose([
        #     transforms.Resize((500, 700)),  # H, W
        #     transforms.ToTensor()
        # ])
        # image_tensors = torch.stack([transform(Image.open(p).convert("RGB")) for p in image_paths])
        # label_tensors = torch.tensor(labels, dtype=torch.long)
        # dataset = TensorDataset(image_tensors, label_tensors)
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2, persistent_workers=True)
        return dataset, loader

    def _save_one_image(self, image: np.ndarray, subfolder_name, image_path_original):
        """
        Write a single CHW image array to disk under the save folder.

        Converts CHW to HWC, scales [0, 1] floats to 0-255 when needed, clips to
        uint8, and saves as an image inside ``save_folder/subfolder_name``.

        Args:
            image (np.ndarray): CHW image array (float in [0, 1] or already 0-255).
            subfolder_name (str): Sub-path under the save folder to write into.
            image_path_original (str): File name to save the image as.

        Returns:
            str: Absolute path of the written image file.
        """
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
        Load and corrupt a single image directly from disk.

        Mirrors the ``_load_images`` transform (resize, then corrupt) for one
        image without loading any others; ``"None"`` severity or a ``"None"``
        augmentation returns the resized image uncorrupted.

        Args:
            image_path (str): Path to the source image.
            aug_class: The ``Augmentation`` instance providing the corruption.
            severity (str): Severity level to apply, or ``"None"`` for no corruption.
            resize (tuple[int, int]): Target (H, W) size, matching ``_load_images``.

        Returns:
            np.ndarray: CHW float32 array normalised to [0, 1].
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

    def _display_row_from_scored_batch(
        self,
        dataset,            # the dataset the batch pass iterated (clean or CorruptedDataset)
        idx: int,           # display index into that dataset
        prediction,         # y_pred[idx] from the same scored pass
        corrupted_dir,      # sub-path under the save folder for this aug/severity
        source_image_path,  # original path, used only for the saved file name
        ground_truth,       # ground-truth label of this image
        class_names,        # Dict[str, str]: class index -> display name
    ):
        """
        Build a display row from an already-scored batch item, no re-inference.

        Pulls the exact corrupted image the model scored (``dataset[idx]``) and
        reuses its batch prediction, so the shown image and prediction come from
        the same pixels while a second corrupt+inference pass is avoided.

        Args:
            dataset: Dataset the batch pass iterated (shuffle must be off so
                ``idx`` aligns with ``prediction``); yields (CHW float image, label).
            idx (int): Display index into ``dataset`` and into the batch's y_pred.
            prediction: The class index the model predicted for this item.
            corrupted_dir: Sub-path under the save folder to write the image into.
            source_image_path: Original image path, used only for the file name.
            ground_truth: Ground-truth label of this image (index into class_names).
            class_names (Dict[str, str]): Mapping from class index to display name.

        Returns:
            List[str]: ``[relative_image_path, ground_truth_name, predicted_name]``.
        """
        image, _ = dataset[idx]  # CHW float tensor in [0, 1] — exactly what was scored
        image_np = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
        saved_path = self._save_one_image(
            image_np, str(corrupted_dir), Path(str(source_image_path)).name
        )
        return [
            str(Path(saved_path).relative_to(self._output_folder)),
            class_names[str(ground_truth)],
            class_names[str(int(prediction))],
        ]
