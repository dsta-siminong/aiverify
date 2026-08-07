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

from .cvrob_util import triplets, DetectionDataset
from .augmentations_class import make_augmentation_dict, custom_parameter_change
from PIL import ImageDraw, ImageFont


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
        if aug_dict is None:
            raise ValueError("Invalid augmentation library provided. Did you get the URL wrong or misspell the library name?")

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
        """
        Build the detection dataset and loader over the given images.

        Wraps the paths and per-image targets in a ``DetectionDataset`` (with a
        ``ToTensor`` transform) and a ``DataLoader`` with ``shuffle=False`` so
        sample order aligns with ``image_paths``/``targets`` for later indexing.

        Args:
            image_paths (List[str]): Image file paths, one per sample.
            targets (List[List[dict]]): Per-image ground-truth annotations, each a
                list of ``{"bbox": [...], "label": <id>}`` dicts.

        Returns:
            Tuple[DetectionDataset, DataLoader]: The dataset and its loader
                (batched, unshuffled, with the detection collate function).
        """
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
        """
        Draw ground-truth and predicted boxes on the image and save it.

        Renders ground-truth boxes in green and predicted boxes (scoring at or
        above ``score_threshold``) in red, labelling each with its class name,
        then writes the annotated PNG. The returned ``drawn_prediction`` is built
        in lockstep with the drawing loop, so it contains exactly the boxes that
        were rendered (never the thresholded-out ones).

        Args:
            image (np.ndarray): CHW image; float in [0, 1] or already 0-255.
            prediction (dict): Model output with ``boxes``, ``labels``, ``scores``
                (each a tensor or array-like, aligned by index).
            gt_boxes (list): Ground-truth annotations, either ``{"bbox", "label"}``
                dicts or bare ``[x1, y1, x2, y2]`` lists; ``None``/empty is allowed.
            class_names (dict): Class index (as str) -> display name.
            subfolder_name (str): Sub-path under the save folder to write into.
            idx (int): Sample index, used in the saved file name.
            score_threshold (float, optional): Minimum score for a predicted box
                to be drawn. Defaults to ``0.5``.

        Returns:
            Tuple[str, dict]: The saved image path and the ``drawn_prediction``
                dict (``boxes``/``labels``/``scores`` for only the drawn boxes).
        """
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
        """
        Save a single CHW image as a PNG (no annotations).

        Converts CHW -> HWC, scales to 0-255 if the image looks normalized
        (max <= 1.5), clips to uint8, and writes the file.

        Args:
            image (np.ndarray): CHW image; float in [0, 1] or already 0-255.
            subfolder_name (str): Sub-path under the save folder to write into.
            idx (int): Sample index, used in the saved file name.

        Returns:
            str: The saved image path.
        """
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
        """
        Corrupt a single image at the given severity, decoding it directly.

        Opens just the target image (no full loader), applies the augmentation at
        ``severity``, and returns it as a CHW float array in [0, 1]. The clean
        image is returned unchanged when the augmentation or severity is "None".

        Args:
            image_paths (List[str]): All image file paths.
            ground_truths (list): Per-image ground-truth annotations (passed to the
                augmentation, which may transform boxes alongside the image).
            aug_class: The ``Augmentation`` instance (its ``corr_func_sample``).
            severity: Severity specifier, or ``"None"`` for the clean image.
            target_idx (int): Index of the image to corrupt.

        Returns:
            np.ndarray: The (possibly corrupted) image, CHW float in [0, 1].
        """
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