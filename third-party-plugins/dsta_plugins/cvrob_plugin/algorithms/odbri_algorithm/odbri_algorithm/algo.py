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
import torch
from .cvrob_util import *
from .augmentations_class import handle_url_algos
from .augmentations_brittle import *
import pandas as pd 
from . import cvrob_algo_common

@dataclass
class _BrittleCtx:
    """
    Everything ``_brittle_method`` resolves once up front and threads to its
    phase helpers, so no single method has to re-derive the setup.

    Attributes:
        image_paths (list): All image file paths.
        ground_truths (list): Ground-truth label per image.
        model: The unwrapped torch model (or API URL string).
        device (torch.device): Device the model runs on.
        class_names (dict): Class index (as str) -> display name.
        class_names_int (dict): Class index (as int) -> display name.
        aug_name (str): The selected augmentation's name.
        aug_class: The selected ``Augmentation`` instance.
        severities (tuple): ``(before, after)`` severity specifiers.
        loader_A: Loader for the before severity (may be the clean loader).
        loader_B: Loader for the after severity.
    """
    image_paths: list
    ground_truths: list
    model: object
    device: object
    class_names: dict
    class_names_int: dict
    aug_name: str
    aug_class: object
    severities: tuple
    loader_A: object
    loader_B: object

@dataclass
class _BrittleScored:
    """
    The images, probabilities and labels captured from the two scoring passes.

    Attributes:
        scores_A: image detection scores before corruption
        scoress_B: image detection scores after corruption
        ground_truths: Ground-truths for bbox, label, etc.
    """
    scores_A: object
    scores_B: object
    ground_truths: object

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
    The Plugin(OD Brittle Algorithm) class specifies methods in generating results for algorithm
    """

    # Some information on plugin
    _name: str = "OD Brittle Algorithm"
    _description: str = "This algorithm shows the top brittle images whos model performance drops the most in OD"
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
        self._brittle_method(aug_dict)
        # Update progress (For 100% completion)
        # self._progress_inst.update(1)

    def _brittle_method(self, aug_dict):
        """
        Master brittle method.

        Returns the predictions/probabilities of before and after corruption,
        determines the brittleness of each image,
        and creates the visual artifacts showing the results
        """
        ctx = self._setup_brittle(aug_dict)

        # Work units: 2 scoring passes + display-info + 3 visualizers.
        self._progress_inst.add_total(6)

        scored = self._score_severities(ctx)
        b_result = self._build_brittleness_results(scored) #, correct_before, correct_before_wrong_after = \
        self._results = self._render_brittle_outputs(
            ctx, b_result
        )

    def _setup_brittle(self, aug_dict):
        """
        Resolve all inputs the brittleness run needs before any scoring.

        Loads the images, unwraps the model, resolves class names and the chosen
        augmentation, validates the before/after severities, and builds the two
        corresponding data loaders.

        Args:
            aug_dict (Dict[str, Any]): Mapping of augmentation name to instance.

        Returns:
            _BrittleCtx: The resolved context threaded through the phase helpers.
        """
        image_paths: list[str] = self._data_instance.get_data()["image_directory"].tolist()
        ground_truths = self._ordered_ground_truth
        _, test_loader = self._load_images_objdet(image_paths, ground_truths)
        # KIV: set a random seed here manually; if we want to manually set it then we'll need to change this
        np.random.seed(42)

        model = self._unwrap_model()

        class_names_arg = self._input_arguments.get('class_names') or None
        class_names = handle_class_names_arg(class_names_arg, model)
        print("Class names:", class_names)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_names_int = {int(k): v for k, v in class_names.items()}

        aug_name = self._input_arguments['aug_method']
        if 'url' in aug_dict and 'http' in aug_dict['url']:
            handle_url_algos(aug_dict, [aug_name])
        aug_class = aug_dict[aug_name]

        severities = self._validate_severities()
        if severities[0] == "None":
            loader_A = test_loader
        else:
            loader_A = aug_class.corr_func_dataloader(test_loader, severity_idx=severities[0])
        loader_B = aug_class.corr_func_dataloader(test_loader, severity_idx=severities[1])

        return _BrittleCtx(
            image_paths=image_paths,
            ground_truths=ground_truths,
            model=model,
            device=self._device,
            class_names=class_names,
            class_names_int=class_names_int,
            aug_name=aug_name,
            aug_class=aug_class,
            severities=severities,
            loader_A=loader_A,
            loader_B=loader_B,
        )

    def _loop_for_display_info(
        self,
        ctx,
        top_k_indices,
    ):
        """
        Build the top-K display rows for the before/after severities.

        For each severity and each selected sample, saves the exact image the
        model scored and reuses its prediction from the same pass, so the shown
        image and predicted label can never disagree and no second
        corrupt+inference is run.

        Args:
            ctx: brittleness context with all the variables
            top_k_indices (List[int]): Sample indices to display.
        """
        display_info = []

        for s_idx in ctx.severities:
            s = ctx.aug_class.determine_severity(s_idx)
            corrupted_images = [
                self._get_one_corrupted_image_direct(ctx.image_paths, ctx.ground_truths, ctx.aug_class, s, idx)
                for idx in top_k_indices
            ]

            corrupted_dir = Path(ctx.aug_name) / f"severity_{s}"

            for i,display_idx in enumerate(top_k_indices):

                image_path = self._save_one_image(corrupted_images[i], str(corrupted_dir), display_idx)
                prediction = get_prediction_from_image(ctx.model, corrupted_images[i], self._device)

                ground_truth = ctx.ground_truths[display_idx]

                image_path2, drawn_prediction = self._save_image_with_predictions(
                    image=corrupted_images[i],
                    prediction=prediction,
                    gt_boxes=ground_truth,        # list of {"bbox": [...], "label": ...}
                    class_names=ctx.class_names,
                    subfolder_name=str(corrupted_dir),
                    idx=display_idx,
                    score_threshold=self._score_thres
                )

                random_display = [
                    str(Path(image_path).relative_to(self._output_folder)),
                    ground_truth,
                    drawn_prediction,
                    str(Path(image_path2).relative_to(self._output_folder)),
                ]
                display_info.append({f"severity_{s}_number_{i+1}": random_display})

        return display_info   

    def _score_severities(self, ctx):
        """
        Score the model over the before/after loaders.

        Runs one inference pass per severity, capturing the exact images and
        class probabilities (with ground-truth labels) for reuse in ranking and
        display. Advances progress once per pass.

        Args:
            ctx (_BrittleCtx): Resolved run context.

        Returns:
            _BrittleScored: Images, probabilities and labels for both severities.
        """
        scores_A = collect_detection_predictions(ctx.model, ctx.loader_A, ctx.device)
        self._progress_inst.update(1)
        scores_B = collect_detection_predictions(ctx.model, ctx.loader_B, ctx.device)
        self._progress_inst.update(1)

        return _BrittleScored(
            scores_A=scores_A, scores_B=scores_B, ground_truths=ctx.ground_truths
        )

    def _build_brittleness_results(self, scored):
        """
        Rank every sample by its confidence drop on the true class.

        Computes per-sample brittleness (P(true) before − after), builds and
        sorts the per-sample results (most brittle first), and derives the two
        filtered subsets used by the visualizers.

        Args:
            scored (_BrittleScored): Per-severity images, probabilities, labels.

        Returns:
            Tuple[BrittlenessResult, list, list]: The aggregate result, the
                samples correct before corruption, and the subset of those that
                became incorrect after.
        """
        scores_A, scores_B, ground_truths = scored.scores_A, scored.scores_B, scored.ground_truths
        brittleness = torch.tensor([
            image_brittleness(a, b, iou_thresh=self._iou_thres)
            for a, b in zip(scores_A, scores_B)
        ])
        
        results_all = [
            BrittlenessResultIndiv(
                index=i,
                label=ground_truths[i],
                predA=scores_A[i],
                predB=scores_B[i],
                pA=float(scores_A[i]["scores"].sum().item()) if len(scores_A[i]["scores"]) else 0.0,
                pB=float(scores_B[i]["scores"].sum().item()) if len(scores_B[i]["scores"]) else 0.0,
                brittleness=float(brittleness[i]),
            )
            for i in range(len(ground_truths))
        ]

        # Sort (most brittle first)
        results_all_sorted = sorted(results_all, key=lambda x: x.brittleness, reverse=True)
        b_result = BrittlenessResult(
            results = results_all_sorted,
            probs_A = scores_A,
            probs_B = scores_B,
            labels = ground_truths
        )

        return b_result

    def _render_brittle_outputs(self, ctx, b_result):
        """
        Build display samples and visualizations, and assemble the results dict.

        Saves the top-K display rows, renders the matplotlib grid (plus
        fragments), the standalone HTML top-K, and the HTML carousel, then packs
        every path and summary field into the output dict. Advances progress once
        for the display pass and once per visualizer.

        Args:
            ctx (_BrittleCtx): Resolved run context.
            scored (_BrittleScored): Per-severity images, probabilities, labels.
            b_result (BrittlenessResult): The ranked aggregate result.
            correct_before (list): Samples correct before corruption.
            correct_before_wrong_after (list): Of those, ones wrong after.

        Returns:
            Dict[str, Any]: The fully-populated results dict for ``self._results``.
        """
        # KIV: define the top_k value here; if want to make custom then we change this
        TOPK_SAFE = 15
        TOPK = 10

        output_results = brittle_res_to_dict(b_result)
        # output_results = {k:v for k,v in output_results.items() if k not in ["imgs_A", "imgs_B"]}
        results_list = output_results['results']
        top_k = sorted(results_list, key=lambda x: x["brittleness"], reverse=True)[:min(TOPK_SAFE, len(results_list))]
        top_k_indices = [item["index"] for item in top_k]

        print("### Starting looping for display info... ###")

        display_info = self._loop_for_display_info(
            ctx,
            top_k_indices
        )
       
        print("### Finished looping for display info!!! ###")
        output_results.update(
            {"display_info": display_info}
        )

        all_result_paths_dict = process_and_visualize_brittleness_method(
            b_result,
            self._output_folder,
            ctx.aug_class,
            ctx.severities,
            ctx.class_names_int,
            ctx.image_paths,
            self._score_thres,
            ctx.aug_name,
            ctx.ground_truths,
            TOPK,
            self._progress_inst,
            iou_thres= self._iou_thres
        )

        mpl_path, mpl_frag_paths = all_result_paths_dict['mpl']
        mpl_path_with_det, mpl_frag_paths_with_det = all_result_paths_dict['mpl_det']
        plotly_path = all_result_paths_dict['plotly']
        plotly_path_with_det = all_result_paths_dict['plotly_det']
        html_path = all_result_paths_dict['html']
        html_path_with_det = all_result_paths_dict['html_det']

        print("### Finished visualization for display info!!! ###")
        output_results.update(
            {
                "matplotlib_image_path": str(mpl_path.relative_to(self._output_folder)),
                "plotly_image_path": str(plotly_path.relative_to(self._output_folder)),
                "html_carousel_path": str(html_path.relative_to(self._output_folder)),
                "matplotlib_fragment_paths": [str(x.relative_to(self._output_folder)) for x in mpl_frag_paths],
                "matplotlib_image_path_with_det": str(mpl_path_with_det.relative_to(self._output_folder)),
                "plotly_image_path_with_det": str(plotly_path_with_det.relative_to(self._output_folder)),
                "html_carousel_path_with_det": str(html_path_with_det.relative_to(self._output_folder)),
                "matplotlib_fragment_paths_with_det": [str(x.relative_to(self._output_folder)) for x in mpl_frag_paths_with_det],
                "dataset_size": len(ctx.image_paths),
                "class_names": ctx.class_names
            }
        )

        return output_results

    def _validate_severities(self):
        """
        Process the severity values from the input arguments.

        Either take the actual names from b4/aft, or take the indices from b4 idx/aft idx.

        Returns:
            Tuple: of the b4-aft severities. 
        """
        severity_before = self._input_arguments.get("severity_before")
        severity_after = self._input_arguments.get("severity_after")
        severity_before_idx = self._input_arguments.get("severity_before_idx")
        severity_after_idx = self._input_arguments.get("severity_after_idx")

        # normalize empty strings to None (important if UI sends "")
        severity_before = severity_before if severity_before != "" else None
        severity_after = severity_after if severity_after != "" else None
        # ---- validation ----
        if (
            severity_before is None
            and severity_after is None
            and severity_before_idx is None
            and severity_after_idx is None
        ):
            raise ValueError(
                "Must provide either severity_before/after (string) "
                "or severity_before_idx/after_idx (integer)"
            )

        # ---- choose strings if provided ----
        if severity_before is not None or severity_after is not None:
            if severity_before is None or severity_after is None:
                raise ValueError(
                    "Both severity_before and severity_after must be provided together"
                )
            severity0 = severity_before
            severity1 = severity_after

        # ---- otherwise use indices ----
        else:
            if severity_before_idx is None or severity_after_idx is None:
                raise ValueError(
                    "Both severity_before_idx and severity_after_idx must be provided together"
                )
            severity0 = severity_before_idx
            severity1 = severity_after_idx

        severities = (severity0, severity1)
        print("SEVERITIES:", severities)
        return severities