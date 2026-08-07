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
from .cvrob_util import *
from .augmentations_class import make_augmentation_dict, custom_parameter_change, handle_url_algos
from .cvrob_algo_common import BasePlugin
from .augmentations_brittle import *
import pandas as pd
import json
from pprint import pprint
from dataclasses import dataclass


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
        imgs_A: CHW image tensors scored at the before severity.
        probs_A: Per-sample class probabilities at the before severity.
        imgs_B: CHW image tensors scored at the after severity.
        probs_B: Per-sample class probabilities at the after severity.
        labels: Ground-truth labels (shared across both passes).
    """
    imgs_A: object
    probs_A: object
    imgs_B: object
    probs_B: object
    labels: object


# =====================================================================================
# NOTE:
# 1. Check that you have installed the aiverify_test_engine latest package.
# 2. Check that you have run tests/install_core_plugins_requirements.sh to install all the
#    requirements required by the core plugins (serializers, data, models).
#    Alternatively, you may install the plugins that you require by installing the
#    requirements individually.
# 3. Do not modify the class name, else the plugin cannot be read by the system.
# =====================================================================================
class Plugin(BasePlugin):
    """
    # TODO: Update the plugin description below
    The Plugin(Brittle Algorithm) class specifies methods in generating results for algorithm
    """

    # Some information on plugin
    _name: str = "Brittle Algorithm"
    _description: str = "This algorithm shows the most brittle images"
    _version: str = "0.1.0"
    _metadata: PluginMetadata = PluginMetadata(_name, _description, _version)
    _plugin_type: PluginType = PluginType.ALGORITHM
    _requires_ground_truth: bool = True
    _supported_algorithm_model_type: List = [ModelType.CLASSIFICATION]

    def generate(self) -> None:
        """
        A method to generate the algorithm results with the provided data, model, ground truth information.
        """
        # Retrieve data information
        self._data = self._data_instance.get_data()
        file_names = [Path(i).name for i in self._data_instance.get_data()["image_directory"]]
        df: pd.DataFrame = self._ground_truth_instance.get_data()
        self._file_name_label = "file_name" #self._input_arguments["file_name_label"]
        self._ordered_ground_truth_df = df.set_index(self._file_name_label).reindex(file_names) 

        # Initialise main image directory
        if self._save_folder.exists():
            shutil.rmtree(self._save_folder)
        self._save_folder.mkdir(parents=True, exist_ok=True)

        aug_dict = self._resolve_aug_dict()
        # Progress is advanced inside _brittle_method across its work units
        # (two scoring passes, the display-info pass, and the three visualizers),
        # so it reaches 100% without a final manual tick here.
        self._brittle_method(aug_dict)

    def _brittle_method(self, aug_dict):
        """
        Orchestrate the brittleness evaluation end to end.

        Resolves setup, scores the before/after severities, ranks samples by
        confidence drop, and renders the display samples and visualizations into
        ``self._results``. Progress advances over the two scoring passes, the
        display-info pass, and the three visualizers.

        Args:
            aug_dict (Dict[str, Any]): Mapping of augmentation name to instance.
        """
        ctx = self._setup_brittle(aug_dict)

        # Work units: 2 scoring passes + display-info + 3 visualizers.
        self._progress_inst.add_total(6)

        scored = self._score_severities(ctx)
        b_result, correct_before, correct_before_wrong_after = \
            self._build_brittleness_results(scored)
        self._results = self._render_brittle_outputs(
            ctx, scored, b_result, correct_before, correct_before_wrong_after
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
        ground_truths = self._ordered_ground_truth_df[self._ground_truth_label].tolist()
        _, test_loader = self._load_images(image_paths, ground_truths)
        # KIV: set a random seed here manually; if we want to manually set it then we'll need to change this
        np.random.seed(42)

        model = self._unwrap_model()

        class_names_arg = self._input_arguments.get('class_names') or None
        class_names = handle_class_names_arg(class_names_arg, model)
        print("Class names:", class_names)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            device=device,
            class_names=class_names,
            class_names_int=class_names_int,
            aug_name=aug_name,
            aug_class=aug_class,
            severities=severities,
            loader_A=loader_A,
            loader_B=loader_B,
        )

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
        imgs_A, probs_A, labels = collect_probs(ctx.model, ctx.loader_A, ctx.device)
        self._progress_inst.update(1)
        imgs_B, probs_B, _ = collect_probs(ctx.model, ctx.loader_B, ctx.device)
        self._progress_inst.update(1)
        return _BrittleScored(
            imgs_A=imgs_A, probs_A=probs_A,
            imgs_B=imgs_B, probs_B=probs_B,
            labels=labels,
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
        probs_A, probs_B, labels = scored.probs_A, scored.probs_B, scored.labels
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
            results=results_all_sorted,
            probs_A=probs_A,
            probs_B=probs_B,
            labels=labels,
        )

        correct_before = [r for r in b_result.results if r.predA == r.label]
        correct_before_wrong_after = [
            r for r in b_result.results if r.predA == r.label and r.predB != r.label
        ]
        return b_result, correct_before, correct_before_wrong_after

    def _render_brittle_outputs(self, ctx, scored, b_result,
                                correct_before, correct_before_wrong_after):
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

        top_k_indices = [item.index for item in correct_before][
            :min(TOPK_SAFE, len(b_result.results))
        ]
        display_info = self._loop_for_display_info(
            ctx.severities,
            ctx.aug_class,
            ctx.image_paths,
            ctx.ground_truths,
            ctx.aug_name,
            top_k_indices,
            ctx.class_names,
            (scored.imgs_A, scored.imgs_B),
            (scored.probs_A, scored.probs_B),
        )
        output_results.update({"display_info": display_info})
        self._progress_inst.update(1)

        aug_dir = self._output_folder / ctx.aug_name
        mpl_dir = aug_dir / "matplotlib"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        plotly_dir = aug_dir / "plotly"
        plotly_dir.mkdir(parents=True, exist_ok=True)

        K = min(TOPK, len(correct_before))
        mpl_path, mpl_frag_paths = visualize_topk_matplotlib(
            correct_before, b_result, scored.imgs_A, scored.imgs_B,
            K=K, class_names=ctx.class_names_int, transform=None,
            directory=mpl_dir, image_paths=ctx.image_paths,
        )
        self._progress_inst.update(1)
        plotly_path = visualize_topk_without_plotly(
            correct_before, b_result, scored.imgs_A, scored.imgs_B,
            K=K, class_names=ctx.class_names_int, transform=None,
            directory=plotly_dir, image_paths=ctx.image_paths,
        )
        self._progress_inst.update(1)
        html_path = visualize_in_html(
            correct_before_wrong_after, b_result, scored.imgs_A, scored.imgs_B,
            class_names=ctx.class_names_int, transform=None,
            directory=plotly_dir, image_paths=ctx.image_paths,
        )
        self._progress_inst.update(1)

        output_results.update({
            "matplotlib_image_path": str(mpl_path.relative_to(self._output_folder)),
            "plotly_image_path": str(plotly_path.relative_to(self._output_folder)),
            "html_carousel_path": str(html_path.relative_to(self._output_folder)),
            "matplotlib_fragment_paths": [
                str(x.relative_to(self._output_folder)) for x in mpl_frag_paths
            ],
            "dataset_size": len(ctx.image_paths),
            "class_names": ctx.class_names,
        })
        return output_results

    def _loop_for_display_info(
        self,
        severities,
        aug_class,
        image_paths,
        ground_truths,
        aug_name,
        top_k_indices,
        class_names,
        imgs_by_severity,
        probs_by_severity,
    ):
        """
        Build the top-K display rows for the before/after severities.

        For each severity and each selected sample, saves the exact image the
        model scored and reuses its prediction from the same pass, so the shown
        image and predicted label can never disagree and no second
        corrupt+inference is run.

        Args:
            severities (tuple): ``(before, after)`` severity specifiers, aligned
                with ``imgs_by_severity`` / ``probs_by_severity``.
            aug_class: The ``Augmentation`` instance (for severity naming).
            image_paths (List[str]): All image file paths.
            ground_truths (List): Ground-truth label per image.
            aug_name (str): Name of the augmentation (for the save sub-path).
            top_k_indices (List[int]): Sample indices to display.
            class_names (Dict[str, str]): Mapping from class index to display name.
            imgs_by_severity (tuple): ``(vb  , imgs_B)`` CHW image tensors from
                the scored passes, indexed by sample.
            probs_by_severity (tuple): ``(probs_A, probs_B)`` per-sample class
                probabilities from the same passes.

        Returns:
            List[Dict[str, List[str]]]: One single-key dict per (severity, rank),
                mapping ``severity_<name>_number_<k>`` to
                ``[relative_image_path, ground_truth_name, predicted_name]``.
        """
        display_info = []
        for pos, s_idx in enumerate(severities):
            severity = aug_class.determine_severity(s_idx)
            corrupted_dir = Path(aug_name) / f"severity_{severity}"
            imgs = imgs_by_severity[pos]
            probs = probs_by_severity[pos]

            for i, display_idx in enumerate(top_k_indices):
                # Pull the exact pixels the model scored and its matching
                # prediction from the same pass (shuffle=False, so display_idx
                # aligns), instead of re-corrupting and re-inferring.
                image_np = imgs[display_idx].detach().cpu().numpy()
                image_path = self._save_one_image(
                    image_np, str(corrupted_dir), Path(str(image_paths[display_idx])).name
                )
                prediction = int(probs[display_idx].argmax().item())
                ground_truth = ground_truths[display_idx]

                random_display = [
                    str(Path(image_path).relative_to(self._output_folder)),
                    class_names[str(ground_truth)],
                    class_names[str(prediction)],
                ]
                display_info.append({f"severity_{severity}_number_{i+1}": random_display})

        return display_info
    
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