import logging
from pathlib import Path, PurePath
from typing import Dict, List, Tuple, Union, Callable, Any, Optional
import copy
import shutil
import time

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
import pandas as pd 
import json
import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.graph_objects as go

from .pycocotools_fdet.coco import COCO
from .pycocotools_fdet.cocoeval import COCOeval, average_curve_dataframes, plot_curve_dataframe
from . import cvrob_algo_common
from dataclasses import dataclass


@dataclass
class _OdBcCtx:
    """
    Everything ``_augmentation_bc_method`` resolves once up front and threads to
    its phase helpers, so no single method has to re-derive the setup.

    Attributes:
        model: The unwrapped torch detection model (or API URL string).
        device (torch.device): Device the model runs on.
        class_names (dict): Class index (as str) -> display name.
        image_paths (list): All image file paths.
        ground_truths (list): Per-image ground-truth annotations (class ids resolved).
        test_loader: Clean detection data loader (shuffle off).
        display_idx (int): Index of the single sample shown per severity.
        aug_methods (list): Augmentation names requested (or ``["all"]``).
        cocoGt (COCO): Ground-truth COCO index, built once and shared across
            all augmentations/severities/epochs (GT is constant for the run).
    """
    model: object
    device: object
    class_names: dict
    image_paths: list
    ground_truths: list
    test_loader: object
    display_idx: int
    aug_methods: list
    cocoGt: object


@dataclass
class _SeverityOutputs:
    """
    The per-severity pieces one ``(aug, severity)`` pass contributes to its
    augmentation's aggregate results.

    Attributes:
        cr (dict): ``{"map": avg_map, **avg_stats}`` classification-report row.
        matrix: Epoch-averaged detection-matching matrix.
        display (list): Display-info row for this severity.
        cm_paths (list): ``[png_rel_path, html_rel_path]`` of the matrix figure.
        coco_graphs (list): Relative paths of the per-severity COCO curve figures.
        coco_summary (dict): Epoch-averaged COCO/F-beta summary.
        fbeta_df: Epoch-averaged F-beta curve DataFrame (for the overall plot).
        pr_df: Epoch-averaged P-R curve DataFrame (for the overall plot).
    """
    cr: dict
    matrix: object
    display: list
    cm_paths: list
    coco_graphs: list
    coco_summary: dict
    fbeta_df: object
    pr_df: object


@dataclass
class _Series:
    """
    One line on a per-class severity panel, backend-agnostic.

    Attributes:
        col (str): Column in the panel DataFrame to plot on the y-axis.
        label (str): Legend label.
        color: Explicit color, or ``None`` to auto-assign (matplotlib color cycle
            / plotly qualitative palette by series index).
        marker: matplotlib marker (e.g. ``'o'``); its presence also switches the
            plotly trace to ``lines+markers``. ``None`` means a plain line.
        dash (bool): Draw dashed (matplotlib ``'--'`` / plotly ``dash='dash'``).
    """
    col: str
    label: str
    color: object = None
    marker: object = None
    dash: bool = False


@dataclass
class _PanelSpec:
    """
    A per-class "metric/count vs severity" panel, rendered to both backends.

    Attributes:
        title (str): Figure title (shared by both backends).
        ylabel (str): matplotlib y-axis label.
        series (list): The ``_Series`` lines to draw.
        stem (str): Output filename stem (``<stem>.png`` / ``<stem>.html``).
        nan_masks (dict | None): Optional ``col -> boolean mask`` marking severities
            whose value was NaN (drawn as ``x`` markers at y=0 in the line color).
        ylim (tuple | None): matplotlib y-limits.
        plotly_range (tuple | None): plotly y-axis range (kept separate from
            ``ylim`` to preserve the original per-backend limits exactly).
        ylabel_plotly (str | None): plotly y-axis label; falls back to ``ylabel``.
    """
    title: str
    ylabel: str
    series: list
    stem: str
    nan_masks: object = None
    ylim: object = None
    plotly_range: object = None
    ylabel_plotly: object = None


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
    The Plugin(OD Augmentation by Class Algorithm) class specifies methods in generating results for algorithm
    """

    # Some information on plugin
    _name: str = "OD Augmentation by Class Algorithm"
    _description: str = "This algorithm shows the relationship between certain augmentations and the class statistics inferred by the model"
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
        
        # Apply user defined parameters to default parameters
        aug_dict = self._resolve_aug_dict()
        self._augmentation_bc_method(aug_dict)
        # Update progress (For 100% completion)
        #self._progress_inst.update(1)

    def _get_display_info_for_severity(
        self,
        model,
        image_paths: list,
        ground_truths: list,
        aug_class,
        severity_name: str,
        display_idx: int,
        class_names: dict,
        corrupted_dir: Path,
    ):
        """
        Obtain the display image for one severity, run inference on it, save both
        the plain image and the annotated overlay, and return the display-info list.

        Returns:
            list: [rel_image_path, ground_truth, prediction_dict, rel_overlay_path]
        """
        display_image = self._get_one_corrupted_image_direct(
            image_paths, ground_truths, aug_class, severity_name, display_idx
        )
        image_path = self._save_one_image(display_image, str(corrupted_dir), display_idx)
        prediction = get_prediction_from_image(model, display_image, self._device)
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
        return [
            str(Path(image_path).relative_to(self._output_folder)),
            ground_truth,
            drawn_prediction,
            str(Path(image_path2).relative_to(self._output_folder)),
        ]

    def _augmentation_bc_method(self, aug_dict):
        """
        Orchestrate the per-class detection evaluation across all augmentations.

        Resolves setup once, then evaluates each in-scope augmentation (per-class
        COCO stats, matrices, curves and display samples per severity) and
        assembles the aggregate results dict into ``self._results``. Progress
        advances once per augmentation that actually runs.

        Args:
            aug_dict (Dict[str, Any]): Mapping of augmentation name to instance.
        """
        ctx = self._setup_odbc(aug_dict)

        # The clean ("None") severity is identical for every augmentation: same
        # loader, corruption skipped, epochs forced to 1. Evaluate it once and
        # reuse the result across augmentations instead of paying a full
        # inference + COCO-eval pass per augmentation. See _run_one_severity.
        self._clean_severity_outputs = None

        num_augs_to_run = sum(
            1 for aug_name in aug_dict if self._should_run_aug(aug_name, ctx.aug_methods)
        )
        self._progress_inst.add_total(num_augs_to_run)

        combined_results = []
        for aug_name, aug_class in aug_dict.items():
            if not self._should_run_aug(aug_name, ctx.aug_methods):
                continue

            combined_results.append(self._run_one_aug(ctx, aug_name, aug_class))
            self._progress_inst.update(1)
            print()

        self._results = {
            "results": combined_results,
            "augmentation_names": [x["Augmentation"] for x in combined_results],
            "class_names": ctx.class_names,
            "dataset_size": len(ctx.image_paths),
        }

    def _setup_odbc(self, aug_dict):
        """
        Resolve all inputs the evaluation needs before any augmentation runs.

        Unwraps the model, resolves class names and URL-backed augmentations,
        picks the device, resolves ground-truth class ids, loads the clean loader,
        chooses the display sample, and writes + parses the COCO ground truth once.

        Args:
            aug_dict (Dict[str, Any]): Mapping of augmentation name to instance.

        Returns:
            _OdBcCtx: The resolved context threaded through the phase helpers.
        """
        model = self._unwrap_model()
        aug_methods = self._get_aug_methods()
        class_names_arg = self._input_arguments['class_names'] or None
        class_names = handle_class_names_arg(class_names_arg, model)

        print("Augmentation methods:", aug_methods)
        if 'url' in aug_dict and 'http' in aug_dict['url']:
            handle_url_algos(aug_dict, aug_methods)

        print("Class names:", class_names)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        ground_truths = self._resolve_class_ids(self._ordered_ground_truth, class_names)
        self._ordered_ground_truth = ground_truths

        image_paths: list[str] = self._data_instance.get_data()["image_directory"].tolist()
        _, test_loader = self._load_images_objdet(image_paths, ground_truths)
        # KIV: set a random seed here manually; if we want to make it configurable
        # later we'll change this.
        np.random.seed(42)
        display_idx = np.random.choice(len(image_paths))

        # Write the COCO ground truth into the output folder (not the CWD) and
        # parse it once here: the GT is constant for the whole run, so there's no
        # reason to reload/re-index it per epoch inside the severity loop.
        gt_json = str(self._save_folder / 'ground_truths.json')
        create_coco_gt(image_paths, ground_truths, class_names, gt_json)
        cocoGt = COCO(gt_json)

        return _OdBcCtx(
            model=model,
            device=device,
            class_names=class_names,
            image_paths=image_paths,
            ground_truths=ground_truths,
            test_loader=test_loader,
            display_idx=display_idx,
            aug_methods=aug_methods,
            cocoGt=cocoGt,
        )

    def _run_one_aug(self, ctx, aug_name, aug_class):
        """
        Evaluate a single augmentation across all its severities.

        Runs each severity, collects the per-severity outputs, then builds the
        per-class metric plots and the overall COCO curve overlays.

        Args:
            ctx (_OdBcCtx): Resolved run context.
            aug_name (str): Name of this augmentation.
            aug_class: The ``Augmentation`` instance.

        Returns:
            dict: The per-augmentation results entry for ``combined_results``.
        """
        severities = ["None"] + aug_class.severities
        if aug_name == "None":
            severities = ["None"]

        per_sev = [
            self._run_one_severity(ctx, aug_name, aug_class, severity_idx, severity_name)
            for severity_idx, severity_name in enumerate(severities)
        ]

        crs = [o.cr for o in per_sev]
        fbeta_df_list = [o.fbeta_df for o in per_sev]
        pr_df_list = [o.pr_df for o in per_sev]

        path_dict = self._detection_method(
            crs, severities, ctx.class_names, Path(aug_name), aug_name
        )
        overall_coco_path_dict = self._overall_coco_path_method(
            fbeta_df_list,
            pr_df_list,
            severities,
            Path(aug_name),
            iou_thres=f"iou={self._iou_thres:.2f}",
            fbeta_metric=None,
        )

        return {
            "Augmentation": aug_name,
            "display_info": {n: o.display for n, o in zip(severities, per_sev)},
            "classification_report": crs,
            "conf_matrix": [o.matrix for o in per_sev],
            "plot_paths": path_dict,
            "confusion_matrix": {n: o.cm_paths for n, o in zip(severities, per_sev)},
            "coco_graphs": {n: o.coco_graphs for n, o in zip(severities, per_sev)},
            "coco_summary": {n: o.coco_summary for n, o in zip(severities, per_sev)},
            "coco_graphs_overall": overall_coco_path_dict,
        }

    def _run_one_severity(self, ctx, aug_name, aug_class, severity_idx, severity_name):
        """
        Evaluate one ``(aug, severity)`` pass and package its contributions.

        Resolves the epoch count, runs the epoch-averaged COCO evaluation, builds
        the display sample, and saves the detection matrix figure.

        Args:
            ctx (_OdBcCtx): Resolved run context.
            aug_name (str): Name of the augmentation.
            aug_class: The ``Augmentation`` instance.
            severity_idx (int): Ordinal of this severity (used for seeding).
            severity_name (str): Severity label (``"None"`` for the clean pass).

        Returns:
            _SeverityOutputs: The per-severity pieces for the aggregate results.
        """
        print('severity idx', severity_idx, 'severity_name', severity_name)

        # The clean pass is augmentation-independent (see _augmentation_bc_method):
        # compute it for the first augmentation, then reuse that _SeverityOutputs
        # for every subsequent one. Its saved artifacts (display/matrix/curve PNGs)
        # live under the first augmentation's severity_None/ folder; the reused
        # results reference that single copy, which is correct since the clean
        # images are identical regardless of augmentation.
        if severity_name == "None" and self._clean_severity_outputs is not None:
            print("Reusing cached clean (None) severity outputs")
            return self._clean_severity_outputs

        corrupted_dir = Path(aug_name) / f"severity_{severity_name}"

        num_epochs = self._input_arguments.get('num_epochs') or 1
        num_epochs = num_epochs if severity_name != "None" else 1
        num_epochs = 1 if num_epochs is None else num_epochs
        num_epochs = 1 if aug_class.deterministic else num_epochs

        save_dir_coco = self._save_folder / corrupted_dir
        save_dir_coco.mkdir(parents=True, exist_ok=True)
        avg_stats, avg_matrix, avg_map, avg_summary, coco_imgs, coco_dfs = self._run_severity_epochs_coco(
            ctx.model, ctx.test_loader, aug_class, severity_name, severity_idx,
            num_epochs, ctx.class_names,
            ctx.cocoGt, ctx.image_paths, save_dir_coco
        )

        display = self._get_display_info_for_severity(
            ctx.model, ctx.image_paths, ctx.ground_truths, aug_class, severity_name,
            ctx.display_idx, ctx.class_names, corrupted_dir,
        )

        save_path, save_path_html = self._save_detection_matrix_path(
            avg_matrix, ctx.class_names, corrupted_dir
        )
        cm_paths = [
            str(Path(save_path).relative_to(self._output_folder)),
            str(Path(save_path_html).relative_to(self._output_folder)),
        ]
        coco_graphs = [str(x.relative_to(self._output_folder)) for x in coco_imgs]

        outputs = _SeverityOutputs(
            cr={"map": avg_map, **avg_stats},
            matrix=avg_matrix,
            display=display,
            cm_paths=cm_paths,
            coco_graphs=coco_graphs,
            coco_summary=avg_summary,
            fbeta_df=coco_dfs[0],
            pr_df=coco_dfs[1],
        )

        # Cache the clean pass so later augmentations skip re-computing it.
        if severity_name == "None":
            self._clean_severity_outputs = outputs

        return outputs

    def _build_combined_df(self, data: list, severities: list) -> "pd.DataFrame":
        """
        Convert per-severity detection stats into a flat DataFrame with columns:
        severity, class, precision, recall, f1_score, TP, FP, FN, support, map.
        """
        rows = []
        for severity, stats in zip(severities, data):
            map_overall = stats.get("map", None)
            if "per_class" in stats:
                per_class = stats["per_class"]
            else:
                per_class = {k: v for k, v in stats.items() if k != "map"}
            per_class = normalize_per_class(per_class)

            for class_name, metrics in per_class.items():
                rows.append({
                    "severity":  severity,
                    "class":     class_name,
                    "precision": metrics["precision"],
                    "recall":    metrics["recall"],
                    "f1_score":  metrics["f1_score"],
                    "TP":        metrics["TP"],
                    "FP":        metrics["FP"],
                    "FN":        metrics["FN"],
                    "support":   metrics["support"],
                    "map":       metrics["map"], #map
                    "map_overall":     map_overall,
                })
        df = pd.DataFrame(rows)
        df['pred_population']   = df['TP'] + df['FP']
        df['actual_population'] = df['TP'] + df['FN']
        return df

    def _render_class_panel(self, df: "pd.DataFrame", spec: "_PanelSpec", save_dir: Path):
        """
        Render one per-class "value vs severity" panel to matplotlib and plotly.

        Draws each ``spec.series`` line against ``df['severity']`` on both
        backends, optionally overlaying ``x`` markers at y=0 for NaN severities
        (``spec.nan_masks``) in the matching line color, and applies the shared
        figure styling. This is the single backend implementation the three
        ``_plot_class_*`` panels delegate to.

        Args:
            df (pd.DataFrame): Panel data; must contain ``severity`` plus every
                ``series.col``. Rows are the x-axis in order.
            spec (_PanelSpec): The panel definition (series, title, limits, stem).
            save_dir (Path): Directory to write ``<stem>.png`` / ``<stem>.html``.

        Returns:
            Tuple[Path, Path]: The saved ``(png_path, html_path)``.
        """
        severity_order = df['severity'].tolist()
        pos_map = {v: i for i, v in enumerate(severity_order)}
        plotly_colors = px.colors.qualitative.Plotly

        # --- matplotlib ---
        fig, ax = plt.subplots(figsize=(10, 6))
        line_colors = {}
        for s in spec.series:
            kwargs = {'label': s.label}
            if s.color is not None:
                kwargs['color'] = s.color
            if s.marker is not None:
                kwargs['marker'] = s.marker
            if s.dash:
                kwargs['linestyle'] = '--'
            line, = ax.plot(df['severity'], df[s.col], **kwargs)
            line_colors[s.col] = line.get_color()

        if spec.nan_masks:
            for s in spec.series:
                mask = spec.nan_masks.get(s.col)
                if mask is not None and mask.any():
                    x_pos = [pos_map[v] for v in df.loc[mask, 'severity']]
                    ax.scatter(x_pos, [0] * len(x_pos), marker='x',
                               color=line_colors[s.col], alpha=0.8, zorder=5,
                               label=f'{s.label} (NaN\u21920)')

        ax.set_title(spec.title)
        ax.set_xlabel("severity")
        ax.set_ylabel(spec.ylabel)
        ax.legend()
        if spec.ylim is not None:
            ax.set_ylim(*spec.ylim)
        plt.xticks(rotation=45)

        png_path = save_dir / f"{spec.stem}.png"
        fig.savefig(png_path, bbox_inches="tight")
        plt.close()

        # --- plotly ---
        fig_html = go.Figure()
        series_colors = {}
        for i, s in enumerate(spec.series):
            color = s.color if s.color is not None else plotly_colors[i % len(plotly_colors)]
            series_colors[s.col] = color
            line_kw = {'color': color}
            if s.dash:
                line_kw['dash'] = 'dash'
            fig_html.add_trace(go.Scatter(
                x=df['severity'], y=df[s.col],
                mode='lines+markers' if s.marker is not None else 'lines',
                name=s.label, line=line_kw,
            ))

        if spec.nan_masks:
            for s in spec.series:
                mask = spec.nan_masks.get(s.col)
                if mask is not None and mask.any():
                    nan_sev = df.loc[mask, 'severity'].tolist()
                    fig_html.add_trace(go.Scatter(
                        x=nan_sev, y=[0] * len(nan_sev), mode='markers',
                        marker=dict(symbol='x', size=12, color=series_colors[s.col],
                                    line=dict(width=2)),
                        name=f'{s.label} (NaN\u21920)', showlegend=True,
                    ))

        fig_html.update_layout(
            title=spec.title,
            width=1600, height=900, xaxis_title="severity",
            yaxis_title=spec.ylabel_plotly or spec.ylabel,
            font=dict(size=20), title_font_size=24,
        )
        if spec.plotly_range is not None:
            fig_html.update_yaxes(range=list(spec.plotly_range))
        html_path = save_dir / f"{spec.stem}.html"
        fig_html.write_html(str(html_path))

        return png_path, html_path

    def _plot_class_metrics(
        self,
        plot_df: "pd.DataFrame",
        nan_masks: dict,
        save_dir: Path,
        aug_name: str,
        class_name: str,
    ):
        """
        Plot precision / recall / f1 / map over severity for one class.

        Returns:
            (png_path, html_path)
        """
        spec = _PanelSpec(
            title=f"{aug_name} - {class_name}",
            ylabel="score",
            ylabel_plotly="metric",
            stem=f"{class_name}_metrics",
            nan_masks=nan_masks,
            ylim=(-0.1, 1.1),
            plotly_range=(-0.1, 1.1),
            series=[
                _Series(col='precision', label='precision'),
                _Series(col='recall', label='recall'),
                _Series(col='f1_score', label='f1'),
                _Series(col='map', label='map'),
            ],
        )
        return self._render_class_panel(plot_df, spec, save_dir)

    def _plot_class_counts(
        self,
        class_df: "pd.DataFrame",
        save_dir: Path,
        aug_name: str,
        class_name: str,
    ):
        """
        Plot TP / FP / FN counts over severity for one class (matplotlib + plotly).
        TN is undefined for object detection and is omitted.

        Returns:
            (png_path_cm, html_path_cm)
        """
        count_cols = ['TP', 'FP', 'FN']
        count_colors = {'TP': '#2ca02c', 'FP': '#ff7f0e', 'FN': '#d62728'}
        count_plot_df = class_df[['severity'] + count_cols].fillna(0)

        spec = _PanelSpec(
            title=f"{aug_name} - {class_name} (TP / FP / FN counts)",
            ylabel="count",
            stem=f"{class_name}_counts",
            series=[
                _Series(col=col, label=col, color=count_colors[col], marker='o')
                for col in count_cols
            ],
        )
        return self._render_class_panel(count_plot_df, spec, save_dir)

    def _plot_class_population(
        self,
        class_df: "pd.DataFrame",
        save_dir: Path,
        aug_name: str,
        class_name: str,
    ):
        """
        Plot actual_population (TP+FN = GT boxes) vs pred_population (TP+FP = predicted
        boxes) over severity for one class (matplotlib + plotly).

        Returns:
            (png_path_pop, html_path_pop)
        """
        pop_df = class_df[['severity', 'TP', 'FP', 'FN']].fillna(0).copy()
        pop_df['actual_population'] = pop_df['TP'] + pop_df['FN']
        pop_df['pred_population']   = pop_df['TP'] + pop_df['FP']
        y_max = pop_df[['actual_population', 'pred_population']].max().max()

        # Note: matplotlib and plotly use slightly different lower bounds here
        # (-5 vs 5), preserved from the original per-backend limits.
        spec = _PanelSpec(
            title=f"{aug_name} - {class_name} (population)",
            ylabel="count",
            stem=f"{class_name}_population",
            ylim=(-5, y_max + 5),
            plotly_range=(5, y_max + 5),
            series=[
                _Series(col='actual_population', label='actual population (GT boxes)',
                        color='steelblue', marker='o'),
                _Series(col='pred_population', label='pred population (pred boxes)',
                        color='darkorange', marker='s', dash=True),
            ],
        )
        return self._render_class_panel(pop_df, spec, save_dir)

    def _plot_map(
        self,
        combined_df: "pd.DataFrame",
        save_dir: Path,
        aug_name: str,
    ):
        """
        Plot mAP@iouthres over severity (matplotlib + plotly).

        Returns:
            (map_png, map_html)
        """
        map_df = combined_df[["severity", "map_overall"]].drop_duplicates().reset_index(drop=True)
        map_nan_mask = map_df['map_overall'].isna()
        plot_map_df = map_df.copy()
        plot_map_df['map_overall'] = plot_map_df['map_overall'].fillna(0)

        map_severity_order = plot_map_df['severity'].tolist()
        map_pos_map = {v: i for i, v in enumerate(map_severity_order)}

        # --- matplotlib ---
        fig, ax = plt.subplots(figsize=(10, 6))
        map_line, = ax.plot(plot_map_df['severity'], plot_map_df['map_overall'])
        map_color = map_line.get_color()

        if map_nan_mask.any():
            x_pos = [map_pos_map[s] for s in plot_map_df.loc[map_nan_mask, 'severity']]
            ax.scatter(x_pos, [0] * len(x_pos), marker='x', color=map_color,
                       alpha=0.8, zorder=5, label='map (NaN\u21920)')

        ax.set_title(f"{aug_name} mAP@{self._iou_thres*100}")
        ax.set_xlabel("severity")
        ax.set_ylabel(f"mAP@{self._iou_thres*100}")
        ax.set_ylim(-0.1, 1.1)
        ax.legend()
        plt.xticks(rotation=45)

        map_png = save_dir / "map.png"
        fig.savefig(map_png, bbox_inches="tight")
        plt.close()

        # --- plotly ---
        fig_html = px.line(plot_map_df, x='severity', y='map_overall',
                           title=f"{aug_name} mAP@{self._iou_thres*100}")
        if map_nan_mask.any():
            nan_severities = plot_map_df.loc[map_nan_mask, 'severity'].tolist()
            fig_html.add_trace(go.Scatter(
                x=nan_severities, y=[0] * len(nan_severities),
                mode='markers',
                marker=dict(symbol='x', size=12,
                            color=px.colors.qualitative.Plotly[0],
                            line=dict(width=2)),
                name='map (NaN\u21920)', showlegend=True
            ))

        fig_html.update_layout(
            width=1600, height=900, xaxis_title="severity", yaxis_title="map",
            font=dict(size=20), title_font_size=24
        )
        map_html = save_dir / "map.html"
        fig_html.write_html(str(map_html))

        return map_png, map_html

    def _plot_all_classes_population_barchart(
        self,
        combined_df: "pd.DataFrame",
        class_names: dict,
        save_dir: Path,
        aug_name: str,
    ):
        """
        Stacked bar chart showing pred_population (TP+FP) per foreground class across
        severities — the object-detection equivalent of the classification
        all_classes_proportions_barchart.

        Background (class_id == '0') is excluded.
        Both matplotlib (.png) and plotly (.html) versions are produced.

        Returns:
            (plt_path, fx_path)
        """
        foreground_names = [v for k, v in class_names.items() if str(k) != '0']

        # pred_population is already in combined_df (TP + FP computed upstream).
        # We need one row per (severity, class), so drop any map duplicates first.
        pop_df = (
            combined_df[combined_df['class'].isin(foreground_names)]
            [['severity', 'class', 'pred_population']]
            .drop_duplicates()
        )

        plot_df = pop_df.pivot(index='severity', columns='class', values='pred_population').fillna(0)

        # --- matplotlib ---
        fig, ax = plt.subplots(figsize=(16, 9))
        colors = plt.cm.jet(np.linspace(0, 1, len(foreground_names)))

        bottom = None
        for idx, class_name in enumerate(foreground_names):
            if class_name not in plot_df.columns:
                continue
            vals = plot_df[class_name].values
            if bottom is None:
                ax.bar(plot_df.index, vals, label=class_name, color=colors[idx])
                bottom = vals.copy()
            else:
                ax.bar(plot_df.index, vals, label=class_name, color=colors[idx], bottom=bottom)
                bottom = bottom + vals

        ax.set_xlabel('severity', fontsize=20)
        ax.set_ylabel('predicted box count', fontsize=20)
        ax.tick_params(axis='x', labelsize=18, rotation=45)
        ax.tick_params(axis='y', labelsize=18)
        ax.legend(title='class', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=18)
        ax.set_title(
            f'{aug_name}: Predicted Box Counts per Class vs Augmentation Severity',
            fontsize=24, pad=30
        )
        plt.subplots_adjust(left=0.15, right=0.8, top=0.88, bottom=0.3)

        plt_path = save_dir / "all_classes_proportions_barchart.png"
        fig.savefig(plt_path, bbox_inches="tight")
        plt.close()

        # --- plotly ---
        long_df = plot_df.reset_index().melt(
            id_vars='severity', var_name='class', value_name='pred_count'
        )
        n = len(foreground_names)
        color_scale = (
            px.colors.sample_colorscale("Jet", [i / max(n - 1, 1) for i in range(n)])
        )
        fx = px.bar(
            long_df, x='severity', y='pred_count', color='class',
            title=f'{aug_name}: Predicted Box Counts per Class vs Augmentation Severity',
            color_discrete_sequence=color_scale,
        )
        fx.update_layout(
            barmode='stack',
            yaxis_title='predicted box count',
            xaxis_title='severity',
        )
        fx_path = save_dir / "all_classes_proportions_plotly_barchart.html"
        fx.write_html(str(fx_path))

        return plt_path, fx_path

    def _detection_method(
        self,
        data,
        severities,
        class_names,
        subfolder_name,
        aug_name
    ):
        plt.rcParams.update({'font.size': 18})

        save_dir0 = self._save_folder / subfolder_name
        save_dir0.mkdir(parents=True, exist_ok=True)
        save_dir = save_dir0 / "figures"
        save_dir.mkdir(parents=True, exist_ok=True)

        combined_df = self._build_combined_df(data, severities)
        print(combined_df.head())

        metric_cols = ['precision', 'recall', 'f1_score', 'map']
        path_dict = {}
        temp = {}

        background_names = {v for k, v in class_names.items() if str(k) == '0'}

        for class_name in class_names.values():
            if class_name in background_names:
                continue

            class_df = combined_df[combined_df["class"] == class_name].reset_index(drop=True)

            nan_masks = {col: class_df[col].isna() for col in metric_cols}
            plot_df = class_df.copy()
            plot_df[metric_cols] = plot_df[metric_cols].fillna(0)

            png_path,     html_path     = self._plot_class_metrics(
                plot_df, nan_masks, save_dir, aug_name, class_name)
            png_path_cm,  html_path_cm  = self._plot_class_counts(
                class_df, save_dir, aug_name, class_name)
            png_path_pop, html_path_pop = self._plot_class_population(
                class_df, save_dir, aug_name, class_name)

            temp[class_name] = [
                str(png_path.relative_to(self._output_folder)),
                str(png_path_cm.relative_to(self._output_folder)),
                str(png_path_pop.relative_to(self._output_folder)),
                str(html_path.relative_to(self._output_folder)),
                str(html_path_cm.relative_to(self._output_folder)),
                str(html_path_pop.relative_to(self._output_folder)),
            ]

        path_dict["class_plot"] = temp

        plt_path, fx_path = self._plot_all_classes_population_barchart(
            combined_df, class_names, save_dir, aug_name
        )
        path_dict['matplotlib_image_path'] = str(plt_path.relative_to(self._output_folder))
        path_dict['plotly_image_path']     = str(fx_path.relative_to(self._output_folder))

        map_png, map_html = self._plot_map(combined_df, save_dir, aug_name)
        path_dict["map_plot"] = [
            str(map_png.relative_to(self._output_folder)),
            str(map_html.relative_to(self._output_folder)),
        ]

        return path_dict

    def _save_detection_matrix_path(
        self,
        det_matrix,
        class_names_dir,
        corrupted_dir
    ):
        class_names = list(class_names_dir.values())
        n_classes = len(class_names)
        fig_size = max(8, n_classes * 1.5)

        # =========================
        # MATPLOTLIB
        # =========================

        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        im = ax.imshow(det_matrix, cmap="Blues")
        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

        plt.setp(
            ax.get_xticklabels(),
            rotation=45,
            ha="right"
        )
        # text values
        for i in range(n_classes):
            for j in range(n_classes):

                ax.text(
                    j,
                    i,
                    f"{det_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8
                )

        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Ground truth class")
        ax.set_title("Detection Matching Matrix")
        fig.colorbar(im)
        plt.tight_layout()
        save_dir = self._save_folder / corrupted_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "detection_matrix.png"
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )
        plt.close(fig)
        print(f"Saved to {save_path} [matplotlib]")

        # =========================
        # PLOTLY
        # =========================

        fig = go.Figure(
            data=go.Heatmap(
                z=det_matrix,
                x=class_names,
                y=class_names,
                colorscale="Blues",
                colorbar=dict(title="Matches"),
                text=np.round(det_matrix, 2),
                texttemplate="%{text}",
                textfont={"size": 12}
            )
        )
        fig.update_layout(
            title="Detection Matching Matrix",
            width=max(600, n_classes * 80),
            height=max(600, n_classes * 80),
            xaxis_title="Predicted class",
            yaxis_title="Ground truth class",
        )
        fig.update_xaxes(tickangle=45)
        save_path_html = save_dir / "detection_matrix.html"
        fig.write_html(str(save_path_html))
        print(f"Saved to {save_path_html} [plotly]")
        return save_path, save_path_html

    def _extract_coco_ap(self, cocoEval): #class_names
        """
        Read overall and per-class Average Precision from a COCO accumulator at
        the single configured ``self._iou_thres``.

        Replaces the former TorchMetrics ``MeanAveragePrecision`` pass: after
        ``cocoEval.accumulate()`` the ``(T, R, K, A, M)`` precision tensor holds
        interpolated precision per (IoU, recall, class, area, maxDet). This
        slices the single IoU row matching ``self._iou_thres``, area ``all`` and
        ``maxDets=100`` (matching COCO's ``AP_50`` convention), then averages
        over the recall axis per class. Absent categories (all ``-1``) yield NaN.

        Args:
            cocoEval (COCOeval): Accumulated evaluator (``accumulate()`` already run).

        Returns:
            Tuple[float, dict]: Overall mAP (mean of valid per-class AP, or -1.0
                if none valid) and ``{class_name: ap}`` per-class AP.

        Raises:
            ValueError: If ``self._iou_thres`` is not on COCO's IoU grid.
        """
        p = cocoEval.params
        precision = cocoEval.eval['precision']  # (T, R, K, A, M)

        t_idx = np.where(np.isclose(p.iouThrs, self._iou_thres))[0]
        if t_idx.size == 0:
            raise ValueError(
                f"iou_thres={self._iou_thres} is not on the COCO IoU grid "
                f"({p.iouThrs[0]:.2f}:{p.iouThrs[-1]:.2f} step 0.05); pick a "
                f"value on the grid so per-class AP can be read out."
            )
        t = t_idx.item()
        aind = p.areaRngLbl.index('all')
        mind = p.maxDets.index(100)

        # catNms is ordered by category id, matching the precision tensor's K axis.
        per_class_ap = {}
        valid_aps = []
        for k, class_name in enumerate(p.catNms):
            s = precision[t, :, k, aind, mind]
            s = s[s > -1]
            ap = float(np.mean(s)) if s.size else float("nan")
            per_class_ap[class_name] = ap
            if s.size:
                valid_aps.append(ap)

        overall = float(np.mean(valid_aps)) if valid_aps else -1.0
        return overall, per_class_ap

    def _run_severity_epochs_coco(
        self,
        model,
        test_loader,
        aug_class,
        severity_name: str,
        severity_idx: int,
        num_epochs: int,
        class_names: dict,
        cocoGt: "COCO",
        image_paths: list,
        save_dir_coco: Path,
    ):
        """
        Run ``num_epochs`` evaluation passes for one (aug, severity) combination
        and return averaged detection statistics.

        Args:
            cocoGt (COCO): The ground-truth COCO index, built once by the caller
                and shared across all severities/epochs (GT is constant).

        Returns:
            avg_stats  - per-class metrics averaged over epochs
            avg_matrix - detection-matching matrix averaged over epochs
            avg_map  - scalar mAP@iouthres averaged over valid epochs (None if none valid)
        """
        # Predictions change every epoch, so keep this file scoped to this
        # (aug, severity) output dir rather than a shared name in the CWD.
        pr_json = str(save_dir_coco / 'predictions.json')
        all_stats = []; all_matrices = []; all_maps = []; all_summaries = []
        all_fbeta_dfs = []; all_pr_dfs = []; all_cocopr_dfs = []

        for i in range(num_epochs):
            print("NUMEPOCHS", num_epochs)
            seed = 1000 * severity_idx + i
            aug_class.set_seed(seed)

            if severity_name == "None":
                corrupted_loader = test_loader
            else:
                corrupted_loader = aug_class.corr_func_dataloader(test_loader, severity_name)

            det_stats = evaluate_detection_and_create_coco_predictions(
                model, corrupted_loader, self._device, class_names, image_paths, pr_json,
                iou_thresh=self._iou_thres, score_thresh=self._score_thres,
                coco_score_threshold=0.0,  # match original create_coco_predictions default
            )
            
            t_coco_start = time.perf_counter()
            cocoDt = cocoGt.loadRes(pr_json)  # initialize COCO prediction api
            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')  # initialize COCO evaluation api
            cocoEval.evaluate()
            cocoEval.accumulateFBeta()
            cocoEval.accumulate()
            summary = cocoEval.collectSummaryResults(fbeta_betas=(1, 2), fbeta_iou_thrs=(self._iou_thres,))

            # Compute (but don't yet plot/save) the curve data for this epoch. Plotting
            # straight to fbeta_filename/pr_filename/cocopr_filename here would just have
            # each epoch overwrite the previous one's PNG, so only the last epoch would
            # ever be reflected on disk. Instead we collect every epoch's curve data and
            # plot the epoch-averaged curves once, after the loop.
            fbeta_df = cocoEval.computeFBetaCurveData(betas=[1,2], iouThr=self._iou_thres, average='macro')
            pr_df = cocoEval.computePRCurveData(average='macro')
            cocopr_df = cocoEval.computeCocoPRCurveData()  #TODO: KIV doing this by class
            per_class_report = cocoEval.generateReport(iouThr=self._iou_thres)
            print(f"COCO eval block: {time.perf_counter() - t_coco_start:.2f}s")

            for k in det_stats['per_class']:
                assert k in per_class_report
                class_report = per_class_report[k]
                for k1,v1 in class_report.items():
                    det_stats['per_class'][k][k1] = v1

            # Overall + per-class AP now come from the COCO accumulator at
            # iou_thres (single source of truth), not from TorchMetrics.
            overall_ap, per_class_ap = self._extract_coco_ap(cocoEval)
            det_stats["map"] = overall_ap
            for k in det_stats['per_class']:
                det_stats['per_class'][k]["map"] = per_class_ap.get(k, float("nan"))

            all_stats.append(det_stats["per_class"])
            all_matrices.append(det_stats["matrix"])
            all_maps.append(det_stats["map"])
            all_summaries.append(summary)
            all_fbeta_dfs.append(fbeta_df)
            all_pr_dfs.append(pr_df)
            all_cocopr_dfs.append(cocopr_df)

        avg_stats = average_detection_stats(all_stats)
        avg_matrix = np.mean(all_matrices, axis=0)
        avg_map = (
            float(np.mean([x for x in all_maps if x >= 0]))
            if any(x >= 0 for x in all_maps)
            else None
        )
        avg_summary = average_summaries(all_summaries)

        # Average the curve data across epochs, then plot/save each curve exactly once,
        # so the saved PNGs (and returned DataFrames) reflect all epochs, not just the last.
        avg_fbeta_df = average_curve_dataframes(all_fbeta_dfs)
        avg_pr_df = average_curve_dataframes(all_pr_dfs)
        avg_cocopr_df = average_curve_dataframes(all_cocopr_dfs)

        fbeta_filename = save_dir_coco / "fbeta_curve.png"
        plot_curve_dataframe(
            avg_fbeta_df, fbeta_filename,
            title=f'macro Fscores for iouThr={self._iou_thres} (avg over {num_epochs} epochs)',
            xlabel='confidence threshold', ylabel='score',
        )
        pr_filename = save_dir_coco / "pr_curve.png"
        plot_curve_dataframe(
            avg_pr_df, pr_filename,
            title=f'P-R curve (avg over {num_epochs} epochs)',
            xlabel='recall', ylabel='precision',
        )
        cocopr_filename = save_dir_coco / "cocopr_curve.png"  #TODO: KIV doing this by class
        plot_curve_dataframe(
            avg_cocopr_df, cocopr_filename,
            title=f'COCO P-R curve (avg over {num_epochs} epochs)',
            xlabel='recall', ylabel='precision',
        )

        coco_filenames = [fbeta_filename, pr_filename, cocopr_filename]
        coco_dfs = [avg_fbeta_df, avg_pr_df]

        return avg_stats, avg_matrix, avg_map, avg_summary, coco_filenames, coco_dfs

    def _overall_coco_path_method(
        self,
        fbeta_df_list, 
        pr_df_list,
        severities, 
        subfolder_name, 
        # aug_name,
        iou_thres = None,
        fbeta_metric = None,
    ):

        plt.rcParams.update({'font.size': 18})
        path_dict = {}
        path_dict['fbeta'] = {}
        path_dict['pr'] = {}

        save_dir0 = self._save_folder / subfolder_name
        save_dir0.mkdir(parents=True, exist_ok=True)
        save_dir = save_dir0 / "figures"
        save_dir.mkdir(parents=True, exist_ok=True)
        # print()
        # print("FBETA", fbeta_metric, "PR", iou_thres)
        # print(fbeta_df_list[0].head())
        # print()

        for col in fbeta_df_list[0].index:
            if fbeta_metric is not None and fbeta_metric != col:
                continue
            fbeta_path = save_dir / f"coco_fbeta_{col}.png"
            plotMultiplePRCurves(fbeta_df_list, col, severities, fbeta_path)
            path_dict['fbeta'][col] = str(fbeta_path.relative_to(self._output_folder))
        
        for col in pr_df_list[0].index:
            if iou_thres is not None and iou_thres != col:
                continue
            safe_col = col.replace("=", "_")
            pr_path = save_dir / f"coco_pr_{safe_col}.png"
            plotMultiplePRCurves(pr_df_list, col, severities, pr_path)
            path_dict['pr'][safe_col] = str(pr_path.relative_to(self._output_folder))

        return path_dict