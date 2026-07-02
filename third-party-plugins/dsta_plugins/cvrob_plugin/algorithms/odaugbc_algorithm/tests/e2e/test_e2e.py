from pathlib import Path

import pytest
from odaugbc_algorithm.algo_init import AlgoInit
from aiverify_test_engine.plugins.enums.model_type import ModelType

binary_classification_pipeline = {
    "data_path": str(
        "/home/bjieyong/aiverify/cvrob/bccd/BCCD/JPEGImages"
    ),
    "model_path": str(
        "/home/bjieyong/aiverify/cvrob/bccd/BCCD/bccdModel"
    ),
    "ground_truth_path": str(
        "/home/bjieyong/aiverify/cvrob/bccd/BCCD/bccd_detection.csv"
    ),
    "run_pipeline": True,
    "model_type": ModelType.CLASSIFICATION,
    "ground_truth": "label",
    "plugin_argument_values": {
        "class_names": None, 
        "aug_library": 'albumentations',
        'aug_methods': 'Rain',
        'custom_parameters': None,
        "num_epochs": 1,
        "iou_thres": 0.65,
        "score_thres": 0.65,
    }
}

@pytest.mark.parametrize(
    "data_set",
    [
        
        binary_classification_pipeline,
        
    ],
)
def test_plugin(data_set):
    # Create an instance of PluginTest with defined paths and arguments and Run.
    core_modules_path = ""
    plugin_test = AlgoInit(
        run_as_pipeline=data_set["run_pipeline"],
        core_modules_path=core_modules_path,
        data_path=data_set["data_path"],
        model_path=data_set["model_path"],
        ground_truth_path=data_set["ground_truth_path"],
        ground_truth=data_set["ground_truth"],
        model_type=data_set["model_type"],
        **data_set["plugin_argument_values"]
    )
    plugin_test.run()

    json_file_path = Path.cwd() / "output" / "results.json"
    assert json_file_path.exists()
