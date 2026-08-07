from pathlib import Path

import pytest
from augmentation_by_class_algorithm.algo_init import AlgoInit
from aiverify_test_engine.plugins.enums.model_type import ModelType


binary_classification_pipeline = {
    "data_path": str(
        "../../../../../../all_images_all_classes"
    ),
    "model_path": str(
        "../../../../../../ship_pipe_sm/ship_pipe_sm"#"../../../../../../api.json"#
    ),
    "ground_truth_path": str(
        "../../../../../../labels_all_classes.csv"
    ),
    "run_pipeline": True,
    "model_type": ModelType.CLASSIFICATION,
    "ground_truth": "label",
    "plugin_argument_values": {
        "class_names": "Barge,CG-P,ContainerShip,Cruise,Dredger,Ferry,LNG-LPG,RORO,Sampan,Trawler-FishingVessel,Tugboat,Warship,Yacht",
        "aug_library": 'http://localhost:8100',
        'aug_methods': 'Rain',
        'custom_parameters': None,
        "num_epochs": 1,
        # "model_api_url": "http://localhost:8000/predict_array",
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
