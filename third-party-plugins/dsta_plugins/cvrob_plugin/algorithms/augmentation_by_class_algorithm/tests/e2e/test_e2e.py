from pathlib import Path

import pytest
from augmentation_by_class_algorithm.algo_init import AlgoInit
from aiverify_test_engine.plugins.enums.model_type import ModelType


binary_classification_pipeline = {
    "data_path": str(
        "/home/bjieyong/aiverify/cvrob/dataset_20200803/all_images_100"
    ),
    "model_path": str(
        "/home/bjieyong/aiverify/cvrob/ship_pipe/ship_model.pt"
    ),
    "ground_truth_path": str(
        "/home/bjieyong/aiverify/cvrob/dataset_20200803/labels_100.csv"
    ),
    "run_pipeline": False,
    "model_type": ModelType.CLASSIFICATION,
    "ground_truth": "label",
    "plugin_argument_values": {
        "class_names": "13",#Barge,CG-P,ContainerShip,Cruise,Dredger,Ferry,LNG-LPG,RORO,Sampan,Trawler-FishingVessel,Tugboat,Warship,Yacht",
        "aug_library": 'albumentations',
        "aug_methods": 'GaussianBlur,GaussianNoise',
        "num_epochs": 2,
        "custom_parameters": "GaussianBlur sigma_limit 1.6,2.6,3.6,4.6 GaussianNoise std_range (0.06,0.06),(0.12,0.12),(0.18,0.18),(0.24,0.24)"
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
