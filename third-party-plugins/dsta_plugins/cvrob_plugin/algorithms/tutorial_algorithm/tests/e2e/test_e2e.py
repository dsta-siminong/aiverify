from pathlib import Path

import pytest
from tutorial_algorithm.algo_init import AlgoInit
from aiverify_test_engine.plugins.enums.model_type import ModelType


binary_classification_pipeline = {
    "data_path": str(
        "https://github.com/aiverify-foundation/aiverify/raw/refs/heads/main/stock-plugins/user_defined_files/data/sample_bc_credit_data.sav"
    ),
    "model_path": str("https://github.com/aiverify-foundation/aiverify/raw/refs/heads/main/stock-plugins/user_defined_files/model/sample_bc_credit_sklearn_linear.LogisticRegression.sav"),
    "ground_truth_path": str(
        "https://github.com/aiverify-foundation/aiverify/raw/refs/heads/main/stock-plugins/user_defined_files/data/sample_bc_credit_data.sav"
    ),
    "run_pipeline": False,
    "model_type": ModelType.CLASSIFICATION,
    "ground_truth": "default",
    "plugin_argument_values": {
        "feature_name": "gender",
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
        data_set["run_pipeline"],
        core_modules_path,
        data_set["data_path"],
        data_set["model_path"],
        data_set["ground_truth_path"],
        data_set["ground_truth"],
        data_set["model_type"],
        data_set["plugin_argument_values"]
   )
    plugin_test.run()

    json_file_path = Path.cwd() / "output" / "results.json"
    assert json_file_path.exists()
