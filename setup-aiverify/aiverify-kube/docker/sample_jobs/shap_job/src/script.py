import logging
import numpy as np 
import pandas as pandas
import pickle
import json
import os
from redis_tools.redis_stream import RedisStream

from pathlib import Path, PurePath
from typing import Any, Dict, List, Tuple, Union
import shap
from utils.explain_types import ExplainType

stream_listener = RedisStream()
is_success, error_message = stream_listener.setup(
    "aiverify-redis-service.aiverify.svc.cluster.local",
    6379,
    "TestEngineTask",
    "MyGroup",
    "Job",
)
redis_instance = stream_listener.get_redis_instance()

test_path = os.environ.get('test_path', "sample_bc_credit_data.sav")         
ground_truth_path = os.environ.get('ground_truth_path', "sample_bc_credit_data.sav")
label_column = os.environ.get('label_column', "default") 
model_path = os.environ.get('model_path', "sample_bc_credit_sklearn_linear.LogisticRegression.sav")
explain_type = os.environ.get('input_explain_type', "global")
input_background_samples = os.environ.get('input_background_samples', str(25))
input_data_samples = os.environ.get('input_data_samples', str(25))
input_background_path = os.environ.get('input_background_path', "sample_bc_credit_data.sav")   
job_id = os.environ.get("job_id", "shap_job")
#http_url = os.environ.get("http_url", "http_url")
sample_seed = 1

print(os.getcwd())
print(os.path.exists("/app/aiverify/uploads/data"))
print(os.path.exists("/app/aiverify/uploads/model"))
print(os.path.exists("/app/aiverify/uploads/data/sample_mc_toxic_data.sav"))
print(job_id)
print(explain_type)

test_df = pickle.load(open(test_path, 'rb'))
ground_truth_df = pickle.load(open(ground_truth_path, 'rb'))
background_df = pickle.load(open(input_background_path, 'rb'))
model = pickle.load(open(model_path, 'rb'))
input_data_samples = int(input_data_samples)
input_background_samples = int(input_background_samples)


def _perform_input_validation(
        input_background_path: str,
        input_background_samples: int,
        input_data_samples: int,
        input_explain_type: str,
    ) -> Tuple[int, str]:
        """
        A helper method to perform input validation

        Args:
            input_background_path (str): Path to background data
            input_background_samples (str): No. of background samples
            input_data_samples (str): No. of data samples
            input_explain_type (str): String of explain type
        Returns:
            Tuple[int, str]: Returns a tuple consisting of total error count and error messages
        """
        error_count = 0
        error_message = ""

        supported_explain_types = [
            explain_type.name.lower() for explain_type in ExplainType
        ]

        if input_background_path == "" or not _is_file(input_background_path):
            error_count += 1
            error_message += "The background path is invalid;"

        if input_background_samples < 0 or not isinstance(
            input_background_samples, int
        ):
            error_count += 1
            error_message += "The background samples are invalid;"

        if input_data_samples < 0 or not isinstance(input_data_samples, int):
            error_count += 1
            error_message += "The samples are invalid;"

        if (
            input_explain_type == ""
            or input_explain_type.lower() not in supported_explain_types
        ):
            error_count += 1
            error_message += "The explain type is invalid;"

        return error_count, error_message
        
        
def _explain_shap():
    """
    A helper method to perform shap explanation
    """    
    global background_df, test_df, ground_truth_df, model
    
    error_count, error_message = _perform_input_validation(
        input_background_path,
        input_background_samples,
        input_data_samples,
        explain_type,
    )
    # if error_count > 0:
    #     self.add_to_log(
    #         logging.ERROR,
    #         f"The algorithm has failed input arguments validation: {error_message}",
    #     )
    #     raise RuntimeError(
    #         f"The algorithm has failed input arguments validation: {error_message}"
    #     )
    # else:
    #     self.add_to_log(
    #         logging.INFO,
    #         f"Validated plugin input arguments - "
    #         f"Background Path: {input_background_path},"
    #         f"Background Samples: {input_background_samples},"
    #         f"Data Samples: {input_data_samples},"
    #         f"Explain Type: {input_explain_type}",
    #     )

    # Set the Explain Type
    if explain_type.lower() == ExplainType.GLOBAL.name.lower():
        input_explain_type = ExplainType.GLOBAL
    else:
        input_explain_type = ExplainType.LOCAL

    # Retrieve data information
    # Check if background dataset has ground truth first
    if label_column in background_df.columns:
        background_df = background_df.drop(
            label_column, axis=1
        )
        test_df = test_df.drop(
            label_column, axis=1
        )
    else:
        background_df = background_df

    # Perform data_sampling and background_sampling
    if input_data_samples > 0:
        num_of_samples = min(len(test_df), input_data_samples)
        test_df = test_df.sample(
            num_of_samples, random_state=sample_seed
        )

        ground_truth_df = ground_truth_df.sample(
            num_of_samples, random_state=sample_seed
        )

    if input_background_samples > 0:
        num_of_samples = min(len(background_df), input_background_samples)
        background_df = background_df.sample(
            num_of_samples, random_state=sample_seed
        )
    # Get explainer function
    explainer = _get_explainer(background_df)

    single_shap_value = explainer.shap_values(
        test_df.sample(1, random_state=sample_seed)
    )

    # Global or Local?
    if input_explain_type is ExplainType.GLOBAL:
        # if (
        #     self._model_instance.get_plugin_type() is PluginType.MODEL
        #     and self._model_instance.get_model_plugin_type()
        #     is ModelPluginType.XGBOOST
        # ):
        #     data = xgboost.DMatrix(self._data, self._ground_truth_data)
        #     shap_values = explainer.shap_values(data)
        # else:
        shap_values = explainer.shap_values(test_df)
        results = {
            "shap_values": shap_values,
            "samples": test_df,
            "single_shap_value": single_shap_value,
            "explainer": explainer,
        }
        
    else:
        # Local Explain Type
        results = {
            "single_shap_value": single_shap_value,
            "explainer": explainer,
        }

    # Format the output results
    output_results = format_result(input_explain_type, results)

    # Assign the output results
    return output_results

def _get_explainer_predict_helper(data: Any) -> Any:
    """
    A self-defined function for explainer to perform prediction proba

    Arg:
        data (Any): The data to be sent for prediction proba

    Returns:
        Any: predicted value
    """
    dict_labels = dict()
    for key in test_df.keys():
        dict_labels[key] = test_df.dtypes[key].name
    dict_item_labels = dict_labels.items()
    predicted_results = model.predict(
        [item for item in data.tolist()]
    )
    if isinstance(predicted_results, list):
        predicted_results = np.array(list(predicted_results), dtype="float32")
    return predicted_results

def _get_explainer(background_df) -> Union[shap.TreeExplainer, shap.KernelExplainer]:
    """
    A function to return upload explainer. Upload method will return explainer based on algorithm configuration

    Returns:
        Union[shap.TreeExplainer, shap.KernelExplainer]: An instance of either
        TreeExplainer, or KernelExplainer for upload
    """
    # if (
    #     self._model_instance.get_plugin_type() is PluginType.MODEL
    #     and self._model_instance.get_model_plugin_type() is ModelPluginType.XGBOOST
    # ):
    #     # Tree Explainer
    #     self.add_to_log(logging.DEBUG, "Using TreeExplainer")
    #     explainer = shap.TreeExplainer(self._model_instance.get_model())
    # else:
        # Others (KernelExplainer)
        # self.add_to_log(logging.DEBUG, "Using KernelExplainer")
        # if (
        #     self._model_instance.get_plugin_type() is PluginType.MODEL
        #     and self._model_instance.get_model_plugin_type()
        #     is ModelPluginType.TENSORFLOW
        # ):
        #     explainer = shap.KernelExplainer(
        #         self._model_instance.get_model(), self._background
        #     )
        # else:
    explainer = shap.KernelExplainer(
        _get_explainer_predict_helper, background_df
    )

    return explainer

def _is_file(argument: str) -> bool:
    """
    A helper function to check if argument is a file

    Args:
        argument (str): path to file

    Returns:
        bool: True if argument is a file
    """
    return Path(argument).is_file()

def format_result(input_explain_type, results: dict) -> dict:
    """
    A helper method to format the results to match output schema

    Args:
        results (dict): Results to be formatted in dict

    Returns:
        dict: Formatted results in dict
    """
    output_dict = dict({"feature_names": list(), "results": dict()})

    # Populate results dictionary
    output_results = dict(
        {
            "num_local_classes": 0,
            "local": list(),
            "single_explainer_values": [],
            "single_shap_values": list(),
            "global_shap_values": list(),
            "global_samples": list(),
            "global": list(),
            "num_global_classes": 0,
        }
    )

    # Local Explainability
    # Set the local shap value
    # Populate single shap value in class0 and class1 and so on.
    local_shap_values = list()
    for count in range(len(results["single_shap_value"])):
        tmp_value = results["single_shap_value"][count]

        # Convert the tmp_value into an array of dimension.
        # If it is already a ndarray of dimension 2, it will remain same
        single_shap_value = np.array(tmp_value, ndmin=2)
        local_shap_values.append(single_shap_value.tolist())
    output_results.update({"local": local_shap_values})
    output_results.update({"num_local_classes": len(results["single_shap_value"])})

    # Set single explainer values and single shap values
    if isinstance(results["explainer"], list):
        temp_value_ndarray = np.array(results["explainer"][0], ndmin=1)
    else:
        temp_value_ndarray = np.array(results["explainer"].expected_value, ndmin=1)
    output_results.update({"single_explainer_values": temp_value_ndarray.tolist()})

    single_shap_values_list = list()
    for temp_value in results["single_shap_value"]:
        # Convert the tmp_value into an array of dimension 2.
        # If it is already a ndarray of dimension 2, it will remain same
        temp_value_ndarray = np.array(temp_value, ndmin=2)
        # Remove one dimension from the double array
        single_shap_values_list.append(temp_value_ndarray.tolist()[0])
    output_results.update({"single_shap_values": single_shap_values_list})

    # Global bar and force plot
    if input_explain_type is ExplainType.GLOBAL:
        temp_shap_value_ndarray = np.array(results["shap_values"], ndmin=3)
        temp_samples_value_ndarray = np.array(results["samples"].values, ndmin=2)
        output_results.update(
            {
                "global_shap_values": temp_shap_value_ndarray.tolist(),
                "global_samples": temp_samples_value_ndarray.tolist(),
            }
        )

        # Calculate average global shap values
        global_avg_shap_values = list()
        global_shap_value_ndarray = np.array(results["shap_values"], ndmin=3)
        num_of_classes = global_shap_value_ndarray.shape[0]
        num_of_features = global_shap_value_ndarray.shape[2]

        for class_count in range(num_of_classes):
            features_average = list()
            for features_count in range(num_of_features):
                temp_data_slice = global_shap_value_ndarray[
                    class_count, :, features_count
                ]
                features_average.append(np.abs(temp_data_slice).mean(0))
            global_avg_shap_values.append(features_average)
        output_results.update({"global": global_avg_shap_values})
        output_results.update({"num_global_classes": num_of_classes})

    # Populate feature names
    output_dict["feature_names"] = test_df.columns.tolist()
    output_dict["results"] = output_results
    
    return output_dict


# check = FeatureDrift()
# result = check.run(train_dataset=background_dataset, test_dataset=drifted_dataset)

# Get the values of the feature name and convert to a list.

results = _explain_shap() #dict
print(type(results))
print(results)

for key, value in results.items():
    print(key, type(key))
    print(value, type(value))
    redis_instance.hset(job_id, key, json.dumps(value))   
redis_output = redis_instance.hgetall(job_id)
print(redis_output['feature_names'])
print(type(redis_output['feature_names']))
print(redis_output['results'])
print(type(redis_output['results']))
