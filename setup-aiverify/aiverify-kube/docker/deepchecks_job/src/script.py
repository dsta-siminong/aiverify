import numpy as np 
import pandas as pandas
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import FeatureDrift, MultivariateDrift, LabelDrift
import pickle
import json
import os
from redis_tools.redis_stream import RedisStream

stream_listener = RedisStream()
is_success, error_message = stream_listener.setup(
    "aiverify-redis-service.aiverify.svc.cluster.local",
    6379,
    "TestEngineTask",
    "MyGroup",
    "Job",
)
redis_instance = stream_listener.get_redis_instance()
            
test_name = os.environ.get('test_name', "Tabular Multivariate Drift")
user_defined_label_col = os.environ.get('label', "default")
data_path = os.environ.get('background_path', "sample_bc_credit_data.sav")
drifted_data_path = os.environ.get('drifted_path', "sample_bc_credit_data_drifted.sav")
job_id = os.environ.get("job_id", "redis_job")

print(os.getcwd())

if test_name == "Tabular Feature Drift":

    background_df = pickle.load(open(data_path, 'rb'))
    drifted_df = pickle.load(open(drifted_data_path, 'rb'))

    if user_defined_label_col != "NA":
        if user_defined_label_col in drifted_df.columns:
            drifted_df = drifted_df.drop(user_defined_label_col, axis=1) 
        if user_defined_label_col in background_df.columns:
            background_df = background_df.drop(user_defined_label_col, axis=1)
    
    background_dataset = Dataset(background_df)
    drifted_dataset = Dataset(drifted_df)

    check = FeatureDrift()
    result = check.run(train_dataset=background_dataset, test_dataset=drifted_dataset)

    # Get the values of the feature name and convert to a list.
    results = {
        "feature_drift": result.value
    }
    
elif test_name == "Tabular Multivariate Drift":

    background_df = pickle.load(open(data_path, 'rb'))
    drifted_df = pickle.load(open(drifted_data_path, 'rb'))
    
    if user_defined_label_col != "NA":
        if user_defined_label_col in drifted_df.columns:
            drifted_df = drifted_df.drop(user_defined_label_col, axis=1) 
        if user_defined_label_col in background_df.columns:
            background_df = background_df.drop(user_defined_label_col, axis=1)
    
    background_dataset = Dataset(background_df)
    drifted_dataset = Dataset(drifted_df)
        
    check = MultivariateDrift()
    result = check.run(train_dataset=background_dataset, test_dataset=drifted_dataset)

    results = {
        "multivariate_drift": result.value
    }
    
elif test_name == "Tabular Label Drift":

    background_df = pickle.load(open(data_path, 'rb'))
    drifted_df = pickle.load(open(drifted_data_path, 'rb'))
    
    background_dataset = Dataset(background_df, label=user_defined_label_col)
    drifted_dataset = Dataset(drifted_df, label=user_defined_label_col)

    check = LabelDrift()
    result = check.run(train_dataset=background_dataset, test_dataset=drifted_dataset)

    # Get the values of the feature name and convert to a list.
    results = {
        "label_drift": result.value
    }

redis_dict = {}
redis_dict.update(
    {
        "output": json.dumps(results)
    }
            )

print(redis_dict)

for key, value in redis_dict.items():
    redis_instance.hset(job_id, key, value)   
redis_output = redis_instance.hgetall(job_id)
print(redis_output['output'])
#print(type(redis_output['output']))
