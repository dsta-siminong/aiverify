import importlib, pathlib
import aiverify_test_engine
base = pathlib.Path(aiverify_test_engine.__file__).parent / "io"

for name in ["lightgbmmodel", "xgboostmodel", "sklearnmodel", "tensorflowmodel", "pytorchmodel"]:
    mod = importlib.import_module(f"aiverify_test_engine.io.{name}.{name}")
    try:
        instance = mod.Plugin(model={"api_type": "image_classifier_v1", "api_url": "http://x"})
        result = instance.is_supported()
        print(name, "-> is_supported():", result)
    except Exception as e:
        print(name, "-> RAISED:", type(e).__name__, e)

mod = importlib.import_module("aiverify_test_engine.io.apimodel.apimodel")
instance = mod.Plugin(model={"api_type": "image_classifier_v1", "api_url": "http://localhost:8000/predict_array"})
print("apimodel -> is_supported():", instance.is_supported())
