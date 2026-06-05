import pickle
import json
import jsonschema
from jsonschema import validate
from jsonschema.exceptions import ValidationError
from aiverify_test_engine.utils.json_utils import load_schema_file, validate_json

# load your data
with open("data.pkl", "rb") as f:
    results = pickle.load(f)

# load schema
schema = load_schema_file("output.schema.json")


def debug_validate(data, schema):
    try:
        validate(instance=data, schema=schema)
        print("✅ VALID")
        return True

    except ValidationError as e:
        print("\n❌ VALIDATION FAILED")
        print("\n--- ERROR MESSAGE ---")
        print(e.message)

        print("\n--- PATH (where it failed) ---")
        print(list(e.path))

        print("\n--- FAILED VALUE ---")
        failed = data
        for p in e.path:
            failed = failed[p]
        print(json.dumps(failed, indent=2, default=str))

        print("\n--- EXPECTED SCHEMA NODE ---")
        print(e.schema)

        return False


debug_validate(results, schema)
