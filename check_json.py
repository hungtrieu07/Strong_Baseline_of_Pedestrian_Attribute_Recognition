import json

with open("openvino_models/reid/reid_model.json") as f:
    try:
        json.load(f)
        print("JSON is valid")
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
