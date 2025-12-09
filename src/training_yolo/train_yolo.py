from ultralytics import YOLO
from typing import Optional

def train_model(is_transfer_learning:bool, dataset_path: str, model_name :str, phase_name :str, epochs: int, freeze_layers: int, lr0:float,  model_path: Optional[str] = None,optimizer: Optional[str] = None, momentum: Optional[float] = None):
    yaml_path = f"../../data/{dataset_path}/data.yaml"

    model_full_path = "../yolo11n.pt" if is_transfer_learning else f"../../models/{model_path}/weights/best.pt"
    model = YOLO(model_full_path)

    output_directory = f"../../models/{phase_name}/{model_name}"
    prefix = "" if is_transfer_learning else "finetune_"
    output_name = f"{prefix}transfer_learn_{model_name}_epoch{epochs}_freeze{freeze_layers}"

    train_params = {
        "data":  yaml_path,
        "epochs": epochs,
        "freeze": freeze_layers,
        "lr0" : lr0,
        "project" : output_directory,
        "name": output_name,
        "imgsz" :  640,
        "batch" : 16,
        "pretrained":  True,
        "device" : "cuda:0",
    }


    if optimizer:
        train_params["optimizer"] = optimizer
    if momentum:
        train_params["momentum"] = momentum


    results = model.train(**train_params)

    metrics = results.results_dict
    print(metrics)

