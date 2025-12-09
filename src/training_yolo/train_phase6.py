
from train_yolo import train_model
import os


EXPERIMENTS = [
 {"dataset_path": "6_marine_animals_1050", "model_name": "marine_animals_1050"},
 {"dataset_path": "7_marine_animals_2526", "model_name": "marine_animals_2526"},
]
PHASE = "Phase 6 - Final model"
PHASE_DIR = F"../../models/{PHASE}/"

def transfer_learn(dataset_path, model_name):
    train_model(
        is_transfer_learning=True,
        dataset_path= dataset_path,
        model_name=model_name,
        phase_name=PHASE,
        epochs=50,
        freeze_layers=11,
        lr0=0.01
    )


def fine_tune(dataset_path, model_name, model_path):
    train_model(
        is_transfer_learning=False,
        dataset_path=dataset_path,
        model_name=model_name,
        phase_name=PHASE,
        epochs=100,
        freeze_layers=0,
        lr0=0.0001,
        model_path= model_path,
        optimizer="SGD",
        momentum=0.9
    )

if __name__ == "__main__":
    for exp in EXPERIMENTS:
        dataset_path = exp["dataset_path"]
        model_name = exp["model_name"]
        model_path = f"{PHASE}/{model_name}/transfer_learn_{model_name}_epoch50_freeze11"

        transfer_learn(dataset_path, model_name)
        fine_tune(dataset_path, model_name, model_path)