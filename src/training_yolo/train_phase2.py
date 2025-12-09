from train_yolo import train_model
import os

EXPERIMENTS = [
    {"dataset_path": "1_belugas_raw_115", "model_name": "belugas_115"},
    {"dataset_path": "2_belugas_aug_275", "model_name": "belugas_275"},
]

PHASE1_DIR = "../../models/Phase 1 - Transfer Learning"
PHASE2_NAME = "Phase 2 - Fine-tuning"

for exp in EXPERIMENTS:
    dataset = exp["dataset_path"]
    model_name = exp["model_name"]

    phase1_model_dir = os.path.join(PHASE1_DIR, model_name)
    for subdir in os.listdir(phase1_model_dir):

        model_path = f"{model_name}/{subdir}"
        train_model(
            is_transfer_learning=False,
            dataset_path=dataset,
            model_name=model_name,
            phase_name=PHASE2_NAME,
            epochs=50,
            freeze_layers=0,
            lr0=0.0001,
            model_path=f"Phase 1 - Transfer Learning/{model_path}"
        )
