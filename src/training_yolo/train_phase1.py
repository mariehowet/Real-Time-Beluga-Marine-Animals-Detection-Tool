
from train_yolo import train_model

EXPERIMENTS = [
    {"dataset_path": "1_belugas_raw_115", "model_name": "belugas_115"},
    {"dataset_path": "2_belugas_aug_275", "model_name": "belugas_275"},
]

FREEZE = [0, 11]
EPOCHS = [10, 50, 100]
PHASE = "Phase 1 - Transfer Learning"
LEARNING_RATE = 0.01

for exp in EXPERIMENTS:
    for fr in FREEZE:
        for ep in EPOCHS:
            train_model(
                is_transfer_learning=True,
                dataset_path= exp["dataset_path"],
                model_name=exp["model_name"],
                phase_name=PHASE,
                epochs=ep,
                freeze_layers=fr,
                lr0=LEARNING_RATE
            )
