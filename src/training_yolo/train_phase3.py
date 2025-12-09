from train_yolo import train_model
import os


EXPERIMENTS = [
    {"dataset_path": "3_belugas_and_dolphins_310", "model_name": "belugas_and_dolphins_310"},
    {"dataset_path": "4_belugas_and_dolphins_744", "model_name": "belugas_and_dolphins_744"},
]

PHASE3_NAME = "Phase 3 - Introduce a new species"
PHASE3_TL = f"{PHASE3_NAME}/1 - Transfer Learning"
PHASE3_TL_DIR = F"../../models/{PHASE3_NAME}/"

PHASE3_FT = f"{PHASE3_NAME}/2 - Fine-tuning"
PHASE3_FT_DIR = F"../../models/{PHASE3_TL}/"

def transfer_learn():
    freezes = [0, 11]

    for exp in EXPERIMENTS:
        for freeze in freezes:
                train_model(
                    is_transfer_learning=True,
                    dataset_path= exp["dataset_path"],
                    model_name=exp["model_name"],
                    phase_name=PHASE3_TL,
                    epochs=50,
                    freeze_layers=freeze,
                    lr0=0.01
                )

def fine_tune():
    for exp in EXPERIMENTS:
        dataset = exp["dataset_path"]
        model_name = exp["model_name"]

        phase3_model_dir = os.path.join(PHASE3_FT_DIR, model_name)
        for subdir in os.listdir(phase3_model_dir):
            model_path = f"{model_name}/{subdir}"

            train_model(
                is_transfer_learning=False,
                dataset_path=dataset,
                model_name=model_name,
                phase_name=PHASE3_FT,
                epochs=100,
                freeze_layers=0,
                lr0=0.0001,
                model_path=f"{PHASE3_TL}/{model_path}"
            )

if __name__ == "__main__":
    transfer_learn()
    fine_tune()