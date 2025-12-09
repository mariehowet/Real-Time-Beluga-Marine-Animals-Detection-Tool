
from train_yolo import train_model

exp = {"dataset_path": "5_belugas_aug_652", "model_name": "belugas_652"}

PHASE = "Phase 4 - Add more beluga images"
PHASE_DIR = F"../../models/{PHASE}/"

def transfer_learn():
    train_model(
        is_transfer_learning=True,
        dataset_path= exp["dataset_path"],
        model_name=exp["model_name"],
        phase_name=PHASE,
        epochs=50,
        freeze_layers=11,
        lr0=0.01
    )


def fine_tune():
    dataset = exp["dataset_path"]
    model_name = exp["model_name"]

    model_path =  f"{PHASE}/{model_name}/transfer_learn_belugas_652_epoch50_freeze11"

    train_model(
        is_transfer_learning=False,
        dataset_path=dataset,
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
    transfer_learn()
    fine_tune()