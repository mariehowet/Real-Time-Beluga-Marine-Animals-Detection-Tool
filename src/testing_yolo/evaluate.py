import os
import yaml
from ultralytics import YOLO
import pandas as pd
import numpy as np

MODELS_DIR = "../../models/"
OUPUTS_DIR = "../../outputs/"


def evaluate_on_test_images(best_pt_path:str, model_name: str, yaml_path: str, output_folder: str):
    model = YOLO(best_pt_path)
    output_path = os.path.join(output_folder, "evaluation",model_name)
    os.makedirs(output_path, exist_ok=True)

    metrics = model.val(data=yaml_path, split='test', project=output_path, name='metrics', conf=0.7)

    metrics_dict = {
        'model': model_name,
        'mAP50': safe_float(metrics.box.map50),
        'mAP50-95': safe_float(metrics.box.map),
        'precision': safe_float(metrics.box.p),
        'recall': safe_float(metrics.box.r),
        'f1': safe_float(2 * metrics.box.p * metrics.box.r / (metrics.box.p + metrics.box.r + 1e-6))
    }

    metrics_df = pd.DataFrame([metrics_dict])
    metrics_csv = os.path.join(output_path, 'test_metrics.csv')
    metrics_df.to_csv(metrics_csv, index=False)

    with open(os.path.join(output_path, 'test_metrics.txt'), 'w') as f:
        f.write(f"Test Set Evaluation - {model_name}\n")
        f.write("=" * 50 + "\n\n")
        for key, value in metrics_dict.items():
            if key != 'model':
                f.write(f"{key:15s}: {value:.4f}\n")

    return metrics_dict

def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(np.mean(value)) if hasattr(value, '__iter__') else 0.0

def load_best_pt(phase_dir, model_name):
    for root, dirs, files in os.walk(phase_dir):
        if model_name in dirs:
            weights_path = os.path.join(root, model_name, "weights", "best.pt")
            if os.path.exists(weights_path):
                return weights_path
            else:
                print(f"⚠ No best.pt found in {weights_path}")
                return None
    print(f"Model directory '{model_name}' not found in {phase_dir}")
    return None


def get_yaml_path_from_args(phase_dir, model_name):
    for root, dirs, files in os.walk(phase_dir):
        if model_name in dirs:
            args_file = os.path.join(root, model_name, "args.yaml")
            if not os.path.exists(args_file):
                print(f"args.yaml not found in {args_file}")
                return None

            with open(args_file, "r") as f:
                args = yaml.safe_load(f)

            data_path = args.get("data")
            if data_path:
                return os.path.join("..", "..", data_path)
            else:
                print(f"'data' field not found in {args_file}")
                return None

    return None

def get_best_model(output_dir):
    csv_path = os.path.join(output_dir, "models_comparison.csv")
    if not os.path.exists(csv_path):
        print(f"No models_comparison.csv found in {output_dir}")
        return None

    df = pd.read_csv(csv_path)
    if 'Model' not in df.columns:
        print(f"'Model' column not found in {csv_path}")
        return None

    return df["Model"].iloc[0]


def process_phases(phases, models_root="models", output_root="outputs"):
    for phase in phases:
        phase_dir = os.path.join(models_root, phase)
        phase_output = os.path.join(output_root, phase)

        best_model = get_best_model(phase_output)
        if not best_model:
            print(f"No model found for phase {phase}")
            continue
        print(f"Evaluating best model '{best_model}' for phase '{phase}'")

        best_pt_path = load_best_pt(phase_dir, best_model)
        if best_pt_path is None:
            continue

        yaml_path = get_yaml_path_from_args(phase_dir, best_model)
        if yaml_path is None:
            continue

        try:
            metrics_dict = evaluate_on_test_images(
                best_pt_path,
                best_model,
                yaml_path,
                phase_output
            )
        except Exception as e:
            print(f"Error evaluating {best_model}: {str(e)}")
            continue

if __name__ == "__main__":
    phases = [
        'Phase 1 - Transfer Learning',
        'Phase 2 - Fine-tuning',
        'Phase 3 - Introduce a new species',
        'Phase 4 - Add more beluga images',
        'Phase 6 - Final model']

    process_phases(phases, MODELS_DIR, OUPUTS_DIR)