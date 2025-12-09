import pandas as pd
import os

MODELS_DIR = "../../models/"
OUPUTS_DIR = "../../outputs/"

def extract_best_metrics(model_paths, output_dir, output_csv='models_comparison.csv'):
    all_results = []

    for model_info in model_paths:
        model_name = model_info['name']
        results_path = os.path.join(model_info['path'], 'results.csv')


        if not os.path.exists(results_path):
            print(f"results.csv not found for {model_name}")
            continue

        df = pd.read_csv(results_path)
        df.columns = df.columns.str.strip()

        map50_col = None
        for col in df.columns:
            if 'mAP50' in col and 'mAP50-95' not in col:
                map50_col = col
                break

        if map50_col is None:
            print(f"Column mAP50not found for {model_name}")
            continue

        best_idx = df[map50_col].idxmax()
        best_row = df.loc[best_idx]


        result = {
            'Model': model_name,
            'Best_Epoch': int(best_row['epoch']) if 'epoch' in df.columns else best_idx,
        }

        metric_mappings = {
            'mAP50': ['metrics/mAP50(B)', 'mAP50'],
            'mAP50-95': ['metrics/mAP50-95(B)', 'mAP50-95'],
            'Precision': ['metrics/precision(B)', 'precision'],
            'Recall': ['metrics/recall(B)', 'recall'],
            'Box_Loss': ['train/box_loss', 'box_loss'],
            'Cls_Loss': ['train/cls_loss', 'cls_loss'],
            'DFL_Loss': ['train/dfl_loss', 'dfl_loss']
        }

        for metric_name, possible_cols in metric_mappings.items():
            for col in possible_cols:
                if col in df.columns:
                    result[metric_name] = round(best_row[col], 4)
                    break
            if metric_name not in result:
                result[metric_name] = 'N/A'

        all_results.append(result)


    df_results = pd.DataFrame(all_results)

    if "mAP50" in df_results.columns:
        df_results = df_results.sort_values('mAP50', ascending=False)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_csv)

    df_results.to_csv(output_path, index=False)

    return df_results


def search_models(base_phase_dir):
    model_paths = []

    for root, dirs, files in os.walk(base_phase_dir):
        if "results.csv" in files:
            model_name = os.path.basename(root)

            model_paths.append({
                "name": model_name,
                "path": root
            })
    return model_paths

def process_phase(phases, models_root="models", output_root="outputs"):
    for phase in phases:
        phase_dir = os.path.join(models_root, phase)
        phase_output = os.path.join(output_root, phase)

        model_paths = search_models(phase_dir)
        if len(model_paths) == 0:
            print(f"No model found in : {phase}")
            continue

        extract_best_metrics(model_paths, phase_output)

if __name__ == "__main__":
    phases = [
        'Phase 1 - Transfer Learning',
        'Phase 2 - Fine-tuning',
        'Phase 3 - Introduce a new species',
        'Phase 4 - Add more beluga images',
        'Phase 6 - Final model']

    process_phase(phases, MODELS_DIR, OUPUTS_DIR)