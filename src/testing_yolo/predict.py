import os
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import yaml
import random
from tqdm import tqdm
from ultralytics import YOLO

DATA_DIR = "../../data/"
VIDEO_DIR = "../videos/yolo/"
MODELS_DIR = "../../models/"
OUTPUT_DIR = "../../outputs/"


def predict_on_image(model_path, images_files, model_name, output_folder, conf_threshold=0.5,  iou=0.4, image_size=640):
    model = YOLO(model_path)
    output_path = os.path.join(output_folder, "prediction", model_name, "images")
    os.makedirs(output_path, exist_ok=True)
    results = model(images_files, conf=conf_threshold, iou=iou, device="cuda:0", imgsz=image_size)
    for idx, result in enumerate(results):
        boxes = result.boxes

        print(f"Image {idx + 1} : {len(boxes)} animals detected.")

        result_filename = os.path.join(output_path, f"result_{idx + 1}.png")
        result.save(result_filename)
        print(f"Result saved to {result_filename}")

def predict_on_video(model_path, video_path,  model_name, output_folder, conf_threshold=0.5):
    model = YOLO(model_path)
    video_name = Path(video_path).stem

    cap = cv2.VideoCapture(video_path)

    fps_input = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_dir = os.path.join(output_folder, "prediction", model_name, "videos")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"yolo_{video_name}_annotated.avi")
    fourcc = cv2.VideoWriter.fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(output_path, fourcc, fps_input, (frame_width, frame_height))

    inference_times = []
    confidences = []
    detections_per_frame = []
    fps_list = []
    bbox_areas = []
    class_counts = {}
    frame_count = 0

    pbar = tqdm(total=total_frames, desc=f"Processing {model_name}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        results = model(frame, conf=conf_threshold, verbose=False)
        inference_time = (time.time() - start_time) * 1000  # en ms

        inference_times.append(inference_time)
        current_fps = 1000 / inference_time if inference_time > 0 else 0
        fps_list.append(current_fps)

        boxes = results[0].boxes
        num_detections = len(boxes)
        detections_per_frame.append(num_detections)

        if num_detections > 0:
            confs = boxes.conf.cpu().numpy()
            confidences.extend(confs)

            classes = boxes.cls.cpu().numpy()
            for cls in classes:
                cls_name = model.names[int(cls)]
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            xyxy = boxes.xyxy.cpu().numpy()
            for box in xyxy:
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)
                bbox_areas.append(area)

        annotated_frame = results[0].plot()

        y_offset = 30
        metrics_text = [
            f'FPS: {current_fps:.1f}',
            f'Inference: {inference_time:.1f}ms',
            f'Detections: {num_detections}',
            f'Frame: {frame_count}/{total_frames}'
        ]

        for i, text in enumerate(metrics_text):
            cv2.putText(annotated_frame, text,
                        (10, y_offset + i * 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

        out.write(annotated_frame)
        frame_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    stats = {
        'model': model_name,
        'video': video_name,
        'total_frames': frame_count,
        'video_duration_sec': frame_count / fps_input,

        'avg_inference_ms': round(np.mean(inference_times), 2),
        'std_inference_ms': round(np.std(inference_times), 2),
        'min_inference_ms': round(np.min(inference_times), 2),
        'max_inference_ms': round(np.max(inference_times), 2),
        'avg_fps': round(np.mean(fps_list), 2),
        'min_fps': round(np.min(fps_list), 2),
        'max_fps': round(np.max(fps_list), 2),

        'total_detections': sum(detections_per_frame),
        'avg_detections_per_frame': round(np.mean(detections_per_frame), 2),
        'std_detections_per_frame': round(np.std(detections_per_frame), 2),
        'max_detections_per_frame': int(np.max(detections_per_frame)),
        'frames_with_detections': sum(1 for d in detections_per_frame if d > 0),
        'detection_rate_%': round(100 * sum(1 for d in detections_per_frame if d > 0) / frame_count, 2),

        'avg_confidence': round(np.mean(confidences), 4) if confidences else 0,
        'std_confidence': round(np.std(confidences), 4) if confidences else 0,
        'min_confidence': round(np.min(confidences), 4) if confidences else 0,
        'max_confidence': round(np.max(confidences), 4) if confidences else 0,

        'avg_bbox_area_px': round(np.mean(bbox_areas), 2) if bbox_areas else 0,
        'std_bbox_area_px': round(np.std(bbox_areas), 2) if bbox_areas else 0,

        'unique_classes': len(class_counts),
        'class_distribution': class_counts,

        'output_video': str(output_path)
    }

    return stats

def get_video_paths(video_dir):
    return [os.path.join(root, f)
            for root, _, files in os.walk(video_dir)
            for f in files if f.lower().endswith(('.mp4', '.avi', '.mov'))]

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

def get_yaml_path_from_args(phase_dir, model_name):
    for root, dirs, _ in os.walk(phase_dir):
        if model_name in dirs:
            args_file = os.path.join(root, model_name, "args.yaml")
            if not os.path.exists(args_file):
                print(f"args.yaml not found in {args_file}")
                return None
            with open(args_file, "r") as f:
                args = yaml.safe_load(f)
            return args.get("data")
    return None

def get_n_test_images(phase_dir, model_name, n=2):
    data_dir = get_yaml_path_from_args(phase_dir, model_name)
    folder_name = os.path.basename(os.path.dirname(data_dir))
    test_dir = os.path.join(DATA_DIR, folder_name, "test", "images")

    all_images = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                all_images.append(os.path.join(root, file))
    if len(all_images) == 0:
        print(f"Aucune image trouvée dans {test_dir}")
        return []

    return random.sample(all_images, min(n, len(all_images)))


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

def process_phases(video_paths, phases, models_root="models", output_root="outputs"):
    for phase in phases:
        phase_dir = os.path.join(models_root, phase)
        phase_output = os.path.join(output_root, phase)

        best_model = get_best_model(phase_output)
        if not best_model:
            print(f"No model found for phase {phase}")
            continue

        all_results = []

        best_pt_path = load_best_pt(phase_dir, best_model)
        if best_pt_path is None:
            continue

        images_paths = get_n_test_images(phase_dir, model_name, 6)
        predict_on_image(best_pt_path, images_paths, model_name, phase_output)

        for video_path in video_paths:
           stats = predict_on_video(best_pt_path, video_path, model_name, phase_output)
           all_results.append(stats)

        df_results = pd.DataFrame(all_results)
        output_dir = os.path.join(phase_output, "prediction", model_name, "videos")
        os.makedirs(output_dir, exist_ok=True)
        output_csv = os.path.join(output_dir, 'videos_yolo_results_detailed.csv')
        df_results.to_csv(output_csv, index=False)


if __name__ == "__main__":
    video_paths = get_video_paths(VIDEO_DIR)

    phases = [
        'Phase 1 - Transfer Learning',
        'Phase 2 - Fine-tuning',
        'Phase 3 - Introduce a new species',
        'Phase 4 - Add more beluga images',
        'Phase 6 - Final model']

    process_phases(video_paths, phases, MODELS_DIR, OUTPUT_DIR)