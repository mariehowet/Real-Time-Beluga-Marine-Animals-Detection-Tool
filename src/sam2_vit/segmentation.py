from ultralytics import SAM
import os
import cv2

INPUT_DIR = "../../data/sam2_vit_marine_animals"
OUTPUT_DIR_SEG = "../../outputs/Phase 5 - SAM2  & ViT/segmentation/sam2_vit_marine_segmented_images"
OUTPUT_DIR_CROP = "../../outputs/Phase 5 - SAM2  & ViT/segmentation/sam2_vit_marine_cropped_images"

CLASSES = ["beluga", "dolphin", "other"]

model = SAM("sam2_b.pt")

# Check if prompt_encoder exists
if not hasattr(model, "prompt_encoder") and hasattr(model, "sam_prompt_encoder"):
    model.prompt_encoder = model.sam_prompt_encoder

def segment_image(image_path):
    results =  model(image_path, conf=0.8, iou =0.5, save=False)
    return results[0]

def segment_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb, conf=0.8, iou=0.5, save=False)
    return results[0]

def save_cropped_images(img, filtered_boxes, image_idx,  output_class_dir_crop):
    saved_crops = []
    for idx, box in enumerate(filtered_boxes):
        x1, y1, x2, y2 = box.astype(int)
        cropped = img[y1:y2, x1:x2]
        crop_path = f"{output_class_dir_crop}/{image_idx}_cropped_{idx}.png"
        os.makedirs(os.path.dirname(crop_path), exist_ok=True)
        cv2.imwrite(crop_path, cropped)
        saved_crops.append(crop_path)
    return saved_crops

def save_image_with_boxes(img, filtered_bboxes, path):
    for box in filtered_bboxes:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)

def filter_boxes(boxes, min_box_size):
    final_boxes = []
    if boxes is not None and len(boxes) > 0:
        raw_boxes = boxes.xyxy.cpu().numpy()

        for box in raw_boxes:
            x1, y1, x2, y2 = box.astype(int)

            if x1 <= 1 and y1 <= 1 and len(raw_boxes) > 1:
                    continue
            if (x2 - x1) < min_box_size or (y2 - y1) < min_box_size:
                continue

            final_boxes.append(box)

    return final_boxes


def extract_bboxes_from_images(input_dir, output_dir_seg, output_dir_crop):
        os.makedirs(output_dir_seg, exist_ok=True)
        os.makedirs(output_dir_crop, exist_ok=True)

        image_files = os.listdir(input_dir)

        for idx, image_file in enumerate(image_files):
            image_path = os.path.join(input_dir, image_file)
            img = cv2.imread(image_path)
            img_copy = img.copy()

            result = segment_image(image_path) # or segment video
            boxes = filter_boxes(result.boxes, min_box_size=20)

            result_image = f"{output_dir_seg}/img_{idx}_seg.png"
            save_image_with_boxes(img_copy, boxes, result_image)

            crops = save_cropped_images(img, boxes, f"img_{idx}", output_dir_crop)
            print(f"Save {len(crops)} cropped images to {output_dir_crop}")


def extract_bboxes_from_video(video_path, output_dir_seg, output_dir_crop, max_frames = 20):
    os.makedirs(output_dir_seg, exist_ok=True)
    os.makedirs(output_dir_crop, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    processed = 0

    while True:
        if processed >= max_frames:
            print(f"[STOP] Limite de {max_frames} frames atteinte.")
            break

        ret, frame = cap.read()
        if not ret:
            break

        result = segment_frame(frame)
        boxes = filter_boxes(result.boxes, min_box_size=20)

        result_path = os.path.join(output_dir_seg, f"{video_name}_frame_{frame_id}_seg.png")
        save_image_with_boxes(frame.copy(), boxes, result_path)

        crops = save_cropped_images(frame, boxes, f"{video_name}_frame_{frame_id}", output_dir_crop)

        print(f"[INFO] Frame {frame_id} → {len(crops)} crops saved")

        frame_id += 1
        processed += 1

    cap.release()
    print(f"[DONE] {processed} frames processed from video {video_path}")

if __name__ == "__main__":
    for class_name in classes:
        input_class_dir = os.path.join(INPUT_DIR, class_name)
        output_class_dir_seg = os.path.join(OUTPUT_DIR_SEG, class_name)
        output_class_dir_crop = os.path.join(OUTPUT_DIR_CROP, class_name)

        extract_bboxes_from_images(input_class_dir, output_class_dir_seg, output_class_dir_crop)

