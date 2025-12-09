from segmentation import extract_bboxes_from_images, extract_bboxes_from_video
from transformers import ViTForImageClassification, ViTImageProcessor
import os
import torch
from PIL import Image
import shutil
import pandas as pd


INPUT_IMAGES = "images"
OUTPUT_IMAGES_SEG = "../../outputs/Phase 5 - SAM2  & ViT/prediction-images/images_seg"
OUTPUT_IMAGES_CROP = "../../outputs/Phase 5 - SAM2  & ViT/prediction-images/images_crop"

INPUT_VIDEOS = "../videos/sam2_vit"
OUTPUT_VIDEO_SEG = "../../outputs/Phase 5 - SAM2  & ViT/prediction-videos/videos_seg"
OUTPUT_VIDEO_CROP = "../../outputs/Phase 5 - SAM2  & ViT/prediction-videos/videos_crop"

MODEL_DIR = "../../models/Phase 5 - SAM2  & ViT/vit_model_2025-11-28_10h08min32sec/vit_model.pt"
CLASSES = ["beluga", "dolphin", "other"]


def get_video_paths(video_dir):
    return [os.path.join(root, f)
            for root, _, files in os.walk(video_dir)
            for f in files if f.lower().endswith(('.mp4', '.avi', '.mov'))]


def load_model(model_path, pretrained_model, classes):
    model = ViTForImageClassification.from_pretrained(pretrained_model)

    model.classifier = torch.nn.Linear(model.classifier.in_features, len(classes))

    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    return model


def classify_crops_with_vit(model, processor, crop_dir):
    for img_name in os.listdir(crop_dir):
        img_path = os.path.join(crop_dir, img_name)

        img = Image.open(img_path).convert("RGB")
        inputs = processor(img, return_tensors="pt")

        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits).item()
        pred_class = CLASSES[pred]

        class_folder = os.path.join(crop_dir, "..", "predicted", pred_class)
        os.makedirs(class_folder, exist_ok=True)

        shutil.copy(img_path, os.path.join(class_folder, img_name))



def run_image_pipeline(input_dir, output_dir_seg, output_dir_crop, vit_model, processor):
    extract_bboxes_from_images(input_dir, output_dir_seg, output_dir_crop)
    classify_crops_with_vit(vit_model, processor, output_dir_crop)


def run_video_pipeline(video_path, output_dir_seg, output_dir_crop, vit_model, processor, max_frames):
    extract_bboxes_from_video(video_path, output_dir_seg, output_dir_crop, max_frames)
    classify_crops_with_vit(vit_model, processor, output_dir_crop)

if __name__ == "__main__":
    PRETRAINED_MODEL = "google/vit-base-patch16-224"

    model = load_model(MODEL_DIR, PRETRAINED_MODEL, CLASSES)

    processor = ViTImageProcessor.from_pretrained(PRETRAINED_MODEL)

    video_paths = get_video_paths(INPUT_VIDEOS)
    run_image_pipeline(INPUT_IMAGES, OUTPUT_IMAGES_SEG, OUTPUT_IMAGES_CROP, model, processor)

    for video_path in video_paths:
        run_video_pipeline(video_path, OUTPUT_VIDEO_SEG, OUTPUT_VIDEO_CROP, model, processor, max_frames=5)




