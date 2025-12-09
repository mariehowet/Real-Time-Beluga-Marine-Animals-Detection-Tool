import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from train_process import train_model
from test_process import test_model
from tools_process import evaluate_training, evaluate_testing
from datetime import datetime

# ====================== CONFIGURATION ======================
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.01
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATASET_PATH = "../../outputs/Phase 5 - SAM2  & ViT/segmentation/sam2_vit_marine_cropped_images"
OUTPUT_TRAIN_DIR = "../../models/Phase 5 - SAM2  & ViT"
OUTPUT_EVAL_DIR = "../../outputs/Phase 5 - SAM2  & ViT/evaluation"

CLASSES = ["beluga", "dolphin", "other"]

PRETRAINED_MODEL = "google/vit-base-patch16-224"

# ====================== PROCESSOR VIT ======================
processor = ViTImageProcessor.from_pretrained(PRETRAINED_MODEL)

mean = processor.image_mean
std = processor.image_std

# ====================== DATA TRANSFORMATION + AUGMENTATION ======================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

dataset = datasets.ImageFolder(root=DATASET_PATH)

# ====================== SPLIT TRAIN / VALIDATION / TEST SETS ======================
generator1 = torch.Generator().manual_seed(42) # random seed pour diviser le dataset toujours de la même façon
train_set, validation_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1], generator=generator1)

train_set.dataset.transform = train_transform
validation_set.dataset.transform = val_test_transform
test_set.dataset.transform = val_test_transform

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

# ====================== MODEL ======================
classes = dataset.classes
class_number = len(classes)

model = ViTForImageClassification.from_pretrained(
    PRETRAINED_MODEL,
    num_labels=class_number,
    ignore_mismatched_sizes=True
)

# Freeze all layers of the ViT model
for param in model.vit.parameters():
    param.requires_grad = False

# Unfreeze the classifier layer
for param in model.classifier.parameters():
    param.requires_grad = True

model.to(DEVICE)
# ====================== LOSS AND OPTIMIZER ======================
loss_function = nn.CrossEntropyLoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE) # Met à jour uniquement les paramètres entraînables de la dernière couche

# ====================== TRAINING  ======================
model, history = train_model( EPOCHS, train_loader, validation_loader,model, optimizer, loss_function, DEVICE)

timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%Mmin%Ssec")
model_name = "vit_model"

evaluate_training(timestamp, model_name, model, history, OUTPUT_TRAIN_DIR)

# ====================== EVALUATION ======================
y_true, y_pred = test_model(test_loader, model, loss_function, DEVICE)
evaluate_testing(
            timestamp=timestamp,
            model_name="vit_model",
            y_true=y_true,
            y_pred=y_pred,
            class_names=CLASSES,
            output_dir=OUTPUT_EVAL_DIR
        )

