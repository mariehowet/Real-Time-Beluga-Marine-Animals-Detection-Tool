import os
import random
import numpy as np
import torch
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, confusion_matrix, recall_score, f1_score


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def compute_model_outputs(inputs, labels, device, model, loss_function, return_predictions=False):
    inputs = inputs.to(device)

    if labels.dim() > 1:
        labels = labels.argmax(dim=1).long()
    labels = labels.to(device)

    outputs = model(inputs)

    outputs = outputs.logits

    loss = loss_function(outputs, labels)
    accuracy = compute_accuracy(labels, outputs)

    if return_predictions:
        predictions = outputs.argmax(dim=1)
        return loss, accuracy, predictions, labels

    return loss, accuracy

def compute_accuracy(labels, outputs):
    predictions = outputs.argmax(dim=1)
    corrects = (predictions == labels)
    accuracy = corrects.sum().float() / float(labels.size(0))
    return accuracy


def plot_learning_curves(history, title, save_path=None):
    def to_list(x):
        return [float(t.cpu()) if isinstance(t, torch.Tensor) else t for t in x]

    train_loss = to_list(history.get('train_loss', []))
    val_loss = to_list(history.get('val_loss', []))
    train_acc = to_list(history.get('train_acc', []))
    val_acc = to_list(history.get('val_acc', []))

    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_loss, 'b-o', label='Train Loss')
    ax1.plot(epochs, val_loss, 'r-s', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_acc, 'b-o', label='Train Accuracy')
    ax2.plot(epochs, val_acc, 'r-s', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{title} - Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, title, normalize=None,save_path=None):
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)

    fmt = '.2f' if np.issubdtype(cm.dtype, np.floating) else 'd'

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def evaluate_training(timestamp, model_name, model, history,  output_dir):
    output_dir = os.path.join(output_dir, f"{model_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, f"{model_name}.pt")
    torch.save(model.state_dict(), model_path)

    curves_path = os.path.join(output_dir, f"learning_curves.png")
    plot_learning_curves(history, model_name, save_path=curves_path)


def evaluate_testing(timestamp, model_name, y_true, y_pred, class_names, output_dir="results"):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    kappa = cohen_kappa_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)

    print(f"\n=== Evaluation :  {model_name} ===")
    print(f"Precision: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Cohen Kappa: {kappa:.4f}")
    print("\nClassification Report:\n")
    print(report)

    output_dir = os.path.join(output_dir, f"{model_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    report_metrics_path = os.path.join(output_dir, f"test_metrics.txt")
    with open(report_metrics_path, "w") as f:
        f.write(f"=== Résultats du modèle {model_name} ===\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Cohen Kappa: {kappa:.4f}\n\n")
        f.write("=== Classification Report ===\n")
        f.write(report)

    cm_path = os.path.join(output_dir, f"confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, class_names, "Confusion Matrix", save_path=cm_path)

    cm_nomalized_path = os.path.join(output_dir, f"confusion_matrix_normalized.png")
    plot_confusion_matrix(y_true, y_pred, class_names,  "Confusion Matrix Normalized", "true",save_path=cm_nomalized_path)

