import torch
from tqdm import tqdm
from tools_process import compute_model_outputs

def train_model(epoch_number, train_loader, validation_loader, model, optimizer, loss_function, device):
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    for epoch in range(epoch_number):
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0

        with tqdm(train_loader, unit=" mini-batch") as progress_epoch:
            progress_epoch.set_description(f"Epoch {epoch + 1}/{epoch_number}")
            for inputs, labels in progress_epoch:
                loss, accuracy = compute_model_outputs(inputs, labels, device, model, loss_function)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_accuracy += accuracy

        train_loss = running_loss / len(train_loader)
        train_acc = running_accuracy / len(train_loader)
        val_loss, val_acc = validate_model(validation_loader, model, loss_function, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"\nEpoch {epoch + 1}/{epoch_number} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return model, history

def validate_model(validation_loader, model, loss_function, device):
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0

    with torch.no_grad():
        with tqdm(validation_loader, unit=" mini-batch") as progress_validation:
            progress_validation.set_description(" Validation step")
            for inputs, labels in progress_validation:
                loss, accuracy = compute_model_outputs(inputs, labels, device, model, loss_function)

                running_loss += loss.item()
                running_accuracy += accuracy

    val_loss = running_loss / len(validation_loader)
    val_acc = running_accuracy / len(validation_loader)
    return val_loss, val_acc
