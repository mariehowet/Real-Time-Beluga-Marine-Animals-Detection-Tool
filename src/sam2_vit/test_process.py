import torch
from tqdm import tqdm
from tools_process import compute_model_outputs


def test_model(test_loader, model, loss_function, device):
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        with tqdm(test_loader, unit=" mini-batch") as progress_test:
            progress_test.set_description("Testing")
            for inputs, labels in progress_test:

                loss, accuracy, predictions, true_labels = compute_model_outputs(
                    inputs, labels, device, model, loss_function, return_predictions=True
                )

                running_loss += loss.item()
                running_accuracy += accuracy

                y_true.extend(true_labels.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())

    test_loss = running_loss / len(test_loader)
    test_acc = running_accuracy / len(test_loader)

    print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    return y_true, y_pred


