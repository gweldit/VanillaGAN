from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def get_device():
    """Get the preferred device:
    check if there is a GPU available:NVIDIA GPU  or MPS (Apple Silcon GPU) if available).
    """

    # check if there is nvidia or cuda gpu
    if torch.cuda.is_available():
        return torch.device("cuda")

    # check if there is an apple silicon gpu
    if torch.backends.mps.is_available():
        return torch.device("mps")

    # otherwise use the cpu
    return torch.device("cpu")


def plot_loss_curve(
    d_losses: List[float], g_losses: List[float], EPOCHS: int, normalize=False
) -> None:
    """Plot loss curve of critic and generator.

    Args:
        d_losses (List[float]): List of Discriminator losses.
        g_losses (List[float]): List of generator losses.
        EPOCHS (int): Total number of epochs.
    """
    x = range(1, EPOCHS + 1)
    if normalize:
        # normalize losses to get a nice graph
        g_losses = np.asarray(g_losses)
        d_losses = np.asarray(d_losses)

        g_losses = (g_losses - g_losses.min()) / (g_losses.max() - g_losses.min())
        d_losses = np.asarray(d_losses)

        d_losses = (d_losses - d_losses.min()) / (d_losses.max() - d_losses.min())

        plt.figure(figsize=(20, 14))

        plt.plot(x, d_losses, color="red", label="Discriminator Loss")
        plt.plot(x, g_losses, color="blue", label="Generator Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss Curve")
        plt.legend()
        plt.savefig("loss_curve.png")
        plt.show()

    else:
        # use subplot curves side by side
        plt.figure(figsize=(20, 14))

        plt.subplot(1, 2, 1)
        plt.plot(x, d_losses, color="red", label="Discriminator Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss Curve")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(x, g_losses, color="blue", label="Generator Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss Curve")
        plt.legend()
        plt.savefig("loss_curve.png")
        plt.show()


def evaluate_gcn_model(gcn_model, graph_val_loader, vocab_size, device):
    gcn_model.eval()
    # val_losses = []
    # val_f1_scores = []

    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in graph_val_loader:
            x = data.x[0].to(device)
            x = F.one_hot(x, num_classes=vocab_size).float()
            edge_index = data.edge_index.to(device)
            edge_weight = data.weight.float().to(device)
            batch = data.batch.to(device)

            out = gcn_model(x, edge_index, edge_weight, batch)
            # loss = F.cross_entropy(out, data.y)
            # val_losses.append(loss.item())

            preds = out.argmax(dim=1).cpu().numpy()
            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(preds)

        print("GCN performance:")
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print(f"Precision: {precision_score(y_true, y_pred, average='macro'):.4f}")
        print(f"Recall: {recall_score(y_true, y_pred, average='macro'):.4f}")
        print(
            f"F1 score: {f1_score(y_true, y_pred, average='macro', zero_division=0.0):.4f}"
        )
        print(f"MCC: {matthews_corrcoef(y_true, y_pred):.4f}")
