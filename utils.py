import torch
import matplotlib.pyplot as plt
from typing import List
import numpy as np


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


def plot_loss_curve(d_losses: List[float], g_losses: List[float], EPOCHS: int) -> None:
    """Plot loss curve of critic and generator.

    Args:
        d_losses (List[float]): List of Discriminator losses.
        g_losses (List[float]): List of generator losses.
        EPOCHS (int): Total number of epochs.
    """
    # normalize losses to get a nice graph
    gen_losses = np.asarray(g_losses)
    critic_losses = np.asarray(d_losses)

    gen_losses = (gen_losses - gen_losses.min()) / (gen_losses.max() - gen_losses.min())
    critic_losses = np.asarray(d_losses)

    critic_losses = (critic_losses - critic_losses.min()) / (
        critic_losses.max() - critic_losses.min()
    )
    plt.figure(figsize=(20, 10))
    x = range(1, EPOCHS + 1)
    plt.plot(x, critic_losses, color="red", label="Discriminator Loss")
    plt.plot(x, gen_losses, color="blue", label="Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Curve")
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.show()
