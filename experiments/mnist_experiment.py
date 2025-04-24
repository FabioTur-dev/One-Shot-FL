import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import resnet18
from utils.data_utils import load_mnist_non_iid, get_client_loader
from client.distillation import ATOMDistiller
from server.server import Server
from utils.fisher import compute_fisher
from plots.performance_plots import plot_results


def get_resnet18_1ch(num_classes: int = 10) -> nn.Module:
    """
    Return a ResNet18 model modified to accept single-channel input.
    """
    model = resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(
        1, model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False
    )
    return model


def run_mnist_experiment(device: str = 'cuda'):
    """
    Run one-shot federated learning experiment on MNIST using ATOM distillation.
    Includes debug prints and displays distilled images without blocking.
    """
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Load non-IID MNIST and determine clients
    dataset, client_map = load_mnist_non_iid(root='./data', num_clients=5)  # set number of clients to 5
    num_clients = len(client_map)
    print(f"[DEBUG] Number of clients detected: {num_clients}")

    # Initialize backbone for single-channel images
    base = get_resnet18_1ch()
    layers = ["layer1", "layer2", "layer3", "layer4"]

    distilled_sets, label_sets = [], []
    # Hyperparameters - tuned for better accuracy
    syn_per_class = 1000  # synthetic images per class (increased)
    distill_iters = 5000  # distillation iterations (increased)
    server_epochs = 30  # global training epochs (increased)
    server_batch_size = 256  # batch size for server training

    # Enable interactive plotting for non-blocking display for non-blocking display
    plt.ion()

    # Client-side distillation with debug
    for client_id, indices in client_map.items():
        print(f"[DEBUG] Starting distillation for Client {client_id}")
        loader = get_client_loader(dataset, indices)
        num_classes = 10
        total_syn = syn_per_class * num_classes

        # Initialize synthetic MNIST-like images
        S_init = torch.randn(total_syn, 1, 28, 28)
        y_init = torch.tensor([i for i in range(num_classes) for _ in range(syn_per_class)])

        # Distill using ATOM protocol (one-shot)
        distiller = ATOMDistiller(base, layers, S_init, y_init, dev, lr=0.02)  # lower LR for smoother optimization
        S_syn, y_syn = distiller.distill(loader, iterations=distill_iters)

        print(f"[DEBUG] Client {client_id} distilled: S_syn shape = {S_syn.shape}")
        # Display first 10 distilled images (privacy-preserving)
        fig, axs = plt.subplots(2, 5, figsize=(10, 4))
        for idx, ax in enumerate(axs.flatten()):
            img = S_syn[idx].view(28, 28).cpu().numpy()
            ax.imshow(img, cmap='gray')
            ax.axis('off')
        fig.suptitle(f"Client {client_id} Distilled Images (first 10) - Privacy Preserved")
        plt.draw()
        plt.pause(0.1)

        distilled_sets.append(S_syn.view(S_syn.size(0), -1))
        label_sets.append(y_syn)

    # Disable interactive mode
    plt.ioff()

    # One-shot server aggregation and training
    print("[DEBUG] Aggregating distilled data from all clients")
    distilled_data = torch.cat(distilled_sets)
    distilled_labels = torch.cat(label_sets)
    print(f"[DEBUG] Total distilled data shape = {distilled_data.shape}")

    server_model = get_resnet18_1ch(num_classes=10)
    server = Server(server_model, dev)
    global_model = server.aggregate_and_train(
        distilled_data, distilled_labels,
        epochs=server_epochs,
        batch_size=server_batch_size
    )
    print("[DEBUG] Global model training completed")

    # Evaluation with Fisher adjustment
    full_loader = get_client_loader(dataset, np.arange(len(dataset)))
    fisher = compute_fisher(global_model, full_loader, dev)
    print("[DEBUG] Fisher Information computed")
    plot_results(global_model, full_loader, fisher, dev)


if __name__ == '__main__':
    run_mnist_experiment()