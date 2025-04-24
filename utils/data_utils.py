import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

def load_mnist_non_iid(root: str, num_clients: int = 10, shards_per_client: int = 2):
    dataset = MNIST(root=root, train=True, download=True,
                    transform=transforms.ToTensor())
    labels = dataset.targets.numpy()
    idxs = np.argsort(labels)
    shards = np.array_split(idxs, num_clients * shards_per_client)
    client_indices = {
        i: np.concatenate(shards[i*shards_per_client:(i+1)*shards_per_client])
        for i in range(num_clients)
    }
    return dataset, client_indices


def get_client_loader(dataset, indices, batch_size=64):
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True)