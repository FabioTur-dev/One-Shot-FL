import torch
from torch.utils.data import DataLoader, TensorDataset

class Server:
    """
    Server-side aggregation and global model training.
    Handles flattened or image-shaped tensors.
    """
    def __init__(self, model: torch.nn.Module, device: torch.device, lr: float = 0.1):
        self.global_model = model.to(device)
        self.device = device
        self.lr = lr

    def aggregate_and_train(self, distilled_data: torch.Tensor,
                            distilled_labels: torch.Tensor,
                            epochs: int = 5,
                            batch_size: int = 64):
        """
        Aggregates distilled data and trains the global model.
        Supports inputs as flattened vectors or images.
        """
        ds = TensorDataset(distilled_data, distilled_labels)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.SGD(self.global_model.parameters(), lr=self.lr)
        self.global_model.train()

        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                # Reshape if flattened
                if xb.dim() == 2:
                    xb = xb.view(xb.size(0), 1, 28, 28)
                preds = self.global_model(xb)
                loss = torch.nn.functional.cross_entropy(preds, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self.global_model