import torch

def compute_fisher(model: torch.nn.Module, data_loader, device: torch.device):
    """
    Compute diagonal Fisher Information Matrix, supporting image or flattened inputs.
    """
    model.to(device).eval()
    fisher = {name: torch.zeros_like(param)
              for name, param in model.named_parameters() if param.requires_grad}

    for inputs, targets in data_loader:
        model.zero_grad()
        # Move and reshape inputs
        x = inputs.to(device)
        if x.dim() == 2:
            # flattened to image
            x = x.view(x.size(0), 1, 28, 28)
        y = targets.to(device)

        outputs = model(x)
        loss = torch.nn.functional.nll_loss(
            torch.log_softmax(outputs, dim=1), y)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += param.grad.data.pow(2)

    # Average over number of batches
    num_batches = len(data_loader)
    for name in fisher:
        fisher[name] /= num_batches
    return fisher