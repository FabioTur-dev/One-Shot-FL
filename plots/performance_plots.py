# File: plots/performance_plots.py

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score

def plot_results(model, data_loader, fisher_matrix, device):
    model.to(device).eval()
    ys, preds = [], []

    for images, labels in data_loader:
        x = images.to(device)
        # Se i dati sono flattenati (shape [B, 784]), rimodella in [B,1,28,28]
        if x.dim() == 2:
            x = x.view(x.size(0), 1, 28, 28)
        outputs = model(x)
        _, p = torch.max(outputs, dim=1)
        ys.extend(labels.numpy())
        preds.extend(p.cpu().numpy())

    # Calcolo accuracy
    acc = accuracy_score(ys, preds)
    print(f"Global Model Accuracy: {acc:.4f}")

    # Plot a barre dell’accuracy
    plt.figure()
    plt.bar(['Accuracy'], [acc])
    plt.ylabel('Accuracy')
    plt.title('Global Model Performance')
    plt.show()

    # Qui puoi aggiungere altri plot, ad esempio curve di loss o distribuzione della Fisher matrix
