import torch

def test(test_loader, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        accuracies = []
        for X, y in test_loader:
            X, y = X.float().to(device), y.long().to(device)
            y_hat_prob = model(X)
            y_hat = torch.argmax(y_hat_prob, dim=1).long()
            accuracy = torch.sum(y_hat == y) / len(y)
            accuracies.append(accuracy)
    print(f"Test accuracy: {sum(accuracies) / len(accuracies)}")
            