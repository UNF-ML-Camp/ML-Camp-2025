"""Modul til træning af neurale netværk"""

import torch
import mlflow
from torch import nn
from options import get_hyperparameters
from time import perf_counter

# Sæt mlflow tracking URI og experiment
mlflow.set_tracking_uri("./mlruns")

def train(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    model: nn.Module,
    ) -> None:
    """
    Træner modellen
    Args:
    train_loader (torch.utils.data.DataLoader): DataLoader for træningsdata
    val_loader (torch.utils.data.DataLoader): DataLoader for valideringsdata
    model (torch.nn.Module): Netværksarkitektur
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Valider at modelen har en optimizer
    if not hasattr(model, "optimizer"):
        raise ValueError("Modelen mangler en optimizer")
    
    # Start mlflow run
    with mlflow.start_run(experiment_id="0", run_name = model.name):
        # Log hyperparametre
        mlflow.log_params(get_hyperparameters(model.hyperparameters, model.optimizer))
        
        # Trænings loop over epochs
        best_val_accuracy = 0
        for epoch in range(model.hyperparameters.epochs):
            losses = []
            val_losses = []
            accuracies = []
            val_accuracies = []
            start_time = perf_counter()

            # Sæt model til træning
            model.train()

            for batch, (X, y) in enumerate(train_loader):
                X, y = X.float().to(device), y.long().to(device)

                # Genstart gradienter
                model.optimizer.zero_grad()

                # Forward pass
                y_hat_prob = model(X)
                y_hat = torch.argmax(y_hat_prob, dim=1).long()
                
                # Beregn loss, accuracy, og validation accuracy
                loss = model.criterion(y_hat_prob, y)
                losses.append(loss.item())
                accuracy = torch.sum(y_hat == y) / len(y)
                accuracies.append(accuracy)

                # Backward pass og opdatering af vægte
                loss.backward()
                model.optimizer.step()

            # Sæt model til evaluation
            model.eval()
            with torch.no_grad():
                for batch, (X_val, y_val) in enumerate(val_loader):
                    X_val, y_val = X_val.float().to(device), y_val.long().to(device)
                    val_y_hat_prob = model(X_val)

                    # Beregn loss og accuracy
                    val_loss = model.criterion(val_y_hat_prob, y_val)
                    val_losses.append(val_loss.item())
                    val_accuracy = torch.sum(torch.argmax(val_y_hat_prob, dim=1) == y_val) / len(y_val)
                    val_accuracies.append(val_accuracy)

            # Print status
            end_time = perf_counter()
            mean_accuracy = sum(accuracies) / len(accuracies)
            mean_val_accuracy = sum(val_accuracies) / len(val_accuracies)
            mean_loss = sum(losses) / len(losses)
            mean_val_loss = sum(val_losses) / len(val_losses)

            print(
                f"[{epoch+1} / {model.hyperparameters.epochs} ({end_time-start_time:.2f}s)] Training - Loss: {mean_loss:3f} Accuracy: {mean_accuracy:3f} | Validation - Loss: {mean_val_loss:3f} Accuracy: {mean_val_accuracy:3f}"
            )

            # Log loss og accuracy
            mlflow.log_metric("loss", mean_loss, step=epoch)
            mlflow.log_metric("accuracy", mean_accuracy, step=epoch)
            mlflow.log_metric("val_loss", mean_val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", mean_val_accuracy, step=epoch)
            mlflow.log_metric("time_per_epoch", end_time-start_time, step=epoch)
        
            if mean_val_accuracy > best_val_accuracy:
                # Vi har fundet en bedre model, så lad os gemme den ved den nuværende epoch
                best_val_accuracy = mean_val_accuracy
                model.save()