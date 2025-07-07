import os
import torch
import matplotlib.pyplot as plt
from time import perf_counter
from tqdm import tqdm
from utils.options import get_hyperparameters


class AverageMeter:

    def __init__(self):
        self.values = []

    def append(self, value):
        self.values.append(value)
    
    @property
    def avg(self):
        return sum(self.values) / len(self.values)

def train(
    train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, model):
    """
    Træner modellen
    Args:
    train_loader (torch.utils.data.DataLoader): DataLoader for træningsdata
    val_loader (torch.utils.data.DataLoader): DataLoader for valideringsdata
    model (torch.nn.Module): Netværksarkitektur
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_acc = 0

    losses_full = []
    accuracies_full = []
    val_losses_full = []
    val_accuracies_full = []
    
    # Trænings loop over epochs
    epoch_loop = tqdm(range(model.hyperparameters.epochs))
    for epoch in epoch_loop:
        losses = AverageMeter()
        accuracies = AverageMeter()
        val_losses = AverageMeter()
        val_accuracies = AverageMeter()
        # Sæt model til træning
        model.train()
        start_time = perf_counter()
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

            #val_X, val_y = next(iter(val_loader))
            #val_y_hat = model(val_X)
            #val_accuracy = torch.sum(torch.argmax(val_y_hat, dim=1) == val_y) / len(val_y)
            #val_accuracies.append(val_accuracy)

            # Backward pass og opdatering af vægte
            loss.backward()
            model.optimizer.step()

            # Print status
            #if batch  == 0:
            #    print(
            #        f"[{epoch} / {model.hyperparameters['epochs']}] Training:   Loss: {loss:3f} Accuracy: {accuracy:3f}"
            #    )
        model.eval()
        with torch.no_grad():
            for batch, (X, y) in enumerate(val_loader):
                X, y = X.float().to(device), y.long().to(device)
                y_hat_prob = model(X)
                val_loss = model.criterion(y_hat_prob, y)
                val_losses.append(val_loss.item())
                val_accuracy = torch.sum(torch.argmax(y_hat_prob, dim=1) == y) / len(y)
                val_accuracies.append(val_accuracy)
        end_time = perf_counter()

        # Gem værdier og print status
        losses_full.append(losses.avg)
        accuracies_full.append(accuracies.avg)
        val_losses_full.append(val_losses.avg)
        val_accuracies_full.append(val_accuracies.avg)
        #(f"[{epoch+1} / {model.hyperparameters.epochs} {end_time-start_time:.2f}s] Training - Loss: {sum(losses) / len(losses):3f} Accuracy: {sum(accuracies) / len(accuracies):3f} | Validation - Loss: {sum(val_losses) / len(val_losses):3f} Accuracy: {sum(val_accuracies) / len(val_accuracies):3f}")
        epoch_loop.set_postfix({"Train_loss": f"{losses_full:.3f}", "Val_loss": f"{val_losses_full:.3f}", "Train_acc": f"{accuracies_full:.3f}", "Val_acc": f"{val_accuracies_full:.3f}", "Time_Taken": f"{end_time-start_time:.3f}"})

        # Model er trænet, gem vægtene
        if not os.path.exists("saved_models/"):
            os.makedirs("saved_models/")
        
        # Note: torch.save gemmer kun vægtene, og derfor kan kun loades hvis du først definere den korrekt model class
        # Det er fordi vægtene alene er ikke nok, da de skal blive sat til neuroner der bliver defineret i model classen
        # I det andet hånd, torch.jit.script gemmer både skeleten og vægtene, og har ikke bruge for at definere classen 
        # igen når den skal loades med torch.jit.load. Problemet med torch.jit.script er at den bliver ikke udviklet længere
        # som betyder at den kan fejle med nogle nyere lage.

        #torch.save(model.state_dict(), f"saved_models/{model.name}.pt")
        scripted_model = torch.jit.script(model)
        scripted_model.save(f'saved_models/{model.name}.pth')
        print(f"Gemt modellen fra sidste checkpoint i 'saved_models/{model.name}.pt'")

        if val_accuracies.avg > best_acc:
            best_acc = val_accuracies.avg
            #torch.save(model.state_dict(), f"saved_models/{model.name}_best.pt")
            scripted_model.save(f'saved_models/{model.name}_best.pth')
            print(f"Gemt modellen med bedste accuracy {best_acc} i 'saved_models/{model.name}_best.pt'")
    return {"Train_loss":losses_full, "Val_loss": val_losses_full, "Train_acc": accuracies_full, "Val_acc": val_accuracies_full}
    
def plot_training_logs(train_logs):
    Fig, ax = plt.subplots(4,1, sharex=True, figsize=(16,10))
    ax[0].plot(range(1,len(train_logs["Train_loss"])+1), train_logs["Train_loss"])
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Training Loss")
    ax[1].plot(range(1,len(train_logs["Val_loss"])+1), train_logs["Val_loss"])
    ax[1].set_ylabel("Loss")
    ax[1].set_title("Validation Loss")
    ax[2].plot(range(1,len(train_logs["Train_acc"])+1), train_logs["Train_acc"])
    ax[2].set_ylabel("Accuracy")
    ax[2].set_title("Training Accuracy")
    ax[3].plot(range(1,len(train_logs["Val_acc"])+1), train_logs["Val_acc"])
    ax[3].set_ylabel("Accuracy")
    ax[3].set_title("Validation Accuracy")
    ax[3].set_xlabel("Epoch")
    return Fig, ax
