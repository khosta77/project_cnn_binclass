from model import SimpleCNN, DeepCNN
from dataloader import get_data_loaders
from config import *
import torch
import torch.optim as optim
from torch.nn import BCELoss
import matplotlib.pyplot as plt
from IPython.display import clear_output

import progressbar
from tqdm import tqdm

from torch import device, cuda

def plot_metrics(train_loss_history, val_loss_history, train_acc_history, val_acc_history, model_name):
    clear_output(wait=True)
    plt.figure(figsize=(12, 6))

    # loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Loss over Epochs ({model_name})")
    plt.legend()
    plt.grid()

    # acc plot
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label="Train Accuracy")
    plt.plot(val_acc_history, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy over Epochs ({model_name})")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(f"{model_name}_training_plot.png")


def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct = 0, 0
    total_samples = 0

    for inputs, labels in tqdm(data_loader):
        inputs = inputs.to(device)

        outputs = model(inputs)

        labels = labels.unsqueeze(1).float().to(device)

        loss = criterion(outputs, labels).to(device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_correct += ((outputs > 0.5).int() == labels.int()).sum().item()
        total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    accuracy = (total_correct / total_samples) * 100

    return avg_loss, accuracy


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            labels = labels = labels.unsqueeze(1).float().to(device) #labels.float().to(device)
            outputs = model(images)
            loss = criterion(outputs, labels).to(device)

            running_loss += loss.item()

            #total_loss += loss.item() * inputs.size(0)
            correct += ((outputs > 0.5).int() == labels.int()).sum().item()
           #total_samples += inputs.size(0)

            #_, predicted = outputs.max(1)
            total += labels.size(0)
           # correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100.0 * correct / total


def train_and_evaluate(model, model_name, train_loader, val_loader, device):
    model.to(device)
    criterion = BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []

    for epoch in range(EPOCHS):
        print(f"Training {model_name} - Epoch {epoch + 1}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")
        plot_metrics(train_loss_history, val_loss_history, train_acc_history, val_acc_history, model_name)

    model_path = f"{model_name}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model {model_name} saved to {model_path}")

if __name__ == "__main__":
    DEVICE=device("cuda:0" if cuda.is_available() else "cpu")
    print(DEVICE, device)

    train_loader, val_loader = get_data_loaders(DATA_DIR, IMG_SIZE, BATCH_SIZE)

    models = {
        #"SimpleCNN": SimpleCNN(img_size=IMG_SIZE),
        "DeepCNN": DeepCNN(img_size=IMG_SIZE)
    }

    for model_name, model in models.items():
        train_and_evaluate(model, model_name, train_loader, val_loader, DEVICE)
