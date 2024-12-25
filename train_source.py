import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import config
from models import Net
from data import get_FlawDataset
from sklearn.metrics import f1_score
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score
from matplotlib import pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_dataloaders(batch_size):
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
        ])
    
    train_dataset, test_dataset = get_FlawDataset(config.DATA_DIR/'source', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def test_model(model, dataloader, plot_confusion_matrix=True):

    total_accuracy = 0
    all_y_true = []
    all_y_pred = []
    all_y_probs = []

    model.eval()
    with torch.no_grad():
        for x, y_true in tqdm(dataloader, leave=False):
            x, y_true = x.to(device), y_true.to(device)
            y_pred = model(x)
            y_prob = torch.softmax(y_pred, dim=1)
            total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()
            all_y_true.extend(y_true.cpu().numpy())
            all_y_pred.extend(y_pred.max(1)[1].cpu().numpy())
            all_y_probs.extend(y_prob.cpu().numpy())
    
    mean_accuracy = total_accuracy / len(dataloader)
    f1 = f1_score(all_y_true, all_y_pred, average='weighted')
    roc_auc = roc_auc_score(all_y_true, all_y_probs, multi_class='ovr', average='weighted')

    print(f'Accuracy on test data: {mean_accuracy:.4f}')
    print(f'F1 Score on test data: {f1:.4f}')
    print(f'ROC AUC on test data: {roc_auc:.4f}')
    print("Classification report:", classification_report(all_y_true, all_y_pred))

    if plot_confusion_matrix:
        class_names = ['cut', 'hole', 'color', 'good']
        cm = confusion_matrix(all_y_true, all_y_pred)
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,  xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()


def do_epoch(model, dataloader, criterion, optim):
    total_loss = 0
    total_accuracy = 0
    all_y_true = []
    all_y_pred = []
    for x, y_true in tqdm(dataloader, leave=False):
        x, y_true = x.to(device), y_true.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y_true)

        optim.zero_grad()
        loss.backward()
        optim.step()

        total_loss += loss.item()
        total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()
        all_y_true.extend(y_true.cpu().numpy())
        all_y_pred.extend(y_pred.max(1)[1].cpu().numpy())

    mean_loss = total_loss / len(dataloader)
    mean_accuracy = total_accuracy / len(dataloader)
    f1 = f1_score(all_y_true, all_y_pred, average='weighted')

    return mean_loss, mean_accuracy, f1


def main(args):
    train_loader, val_loader = create_dataloaders(args.batch_size)

    model = Net().to(device)
    optim = torch.optim.Adam(model.parameters())
    #lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=1, verbose=True)
    criterion = torch.nn.CrossEntropyLoss()

    best_accuracy = 0

    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss, train_accuracy, train_f1 = do_epoch(model, train_loader, criterion, optim)
        print("EPOCH {:03d}: train_loss={:.4f}, train_accuracy={:.4f}, train_f1={:.4f}".format(epoch, train_loss, train_accuracy, train_f1))
        
        if train_accuracy > best_accuracy:
            print('Saving model...')
            best_accuracy = train_accuracy
            torch.save(model.state_dict(), 'trained_models/source.pt')

        #lr_schedule.step(test_loss)

    test_model(model, val_loader)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Train a network on MNIST')
    arg_parser.add_argument('--batch_size', type=int, default=64)
    arg_parser.add_argument('--epochs', type=int, default=30)
    args = arg_parser.parse_args()
    main(args)
