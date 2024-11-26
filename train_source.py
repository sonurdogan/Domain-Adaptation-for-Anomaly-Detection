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
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_dataloaders(batch_size=64):
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
        ])
    
    train_dataset, test_dataset = get_FlawDataset(config.DATA_DIR/'source', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def do_epoch(model, dataloader, criterion, optim=None):
    total_loss = 0
    total_accuracy = 0
    all_y_true = []
    all_y_pred = []
    for x, y_true in tqdm(dataloader, leave=False):
        x, y_true = x.to(device), y_true.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y_true)

        if optim is not None:
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

    return mean_loss, mean_accuracy, f1, all_y_true, all_y_pred


def main(args):
    train_loader, val_loader = create_dataloaders(args.batch_size)

    model = Net().to(device)
    optim = torch.optim.Adam(model.parameters())
    lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=1, verbose=True)
    criterion = torch.nn.CrossEntropyLoss()

    best_accuracy = 0
    all_val_y_true = []
    all_val_y_pred = []
    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss, train_accuracy, train_f1, _, _ = do_epoch(model, train_loader, criterion, optim=optim)

        model.eval()
        with torch.no_grad():
            test_loss, test_accuracy, test_f1, val_y_true, val_y_pred = do_epoch(model, val_loader, criterion, optim=None)
            all_val_y_true.extend(val_y_true)
            all_val_y_pred.extend(val_y_pred)

        tqdm.write(f'EPOCH {epoch:03d}: train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f} , train_f1={train_f1:.4f} '
                   f'test_loss={test_loss:.4f}, test_accuracy={test_accuracy:.4f}, test_f1={test_f1:.4f}')

        if test_accuracy > best_accuracy:
            print('Saving model...')
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), 'trained_models/source.pt')

        lr_schedule.step(test_loss)


    final_f1 = f1_score(all_val_y_true, all_val_y_pred, average='weighted')
    final_accuracy = np.mean(np.array(all_val_y_true) == np.array(all_val_y_pred))

    print("Classification report:", classification_report(all_val_y_true, all_val_y_pred))
    print(f'Final Validation Accuracy: {final_accuracy:.4f}')
    print(f'Final Validation F1 Score: {final_f1:.4f}')
    print('Confusion Matrix:')
    class_names = ['cut', 'hole', 'color', 'good']
    cm = confusion_matrix(all_val_y_true, all_val_y_pred)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,  xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Train a network on MNIST')
    arg_parser.add_argument('--batch_size', type=int, default=64)
    arg_parser.add_argument('--epochs', type=int, default=30)
    args = arg_parser.parse_args()
    main(args)
