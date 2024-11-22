import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import config
from models import Net
from utils import GrayscaleToRgb
from data import TextureDataset
from sklearn.metrics import f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_dataloaders(batch_size=64):
    
    dataset = TextureDataset(config.DATA_DIR/'source')
    
    train_size = int(0.8 * len(dataset)) 
    test_size = len(dataset) - train_size  

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

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

    return mean_loss, mean_accuracy, f1


def main(args):
    train_loader, val_loader = create_dataloaders(args.batch_size)

    model = Net().to(device)
    optim = torch.optim.Adam(model.parameters())
    lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=1, verbose=True)
    criterion = torch.nn.CrossEntropyLoss()

    best_accuracy = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss, train_accuracy, train_f1 = do_epoch(model, train_loader, criterion, optim=optim)

        model.eval()
        with torch.no_grad():
            test_loss, test_accuracy, test_f1 = do_epoch(model, val_loader, criterion, optim=None)

        tqdm.write(f'EPOCH {epoch:03d}: train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f} , train_f1={train_f1:.4f} '
                   f'test_loss={test_loss:.4f}, test_accuracy={test_accuracy:.4f}, test_f1={test_f1:.4f}')

        if test_accuracy > best_accuracy:
            print('Saving model...')
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), 'trained_models/source.pt')

        lr_schedule.step(test_loss)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Train a network on MNIST')
    arg_parser.add_argument('--batch-size', type=int, default=64)
    arg_parser.add_argument('--epochs', type=int, default=30)
    args = arg_parser.parse_args()
    main(args)
