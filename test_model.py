import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from data import get_FlawDataset
from tqdm import tqdm
from sklearn.metrics import f1_score
import config
from models import ResNetClassifier
from torchvision import transforms
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
        ])
    
    dataset = get_FlawDataset(config.DATA_DIR/'target', transform=transform, split_data = False)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = ResNetClassifier(num_classes=2).to(device)
    model.load_state_dict(torch.load(args.MODEL_FILE))
    model.eval()

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
    roc_auc = roc_auc_score(all_y_true, np.array(all_y_probs)[:, 0])
    print(f'Accuracy on target data: {mean_accuracy:.4f}')
    print(f'F1 Score on target data: {f1:.4f}')
    print(f'ROC AUC on target data: {roc_auc:.4f}')

    
    print("Classification report:", classification_report(all_y_true, all_y_pred))
        
    
    class_names = ['cut', 'hole', 'color', 'good']
    cm = confusion_matrix(all_y_true, all_y_pred)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,  xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Test a model on MNIST-M')
    arg_parser.add_argument('MODEL_FILE', help='A model in trained_models')
    arg_parser.add_argument('--batch-size', type=int, default=64)
    args = arg_parser.parse_args()
    main(args)