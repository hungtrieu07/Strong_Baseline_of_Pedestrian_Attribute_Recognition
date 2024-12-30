import os
import numpy as np
import random
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from easydict import EasyDict
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

np.random.seed(0)
random.seed(0)

group_order = [10, 18, 19, 30, 15, 7, 9, 11, 14, 21, 26, 29, 32, 33, 34, 6, 8, 12, 25, 27, 31, 13, 23, 24, 28, 4, 5,
               17, 20, 22, 0, 1, 2, 3, 16]

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def generate_data_description(save_dir, reorder):
    """
    Create a dataset description file, which consists of images and labels
    """
    peta_data = loadmat(os.path.join(save_dir, 'PETA.mat'))
    dataset = EasyDict()
    dataset.description = 'peta'
    dataset.reorder = 'group_order'
    dataset.root = os.path.join(save_dir, 'images')
    dataset.image_name = [f'{i + 1:05}.png' for i in range(19000)]

    raw_attr_name = [i[0][0] for i in peta_data['peta'][0][0][1]]
    raw_label = peta_data['peta'][0][0][0][:, 4:]

    dataset.label = raw_label[:, :35]
    dataset.attr_name = raw_attr_name[:35]
    if reorder:
        dataset.label = dataset.label[:, np.array(group_order)]
        dataset.attr_name = [dataset.attr_name[i] for i in group_order]

    dataset.partition = EasyDict()
    dataset.partition.train = []
    dataset.partition.val = []
    dataset.partition.trainval = []
    dataset.partition.test = []

    dataset.weight_train = []
    dataset.weight_trainval = []

    for idx in range(5):
        train = peta_data['peta'][0][0][3][idx][0][0][0][0][:, 0] - 1
        val = peta_data['peta'][0][0][3][idx][0][0][0][1][:, 0] - 1
        test = peta_data['peta'][0][0][3][idx][0][0][0][2][:, 0] - 1
        trainval = np.concatenate((train, val), axis=0)

        dataset.partition.train.append(train)
        dataset.partition.val.append(val)
        dataset.partition.trainval.append(trainval)
        dataset.partition.test.append(test)

        weight_train = np.mean(dataset.label[train], axis=0)
        weight_trainval = np.mean(dataset.label[trainval], axis=0)

        dataset.weight_train.append(weight_train)
        dataset.weight_trainval.append(weight_trainval)

    with open(os.path.join(save_dir, 'dataset.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)

class PETADataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, labels

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, save_path='model.pth', log_path='training.log'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    best_val_loss = float('inf')

    with open(log_path, 'w') as log_file:
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} - Training") as pbar:
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * inputs.size(0)
                    pbar.set_postfix({'Train Loss': loss.item()})
                    pbar.update(1)

            train_loss /= len(train_loader.dataset)
            train_log = f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}"
            print(train_log)
            log_file.write(train_log + '\n')

            model.eval()
            val_loss = 0.0
            all_targets = []
            all_preds = []

            with torch.no_grad():
                with tqdm(total=len(val_loader), desc=f"Epoch {epoch+1}/{num_epochs} - Validation") as pbar:
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(device), targets.to(device)

                        outputs = model(inputs)
                        loss = criterion(outputs, targets)

                        val_loss += loss.item() * inputs.size(0)

                        preds = (torch.sigmoid(outputs) > 0.5).float()
                        all_preds.append(preds.cpu())
                        all_targets.append(targets.cpu())

                        pbar.set_postfix({'Val Loss': loss.item()})
                        pbar.update(1)

            val_loss /= len(val_loader.dataset)
            all_preds = torch.cat(all_preds, dim=0).numpy()
            all_targets = torch.cat(all_targets, dim=0).numpy()
            accuracy = accuracy_score(all_targets, all_preds)

            val_log = f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}"
            print(val_log)
            log_file.write(val_log + '\n')

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                model_log = f"Model improved. Saved to {save_path}"
                print(model_log)
                log_file.write(model_log + '\n')

if __name__ == "__main__":
    save_dir = './data/PETA/'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    os.makedirs("exp_result/resnet50", exist_ok=True)

    # Generate the dataset description
    generate_data_description(save_dir, True)

    # Load dataset description
    with open(os.path.join(save_dir, 'dataset.pkl'), 'rb+') as f:
        dataset = pickle.load(f)

    train_transform = transforms.Compose([
        transforms.Resize((256, 192)),
        transforms.Pad(10),
        transforms.RandomCrop((256, 192)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    valid_transform = transforms.Compose([
        transforms.Resize((256, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_indices = dataset.partition.train[0]
    val_indices = dataset.partition.val[0]

    train_dataset = PETADataset(
        [os.path.join(dataset.root, dataset.image_name[i]) for i in train_indices],
        dataset.label[train_indices],
        transform=train_transform,
    )

    val_dataset = PETADataset(
        [os.path.join(dataset.root, dataset.image_name[i]) for i in val_indices],
        dataset.label[val_indices],
        transform=valid_transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    num_labels = dataset.label.shape[1]
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_labels)

    weight_train = torch.tensor(dataset.weight_train[0], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight_train)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=300, save_path='exp_result/resnet50/peta_resnet50.pth', log_path='exp_result/resnet50/peta_resnet50.log')
