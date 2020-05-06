import os

import wandb
from box import Box
from lgblkb_tools import logger, Folder
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lgblkb_tools.visualize import Plotter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import torch
import torchvision
import torchvision.transforms as transforms

is_cuda_available = torch.cuda.is_available()
logger.info('is_cuda_available: %s', is_cuda_available)
if not is_cuda_available:
    raise SystemError
device = 'cuda' if is_cuda_available else 'cpu'

this_folder = Folder(__file__)
data_folder = this_folder.parent()['data']

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyper-parameters


def get_loaders(batch_size):
    raw_X = np.load(data_folder['X.npy'])
    raw_y = np.load(data_folder['y.npy'])
    scaler = MinMaxScaler()
    scaler.fit(raw_X.reshape((-1, 4)))
    X = scaler.transform(raw_X.reshape((-1, 4))).reshape((-1, 4, 4))
    y = scaler.transform(raw_y.reshape((-1, 4))).reshape((-1, 4))
    
    logger.debug("X.shape: %s", X.shape)
    logger.debug("y.shape: %s", y.shape)
    x_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()
    whole_dataset = TheDataset(x_tensor, y_tensor)
    
    train_data, test_data = train_test_split(whole_dataset)
    logger.debug("len(train_data): %s", len(train_data))
    logger.debug("len(test_data): %s", len(test_data))
    
    # Data loader
    train_loader = DataLoader(dataset=train_data,
                              batch_size=batch_size,
                              shuffle=True, )
    
    test_loader = DataLoader(dataset=test_data,
                             batch_size=batch_size,
                             shuffle=False)
    return train_loader, test_loader


# MNIST dataset
def get_mnist_loaders(batch_size):
    train_dataset = torchvision.datasets.MNIST(root='../data/',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    
    test_dataset = torchvision.datasets.MNIST(root='../data/',
                                              train=False,
                                              transform=transforms.ToTensor())
    
    # Data loader
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False)
    return train_loader, test_loader


# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


class TheDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __getitem__(self, item):
        return self.x[item], self.y[item]
    
    def __len__(self):
        return len(self.y)


def main():
    sequence_length = 4
    input_size = 4
    # hidden_size = 128
    # num_layers = 5
    num_classes = 4
    # batch_size = 20000
    # num_epochs = 10
    # learning_rate = 0.01
    
    hyperparameter_defaults = Box(
        hidden_size=128,
        num_layers=5,
        batch_size=20000,
        learning_rate=0.01,
        num_epochs=10,
    )
    
    wandb.init(project="dm_final_project", config=hyperparameter_defaults)
    model = RNN(input_size,
                hidden_size=hyperparameter_defaults.hidden_size,
                num_layers=hyperparameter_defaults.num_layers,
                num_classes=num_classes).to(device)
    config = wandb.config
    wandb.watch(model)
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    train_loader, test_loader = get_loaders(config.batch_size)
    # Train the model
    total_step = len(train_loader)
    for epoch in range(config.num_epochs):
        losses = list()
        for i, (images, labels) in enumerate(train_loader):
            # logger.debug("labels.data.numpy().shape: %s", labels.data.numpy().shape)
            # logger.debug("images.data.numpy().shape: %s", images.data.numpy().shape)
            # Plotter(labels.data.numpy())
            images = images.reshape(-1, sequence_length, input_size).to(device)
            # logger.debug("images.shape: %s",images.shape)
            labels = labels.to(device)
            # return
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            # if i % 5 == 0:
            #     wandb.log({"Training loss (average)": np.average(losses)})
            # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
            #       .format(epoch + 1, num_epochs, i + 1, total_step, np.average(losses)))
        # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
        #       .format(epoch + 1, num_epochs, i + 1, total_step, np.average(losses)))
        wandb.log({"Training loss (average)": np.average(losses)})
        
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.reshape(-1, sequence_length, input_size).to(device)
                labels = labels.to(device)
                outputs = model(images)
                # logger.debug("outputs.data.numpy().shape: %s", outputs.data.cpu().numpy().shape)
                loss = criterion(outputs, labels)
                wandb.log({"val_loss": np.average(loss.data.cpu().numpy())})
                # logger.debug("delta: %s", delta)
                # _, predicted = torch.max(outputs.data, 1)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()
        model.train()
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, f'model_epoch_{epoch}.pt'))
    # print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
    
    # Save the model checkpoint
    # torch.save(model.state_dict(), 'model.ckpt')
    
    pass


if __name__ == '__main__':
    main()
