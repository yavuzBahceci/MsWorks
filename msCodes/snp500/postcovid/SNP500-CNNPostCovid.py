import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import os
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (Dataset, TensorDataset, DataLoader, Subset)
from collections import OrderedDict
from chapter_10_utils import create_input_data, custom_set_seed

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

"""
Based on the preceding diagram, we briefly describe the elements of a typical CNN architecture:

1. Convolutional Layer: 
The goal of this layer is to apply convolutional
filtering to extract potential features

2. Pooling Layer:
This layer reduces the size of the image or series while
preserving the important characteristics identified 
by the convolutional layer

3. Fully Connected Layer:
Usually, there are a few fully connected layers at the end 
of the network to map the features extracted by the network
to classes or values
"""


# Define CNN Architecture
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


if __name__ == '__main__':
    print(torch.__version__)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 2. Define Parameters
    TICKER = '^GSPC'
    START_DATE = '2020-03-01'
    END_DATE = '2023-05-01'
    VALID_START = '2023-03-01'
    N_LAGS = 30

    # neural network =
    BATCH_SIZE = 5
    N_EPOCHS = 2000

    # 3. Download and prepare the data

    df = yf.download(TICKER.strip(),
                     start=START_DATE,
                     end=END_DATE,
                     progress=False)

    print(df)
    df = df.resample('D').last()
    df = df.dropna()
    valid_size = df.loc[VALID_START:END_DATE].shape[0]
    prices = df['Close']

    fig, ax = plt.subplots()
    ax.plot(df.index, prices)
    ax.set(title=f"{TICKER}'s Stock Price",
           xlabel='Time',
           ylabel='Price ($)')

    plt.show()

    # 4. Transform the time series into input for CNN
    X, y = create_input_data(prices, N_LAGS)

    # 5. Obtain the naive forecast
    y_valid = prices[len(prices) - valid_size:]



    # 6. Prepare the Dataloader objects
    custom_set_seed(42)

    valid_ind = len(X) - valid_size

    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float().unsqueeze(dim=1)

    dataset = TensorDataset(X_tensor, y_tensor)

    train_dataset = Subset(dataset, list(range(valid_ind)))
    valid_dataset = Subset(dataset, list(range(valid_ind, len(X))))

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=BATCH_SIZE)

    print(f"Size of the datasets - training: {len(train_loader.dataset)} | validation: {len(valid_loader.dataset)}")

    # 7. Define CNN Architecture

    model = nn.Sequential(OrderedDict([
        ('conv_1', nn.Conv1d(1, 32, 3, padding=1)),
        ('max_pool_1', nn.MaxPool1d(2)),
        ('relu_1', nn.ReLU()),
        ('flatten', Flatten()),
        ('fc_1', nn.Linear(480, 50)),
        ('relu_2', nn.ReLU()),
        ('dropout_1', nn.Dropout(0.4)),
        ('fc_2', nn.Linear(50, 1))
    ]))

    print(model)

    # 8. Instantiate the model, the loss function and the optimizer:

    model = model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 9. Train the network

    PRINT_EVERY = 50
    train_losses, valid_losses = [], []

    for epoch in range(N_EPOCHS):
        running_loss_train = 0
        running_loss_valid = 0

        model.train()

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()

            x_batch = x_batch.to(device)
            x_batch = x_batch.view(x_batch.shape[0], 1, N_LAGS)
            y_batch = y_batch.to(device)
            y_batch = y_batch.view(y_batch.shape[0], 1, 1)
            y_hat = model(x_batch).view(y_batch.shape[0], 1, 1)
            loss = torch.sqrt(loss_fn(y_batch, y_hat))
            loss.backward()
            optimizer.step()
            running_loss_train += loss.item() * x_batch.size(0)

        epoch_loss_train = running_loss_train / len(train_loader.dataset)
        train_losses.append(epoch_loss_train)

        with torch.no_grad():
            model.eval()
            for x_val, y_val in valid_loader:
                x_val = x_val.to(device)
                x_val = x_val.view(x_val.shape[0], 1, N_LAGS)
                y_val = y_val.to(device)
                y_val = y_val.view(y_val.shape[0], 1, 1)
                y_hat = model(x_val).view(y_val.shape[0], 1, 1)
                loss = torch.sqrt(loss_fn(y_val, y_hat))
                running_loss_valid += loss.item() * x_val.size(0)

            epoch_loss_valid = running_loss_valid / len(valid_loader.dataset)

            if epoch > 0 and epoch_loss_valid < min(valid_losses):
                best_epoch = epoch
                torch.save(model.state_dict(), 'cnn_checkpoint.pth')

            valid_losses.append(epoch_loss_valid)

        if epoch % PRINT_EVERY == 0:
            print(f"<{epoch}> - Train, loss: {epoch_loss_train:.6f}, \t Valid loss: {epoch_loss_valid:.6f}")

    print(f'Lowest loss recorded in epoch: {best_epoch}')

    # 10. Plot the losses over epochs:

    train_losses = np.array(train_losses)
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots()

    ax.plot(train_losses, color='blue', label='Training Loss')
    ax.plot(valid_losses, color='red', label='Validation Loss')

    ax.set(title='Loss over epochs',
           xlabel='Epoch',
           ylabel='Time')

    ax.legend()
    plt.tight_layout()
    plt.show()

    # 11. Load the best model (with the lowest validation loss):

    state_dict = torch.load('cnn_checkpoint.pth')
    model.load_state_dict(state_dict)

    # 12. Obtain the predictions:
    y_pred, y_valid = [], []

    with torch.no_grad():
        model.eval()

        for x_val, y_val in valid_loader:
            x_val = x_val.to(device)
            x_val = x_val.view(x_val.shape[0], 1, N_LAGS)
            y_pred.append(model(x_val))
            y_valid.append(y_val)

    y_pred = torch.cat(y_pred).numpy().flatten()
    y_valid = torch.cat(y_valid).numpy().flatten()

    # 13. Evaluate the predictions:

    mape = mean_absolute_percentage_error(y_valid, y_pred)
    print(f"CNN's forecast - MAPE: {mape:.2f}%")
    cnn_mse = mean_squared_error(y_valid, y_pred)
    cnn_rmse = np.sqrt(cnn_mse)
    print(f"CNN's forecast - MSE: {cnn_mse:.2f}, RMSE: {cnn_rmse:.2f}")

    fig, ax = plt.subplots()

    ax.plot(y_valid, color='blue', label='Actual')
    ax.plot(y_pred, color='red', label='Prediction')
    # ax.plot(naive_pred, color='green', label='Naive')

    ax.set(title=f"CNN's Forecasts\nMAPE: {mape:.2f}%",
           xlabel='Date',
           ylabel='Price ($)')

    ax.legend()

    plt.tight_layout()
    plt.show()
