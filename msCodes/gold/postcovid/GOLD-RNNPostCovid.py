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
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


# 8. Define the model
class RNN(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 n_layers,
                 output_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size,
                          n_layers, batch_first=True,
                          nonlinearity='relu')
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1, N_LAGS)
        output, _ = self.rnn(x)
        output = self.fc(output[:, -1, :])
        return output


if __name__ == '__main__':
    print(torch.__version__)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 2. Define Parameters
    TICKER = 'GC=F'
    START_DATE = '2020-03-01'
    END_DATE = '2023-05-01'
    VALID_START = '2023-03-01'
    N_LAGS = 30

    # neural network =
    BATCH_SIZE = 16
    N_EPOCHS = 100

    # 3. Download and prepare the data

    df = yf.download(TICKER.strip(),
                     start=START_DATE,
                     end=END_DATE,
                     progress=False)

    print(df)
    valid_size = df.loc[VALID_START:END_DATE].shape[0]
    prices = df['Adj Close'].values.reshape(-1, 1)

    fig, ax = plt.subplots()
    ax.plot(df.index, prices)
    ax.set(title=f"{TICKER}'s Stock Price",
           xlabel='Time',
           ylabel='Price ($)')

    plt.show()

    # 4. Scale the time series of Prices

    valid_ind = len(prices) - valid_size
    minmax = MinMaxScaler(feature_range=(0, 1))

    prices_train = prices[:valid_ind - N_LAGS]
    prices_valid = prices[valid_ind - N_LAGS:]

    minmax.fit(prices_train)

    prices_train = minmax.transform(prices_train)
    prices_valid = minmax.transform(prices_valid)

    prices_scaled = np.concatenate((prices_train,
                                    prices_valid)).flatten()

    # plt.plot(prices_scaled)

    # 5. Transform the time series into input for RNN
    X, y = create_input_data(prices_scaled, N_LAGS)

    # 6. Obtain the naive forecast:
    naive_pred = prices[len(prices) - valid_size - N_LAGS: -N_LAGS]
    y_valid = prices[len(prices) - valid_size:]

    naive_mse = mean_squared_error(y_valid, naive_pred)
    naive_rmse = np.sqrt(naive_mse)
    print(f"Naive forecast - MSE: {naive_mse:.4f} | RMSE: {naive_rmse:.4f}")

    # 7. Prepare the data loader objects
    custom_set_seed(42)

    valid_ind = len(X) - valid_size

    X_tensor = torch.from_numpy(X).float().unsqueeze(2)
    y_tensor = torch.from_numpy(y).float()

    dataset = TensorDataset(X_tensor, y_tensor)

    train_dateset = Subset(dataset, list(range(valid_ind)))
    valid_dataset = Subset(dataset, list(range(valid_ind, len(X))))

    train_loader = DataLoader(dataset=train_dateset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=BATCH_SIZE)

    # Check the size of the dataset
    print(f'Size of Dataset : training - {len(train_loader.dataset)} | Validation: {len(valid_loader.dataset)}')

    # 9. Instantiate the model, the loss function and the optimizer:
    model = RNN(input_size=N_LAGS, hidden_size=6, n_layers=1, output_size=1).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 10. Train the network
    PRINT_EVERY = 10
    train_losses, valid_losses = [], []

    for epoch in range(N_EPOCHS):
        running_loss_train = 0
        running_loss_valid = 0

        model.train()

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_hat = model(x_batch)
            loss = torch.sqrt(loss_fn(y_batch, y_hat))
            loss.backward()
            optimizer.step()
            running_loss_train += loss.item() * x_batch.size(0)

        epoch_loss_train = running_loss_train / len(train_loader)
        train_losses.append(epoch_loss_train)

        with torch.no_grad():
            model.eval()
            for x_val, y_val in valid_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                y_hat = model(x_val)
                loss = torch.sqrt(loss_fn(y_val, y_hat))
                running_loss_valid += loss.item() * x_val.size(0)

            epoch_loss_valid = running_loss_valid / len(valid_loader.dataset)

            if epoch > 0 and epoch_loss_valid < min(valid_losses):
                best_epoch = epoch
                torch.save(model.state_dict(), './rnn_checkpoint.pth')

            valid_losses.append(epoch_loss_valid)

            if epoch % PRINT_EVERY == 0:
                print(f"<{epoch}> - Train. Loss: {epoch_loss_train:.4f}, \t Valid. Loss: {epoch_loss_valid:.4f}")
    print(f"Lowest loss recorded in epoch: {best_epoch}")

    # 11. Plot the losses over epochs:
    train_losses = np.array(train_losses)
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots()

    ax.plot(train_losses, color='blue', label="Training loss")
    ax.plot(valid_losses, color='red', label="Validateion Loss")

    ax.set(title="Loss over epochs",
           xlabel='Epoch',
           ylabel='Loss')
    ax.legend()

    plt.tight_layout()
    plt.show()

    # 12. Load the best model (with the lowest validation loss):
    state_dict = torch.load('rnn_checkpoint.pth')
    model.load_state_dict(state_dict)

    # 13. Obtain the predictions:
    y_pred = []

    with torch.no_grad():
        model.eval()

        for x_val, y_val in valid_loader:
            x_val = x_val.to(device)
            y_hat = model(x_val)
            y_pred.append(y_hat)

    y_pred = torch.cat(y_pred).numpy()
    y_pred = minmax.inverse_transform(y_pred).flatten()

    mape = mean_absolute_percentage_error(y_valid, y_pred)
    print(f"RNN's forecast - MAPE: {mape:.2f}%")

    rnn_mse = mean_squared_error(y_valid, y_pred)
    rnn_rmse = np.sqrt(rnn_mse)
    print(f"RNN's forecast - MSE: {rnn_mse:.4f}, RMSE: {rnn_rmse:.4f}")

    fig, ax = plt.subplots()

    ax.plot(y_valid, color='blue', label='Actual')
    ax.plot(y_pred, color='red', label='RNN')
    ax.plot(naive_pred, color='green', label='Naive')

    ax.set(title="RNN's forecasts", xlabel='Time', ylabel='Price ($)')
    ax.legend()

    plt.tight_layout()
    plt.show()
